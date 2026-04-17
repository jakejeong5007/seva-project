# File: test_model_factory.py
# Description: End-to-end smoke test for model_factory.py using a real SEVA
#              backbone, real batch loading, and lightweight dummy conditioning
#              modules.

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import torch

# Make sure at least one SDPA backend is available.
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.clip_dataset import SevaClipDataset, seva_clip_collate
from train.training.conditioning import build_seva_conditioning, flatten_conditioning_for_model
from train.training.model_factory import (
    build_optimizer,
    build_seva_bundle,
    infer_batch_shape,
    smoke_test_forward,
    summarize_bundle,
)


class DummyConditioner(nn.Module):
    """Lightweight stand-in for CLIPConditioner during pipeline tests."""

    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(x.shape)}")
        n = x.shape[0]
        return torch.randn(n, self.dim, device=x.device, dtype=x.dtype)


class DummyAE(nn.Module):
    """Lightweight stand-in for the SEVA autoencoder during pipeline tests."""

    def __init__(self, latent_channels: int = 4, downsample_factor: int = 8) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)

    def encode(self, x: torch.Tensor, encoding_t: int = 1) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(x.shape)}")
        n, _, h, w = x.shape
        if h % self.downsample_factor != 0 or w % self.downsample_factor != 0:
            raise ValueError(
                f"Input spatial size {(h, w)} must be divisible by downsample factor "
                f"{self.downsample_factor}."
            )
        lh = h // self.downsample_factor
        lw = w // self.downsample_factor
        return torch.randn(
            n,
            self.latent_channels,
            lh,
            lw,
            device=x.device,
            dtype=x.dtype,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for train/model_factory.py")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Parsed SEVA-style dataset root, e.g. dl3dv_parsed/10K",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--total_frames", type=int, default=8)
    parser.add_argument(
        "--num_input_views",
        type=str,
        default="1",
        help="Comma-separated input-view options, e.g. 1,6",
    )
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_downsample_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize_intrinsics",
        action="store_true",
        default=True,
        help="Normalize K by image width/height inside the dataset.",
    )
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        default=False,
        help="Build the bundle with trainable backbone parameters and test optimizer creation.",
    )
    return parser.parse_args()


def parse_num_input_views(text: str) -> tuple[int, ...]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("--num_input_views must contain at least one integer.")
    return tuple(values)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    num_input_views = parse_num_input_views(args.num_input_views)
    dtype = resolve_dtype(args.dtype)

    ds = SevaClipDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        num_input_views=num_input_views,
        total_frames=args.total_frames,
        target_hw=(args.height, args.width),
        normalize_world=False,
        normalize_intrinsics=args.normalize_intrinsics,
        shuffle_test_frames=True,
        seed=args.seed,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=seva_clip_collate,
        num_workers=0,
    )

    batch = next(iter(loader))
    b, t, h, w = infer_batch_shape(batch)
    print("Loaded batch:")
    print(f"  imgs:        {tuple(batch['imgs'].shape)} {batch['imgs'].dtype}")
    print(f"  Ks:          {tuple(batch['Ks'].shape)} {batch['Ks'].dtype}")
    print(f"  c2ws:        {tuple(batch['c2ws'].shape)} {batch['c2ws'].dtype}")
    print(f"  input_mask:  {tuple(batch['input_mask'].shape)} {batch['input_mask'].dtype}")
    print(f"  scene_name:  {batch['scene_name']}")
    print(f"  num_input_views: {batch['num_input_views'].tolist()}")

    dummy_conditioner = DummyConditioner(dim=1024)
    dummy_ae = DummyAE(
        latent_channels=args.latent_channels,
        downsample_factor=args.latent_downsample_factor,
    )

    conditioning_output = build_seva_conditioning(
        batch=batch,
        conditioner=dummy_conditioner,
        ae=dummy_ae,
        latent_downsample_factor=args.latent_downsample_factor,
        return_unconditional=True,
    )
    flat_c = flatten_conditioning_for_model(conditioning_output.c)

    print("\nConditioning:")
    for key, value in conditioning_output.c.items():
        print(f"  c[{key!r}]: {tuple(value.shape)} {value.dtype}")
    for key, value in flat_c.items():
        print(f"  flat_c[{key!r}]: {tuple(value.shape)} {value.dtype}")

    bundle = build_seva_bundle(
        device=args.device,
        dtype=dtype,
        num_frames=t,
        build_conditioner=False,
        train_backbone=args.train_backbone,
        ae=None,
        train_ae=False,
    )

    print("\nBundle summary:")
    summary = summarize_bundle(bundle)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bundle.device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            pred = smoke_test_forward(
                bundle=bundle,
                batch=batch,
                flat_conditioning=flat_c,
            )

    expected_shape = (
        b * t,
        args.latent_channels,
        h // args.latent_downsample_factor,
        w // args.latent_downsample_factor,
    )
    assert tuple(pred.shape) == expected_shape, (
        f"Prediction shape mismatch: got {tuple(pred.shape)}, expected {expected_shape}"
    )

    print("\nForward output:")
    print(f"  pred: {tuple(pred.shape)} {pred.dtype}")
    print(f"  pred min/max: {float(pred.min()):.6f} / {float(pred.max()):.6f}")

    if args.train_backbone:
        optimizer = build_optimizer(bundle, lr=1e-4)
        print("\nOptimizer:")
        print(f"  type: {optimizer.__class__.__name__}")
        print(f"  param_groups: {len(optimizer.param_groups)}")

    print("\nmodel_factory smoke test passed")


if __name__ == "__main__":
    main()
