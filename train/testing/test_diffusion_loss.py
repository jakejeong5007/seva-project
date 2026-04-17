# File: test_diffusion_loss.py
# Description: End-to-end smoke test for diffusion_loss.py using a real SEVA
#              backbone plus lightweight dummy conditioner and autoencoder.

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
except Exception:  # pragma: no cover
    sdpa_kernel = None
    SDPBackend = None

from train.data.clip_dataset import SevaClipDataset, seva_clip_collate
from train.training.diffusion_loss import compute_seva_diffusion_loss
from train.training.model_factory import (
    build_optimizer,
    build_seva_bundle,
    infer_batch_shape,
    summarize_bundle,
)


class DummyConditioner(nn.Module):
    """Lightweight stand-in for CLIPConditioner during loss tests."""

    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(x.shape)}")
        n = x.shape[0]
        return torch.randn(n, self.dim, device=x.device, dtype=x.dtype)


class DummyAE(nn.Module):
    """Lightweight stand-in for the SEVA autoencoder during loss tests."""

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
    parser = argparse.ArgumentParser(description="Smoke test for train/training/diffusion_loss.py")
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
    parser.add_argument("--autocast_dtype", type=str, default="bfloat16", choices=["none", "float16", "bfloat16"])
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_downsample_factor", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalize_intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize K by image width/height inside the dataset.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine_vp",
        choices=["cosine_vp", "rf_linear"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="epsilon",
        choices=["epsilon", "x0", "v", "velocity"],
    )
    parser.add_argument("--target_frame_weight", type=float, default=1.0)
    parser.add_argument("--input_frame_weight", type=float, default=0.0)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.0)
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        default=False,
        help="Enable gradients, build an optimizer, and test backward/step.",
    )
    parser.add_argument(
        "--sdpa_backend",
        type=str,
        default="math",
        choices=["auto", "math"],
        help="Force a PyTorch SDPA backend for the forward pass.",
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
        "none": None,
    }
    return mapping[name]  # type: ignore[return-value]


def make_sdpa_context(mode: str):
    if mode == "auto" or sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    if mode == "math":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def count_finite_grads(module: nn.Module) -> tuple[int, int]:
    total = 0
    finite = 0
    for param in module.parameters():
        if param.grad is None:
            continue
        total += 1
        if torch.isfinite(param.grad).all():
            finite += 1
    return finite, total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    num_input_views = parse_num_input_views(args.num_input_views)
    dtype = resolve_dtype(args.dtype)
    autocast_dtype = resolve_dtype(args.autocast_dtype)

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
    B, T, H, W = infer_batch_shape(batch)
    print("Loaded batch:")
    print(f"  imgs:        {tuple(batch['imgs'].shape)} {batch['imgs'].dtype}")
    print(f"  Ks:          {tuple(batch['Ks'].shape)} {batch['Ks'].dtype}")
    print(f"  c2ws:        {tuple(batch['c2ws'].shape)} {batch['c2ws'].dtype}")
    print(f"  input_mask:  {tuple(batch['input_mask'].shape)} {batch['input_mask'].dtype}")
    print(f"  scene_name:  {batch['scene_name']}")
    print(f"  num_input_views: {batch['num_input_views'].tolist()}")

    device = torch.device(args.device) if args.device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    dummy_ae = DummyAE(
        latent_channels=args.latent_channels,
        downsample_factor=args.latent_downsample_factor,
    )
    dummy_conditioner = DummyConditioner(dim=1024).to(device)

    bundle = build_seva_bundle(
        device=device,
        dtype=dtype,
        num_frames=T,
        build_conditioner=False,
        train_backbone=args.train_backbone,
        ae=dummy_ae,
        train_ae=False,
    )
    bundle.conditioner = dummy_conditioner

    print("\nBundle summary:")
    summary = summarize_bundle(bundle)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        try:
            torch.backends.cuda.enable_cudnn_sdp(True)
        except Exception:
            pass

    optimizer = None
    if args.train_backbone:
        optimizer = build_optimizer(bundle, lr=1e-4)
        optimizer.zero_grad(set_to_none=True)

    sdpa_context = make_sdpa_context(args.sdpa_backend)
    with sdpa_context:
        loss_out = compute_seva_diffusion_loss(
            bundle=bundle,
            batch=batch,
            device=device,
            latent_downsample_factor=args.latent_downsample_factor,
            latent_scaling_factor=1.0,
            encoding_t=1,
            sample_posterior=False,
            camera_scale=2.0,
            schedule=args.schedule,
            objective=args.objective,
            target_frame_weight=args.target_frame_weight,
            input_frame_weight=args.input_frame_weight,
            cfg_dropout_prob=args.cfg_dropout_prob,
            return_unconditional=True,
            conditioner_no_grad=True,
            encoder_no_grad=True,
            autocast_dtype=autocast_dtype,
            include_replace_in_conditioning=False,
        )

    expected_latent_hw = (H // args.latent_downsample_factor, W // args.latent_downsample_factor)
    expected_flat_shape = (B * T, args.latent_channels, *expected_latent_hw)
    expected_bt_shape = (B, T)

    assert tuple(loss_out.pred.shape) == expected_flat_shape, (
        f"pred shape mismatch: got {tuple(loss_out.pred.shape)}, expected {expected_flat_shape}"
    )
    assert tuple(loss_out.target.shape) == expected_flat_shape, (
        f"target shape mismatch: got {tuple(loss_out.target.shape)}, expected {expected_flat_shape}"
    )
    assert tuple(loss_out.latents.shape[:2]) == expected_bt_shape
    assert tuple(loss_out.noisy_latents.shape[:2]) == expected_bt_shape
    assert tuple(loss_out.noise.shape[:2]) == expected_bt_shape
    assert tuple(loss_out.mse_per_item.shape) == expected_bt_shape
    assert tuple(loss_out.weights.shape) == expected_bt_shape
    assert tuple(loss_out.timesteps.shape) == expected_bt_shape
    assert tuple(loss_out.alpha.shape) == expected_bt_shape
    assert tuple(loss_out.sigma.shape) == expected_bt_shape
    assert torch.isfinite(loss_out.loss).item(), "loss is not finite"
    assert torch.isfinite(loss_out.pred).all().item(), "pred contains non-finite values"
    assert torch.isfinite(loss_out.target).all().item(), "target contains non-finite values"

    print("\nLoss output:")
    print(f"  loss:        {float(loss_out.loss):.6f}")
    print(f"  pred:        {tuple(loss_out.pred.shape)} {loss_out.pred.dtype}")
    print(f"  target:      {tuple(loss_out.target.shape)} {loss_out.target.dtype}")
    print(f"  latents:     {tuple(loss_out.latents.shape)} {loss_out.latents.dtype}")
    print(f"  noisy:       {tuple(loss_out.noisy_latents.shape)} {loss_out.noisy_latents.dtype}")
    print(f"  mse_per_item:{tuple(loss_out.mse_per_item.shape)} {loss_out.mse_per_item.dtype}")
    print(f"  weights:     {tuple(loss_out.weights.shape)} {loss_out.weights.dtype}")
    print(f"  timesteps:   {tuple(loss_out.timesteps.shape)} {loss_out.timesteps.dtype}")
    print(f"  alpha range: {float(loss_out.alpha.min()):.6f} .. {float(loss_out.alpha.max()):.6f}")
    print(f"  sigma range: {float(loss_out.sigma.min()):.6f} .. {float(loss_out.sigma.max()):.6f}")
    print(f"  pred min/max:{float(loss_out.pred.min()):.6f} / {float(loss_out.pred.max()):.6f}")

    print("\nConditioning keys used inside loss:")
    for key, value in loss_out.flat_conditioning.items():
        print(f"  flat_c[{key!r}]: {tuple(value.shape)} {value.dtype}")

    if args.train_backbone:
        loss_out.loss.backward()
        finite_grads, total_grads = count_finite_grads(bundle.backbone)
        print("\nBackward check:")
        print(f"  backbone grads finite: {finite_grads}/{total_grads}")
        assert total_grads > 0, "No backbone gradients were produced."
        assert finite_grads == total_grads, "Some backbone gradients are non-finite."
        assert optimizer is not None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print("  optimizer step: ok")

    print("\ndiffusion_loss smoke test passed")


if __name__ == "__main__":
    main()
