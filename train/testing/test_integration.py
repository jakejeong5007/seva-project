from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.clip_dataset import SevaClipDataset, seva_clip_collate
from train.training.conditioning import build_seva_conditioning, flatten_conditioning_for_model


class DummyConditioner(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], self.dim, device=x.device, dtype=x.dtype)


class DummyAE(nn.Module):
    def encode(self, x: torch.Tensor, encoding_t: int = 1) -> torch.Tensor:
        del encoding_t
        return torch.randn(
            x.shape[0],
            4,
            x.shape[-2] // 8,
            x.shape[-1] // 8,
            device=x.device,
            dtype=x.dtype,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integration smoke test for dataset + conditioning."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_input_views", type=str, default="1")
    parser.add_argument("--total_frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=576)
    return parser.parse_args()


def parse_num_input_views(text: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--num_input_views must contain at least one integer.")
    return tuple(values)


def main() -> None:
    args = parse_args()
    dataset = SevaClipDataset(
        dataset_root=args.dataset_root,
        split=args.split,
        num_input_views=parse_num_input_views(args.num_input_views),
        total_frames=args.total_frames,
        target_hw=(args.height, args.width),
        normalize_intrinsics=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=seva_clip_collate,
    )
    batch = next(iter(loader))

    output = build_seva_conditioning(
        batch=batch,
        conditioner=DummyConditioner(),
        ae=DummyAE(),
        camera_scale=2.0,
        latent_downsample_factor=8,
    )

    print("additional_model_inputs:", output.additional_model_inputs)
    for key, value in output.c.items():
        print("c", key, value.shape)

    flat_c = flatten_conditioning_for_model(output.c)
    for key, value in flat_c.items():
        print("flat", key, value.shape)

    batch_size, num_frames = batch["imgs"].shape[:2]
    assert output.additional_model_inputs["num_frames"] == num_frames
    assert flat_c["crossattn"].shape[0] == batch_size * num_frames
    assert flat_c["concat"].shape[0] == batch_size * num_frames
    assert flat_c["dense_vector"].shape[0] == batch_size * num_frames

    print("integration test passed")


if __name__ == "__main__":
    main()
