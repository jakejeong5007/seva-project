from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.clip_dataset import SevaClipDataset, seva_clip_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for train.data.clip_dataset.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_input_views", type=str, default="1")
    parser.add_argument("--total_frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
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
        normalize_world=False,
        normalize_intrinsics=True,
        shuffle_test_frames=True,
        seed=args.seed,
    )

    print("num scenes:", len(dataset))
    sample = dataset[0]

    print("scene_name:", sample["scene_name"])
    print(
        "imgs:",
        sample["imgs"].shape,
        sample["imgs"].dtype,
        float(sample["imgs"].min()),
        float(sample["imgs"].max()),
    )
    print("Ks:", sample["Ks"].shape, sample["Ks"].dtype)
    print("c2ws:", sample["c2ws"].shape, sample["c2ws"].dtype)
    print("input_mask:", sample["input_mask"].shape, sample["input_mask"].dtype)
    print("frame_ids:", sample["frame_ids"])
    print("num_input_views:", sample["num_input_views"])
    print("input_indices:", sample["input_indices"])
    print("target_indices:", sample["target_indices"])

    num_frames = sample["imgs"].shape[0]
    assert sample["imgs"].shape == (num_frames, 3, args.height, args.width)
    assert sample["Ks"].shape == (num_frames, 3, 3)
    assert sample["c2ws"].shape == (num_frames, 4, 4)
    assert sample["input_mask"].shape == (num_frames,)
    assert sample["input_mask"].dtype == torch.bool
    assert sample["frame_ids"].shape == (num_frames,)
    assert sample["imgs"].dtype == torch.float32
    assert sample["Ks"].dtype == torch.float32
    assert sample["c2ws"].dtype == torch.float32
    assert sample["imgs"].min() >= -1.0001
    assert sample["imgs"].max() <= 1.0001
    assert int(sample["input_mask"].sum().item()) == sample["num_input_views"]
    assert int((~sample["input_mask"]).sum().item()) >= 1

    bottom_row = sample["c2ws"][:, 3, :]
    expected = torch.tensor([0, 0, 0, 1], dtype=bottom_row.dtype)
    assert torch.allclose(bottom_row, expected.expand_as(bottom_row), atol=1e-5)

    print("single-sample checks passed")

    batch = seva_clip_collate([dataset[0], dataset[1]])
    print("batched imgs:", batch["imgs"].shape)
    print("batched Ks:", batch["Ks"].shape)
    print("batched c2ws:", batch["c2ws"].shape)
    print("batched input_mask:", batch["input_mask"].shape)

    batch_size, batch_frames = batch["imgs"].shape[:2]
    assert batch["imgs"].shape == (batch_size, batch_frames, 3, args.height, args.width)
    assert batch["Ks"].shape == (batch_size, batch_frames, 3, 3)
    assert batch["c2ws"].shape == (batch_size, batch_frames, 4, 4)
    assert batch["input_mask"].shape == (batch_size, batch_frames)

    print("collate checks passed")


if __name__ == "__main__":
    main()
