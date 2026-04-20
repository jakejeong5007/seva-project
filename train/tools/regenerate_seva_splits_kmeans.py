#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans


def load_frames(scene_dir: Path) -> list[dict]:
    with open(scene_dir / "transforms.json", "r") as f:
        tf = json.load(f)
    frames = tf["frames"]

    # Important sanity check: SEVA split IDs index this exact frame order.
    paths = [fr["file_path"] for fr in frames]
    if paths != sorted(paths):
        print(f"[WARN] {scene_dir}: transforms frame order != sorted(file_path).")
        print("       Split IDs will still index transforms.json order, but check this scene manually.")

    return frames


def camera_features(frames: list[dict]) -> np.ndarray:
    feats = []
    for fr in frames:
        c2w = np.asarray(fr["transform_matrix"], dtype=np.float64)

        center = c2w[:3, 3]

        # Match official example style: use camera position + local z direction.
        # Do not flip here unless your transforms are known to be in a different convention.
        direction = c2w[:3, 2]
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        feats.append(np.concatenate([center, direction], axis=0))

    feats = np.asarray(feats, dtype=np.float64)

    # Normalize position dimensions so KMeans is not dominated by scene scale.
    pos = feats[:, :3]
    dir_ = feats[:, 3:]

    pos_mean = pos.mean(axis=0, keepdims=True)
    pos_std = pos.std(axis=0, keepdims=True) + 1e-8
    pos_norm = (pos - pos_mean) / pos_std

    return np.concatenate([pos_norm, dir_], axis=1)


def kmeans_train_ids(frames: list[dict], num_inputs: int, seed: int = 42) -> list[int]:
    n = len(frames)
    if num_inputs >= n:
        raise ValueError(f"num_inputs={num_inputs} must be less than num_frames={n}")

    feats = camera_features(frames)

    kmeans = KMeans(n_clusters=num_inputs, random_state=seed, n_init="auto")
    kmeans.fit(feats)

    train_ids = []
    for center in kmeans.cluster_centers_:
        d = np.linalg.norm(feats - center[None, :], axis=1)
        train_ids.append(int(np.argmin(d)))

    return sorted(set(train_ids))


def make_split(
    scene_dir: Path,
    num_inputs: int,
    stride: int,
    seed: int,
    overwrite: bool,
) -> None:
    frames = load_frames(scene_dir)
    n = len(frames)

    train_ids = kmeans_train_ids(frames, num_inputs=num_inputs, seed=seed)

    # Rare case: duplicate nearest frames from KMeans centers.
    # Fill missing anchors with farthest frames from selected features.
    if len(train_ids) < num_inputs:
        feats = camera_features(frames)
        selected = set(train_ids)
        while len(train_ids) < num_inputs:
            selected_feats = feats[sorted(selected)]
            dist_to_selected = np.min(
                np.linalg.norm(feats[:, None, :] - selected_feats[None, :, :], axis=2),
                axis=1,
            )
            for idx in selected:
                dist_to_selected[idx] = -1.0
            new_idx = int(np.argmax(dist_to_selected))
            selected.add(new_idx)
            train_ids.append(new_idx)
        train_ids = sorted(train_ids)

    train_set = set(train_ids)
    remaining = [i for i in range(n) if i not in train_set]

    # This is the key difference from your current parser:
    # use a target stride like the author split, instead of all remaining frames.
    test_ids = remaining[::stride]

    split = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "num_frames": n,
        "num_inputs": num_inputs,
        "strategy": "kmeans_position_direction_inputs_strided_targets",
        "target_stride": stride,
        "seed": seed,
    }

    out = scene_dir / f"train_test_split_{num_inputs}.json"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists. Pass --overwrite to replace it.")

    with open(out, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Wrote {out}")
    print(f"  train_ids: {train_ids}")
    print(f"  first 20 test_ids: {test_ids[:20]}")
    print(f"  num_targets: {len(test_ids)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--num_inputs", type=int, nargs="+", default=[1, 6, 32])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for n in args.num_inputs:
        make_split(
            scene_dir=args.scene_dir,
            num_inputs=n,
            stride=args.stride,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()