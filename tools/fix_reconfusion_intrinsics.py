#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


DROP_TOP_LEVEL_CAMERA_KEYS = {
    "w", "h",
    "fl_x", "fl_y", "cx", "cy",
    "k1", "k2", "k3", "k4",
    "p1", "p2",
    "camera_model",
}


DROP_FRAME_DISTORTION_KEYS = {
    "k1", "k2", "k3", "k4",
    "p1", "p2",
}


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def fix_scene(scene_dir: Path, *, dry_run: bool = False, backup: bool = True) -> None:
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"Missing {transforms_path}")

    with open(transforms_path, "r") as f:
        metadata = json.load(f)

    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError(f"{scene_dir} has no frames in transforms.json")

    changed = False
    report_rows = []

    for i, frame in enumerate(frames):
        rel_path = frame.get("file_path")
        if rel_path is None:
            continue

        image_path = scene_dir / rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Frame {i} image does not exist: {image_path}")

        with Image.open(image_path) as img:
            actual_w, actual_h = img.size

        raw_w = _first_not_none(frame.get("w"), metadata.get("w"), actual_w)
        raw_h = _first_not_none(frame.get("h"), metadata.get("h"), actual_h)

        raw_fl_x = _first_not_none(frame.get("fl_x"), metadata.get("fl_x"))
        raw_fl_y = _first_not_none(frame.get("fl_y"), metadata.get("fl_y"))
        raw_cx = _first_not_none(frame.get("cx"), metadata.get("cx"))
        raw_cy = _first_not_none(frame.get("cy"), metadata.get("cy"))

        missing = [
            name
            for name, value in [
                ("w", raw_w),
                ("h", raw_h),
                ("fl_x", raw_fl_x),
                ("fl_y", raw_fl_y),
                ("cx", raw_cx),
                ("cy", raw_cy),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(f"{scene_dir} frame {i} missing camera fields: {missing}")

        raw_w = float(raw_w)
        raw_h = float(raw_h)

        if raw_w <= 0 or raw_h <= 0:
            raise ValueError(f"{scene_dir} frame {i} has invalid raw size: {raw_w}x{raw_h}")

        scale_x = float(actual_w) / raw_w
        scale_y = float(actual_h) / raw_h

        new_fl_x = float(raw_fl_x) * scale_x
        new_fl_y = float(raw_fl_y) * scale_y
        new_cx = float(raw_cx) * scale_x
        new_cy = float(raw_cy) * scale_y

        old_tuple = (
            frame.get("w"),
            frame.get("h"),
            frame.get("fl_x"),
            frame.get("fl_y"),
            frame.get("cx"),
            frame.get("cy"),
        )
        new_tuple = (
            int(actual_w),
            int(actual_h),
            float(new_fl_x),
            float(new_fl_y),
            float(new_cx),
            float(new_cy),
        )

        if old_tuple != new_tuple:
            changed = True

        frame["w"] = int(actual_w)
        frame["h"] = int(actual_h)
        frame["fl_x"] = float(new_fl_x)
        frame["fl_y"] = float(new_fl_y)
        frame["cx"] = float(new_cx)
        frame["cy"] = float(new_cy)

        # ReconfusionParser treats this path as pinhole K. Distortion is not used here.
        # Remove stale distortion values so they do not mislead later tooling.
        for key in DROP_FRAME_DISTORTION_KEYS:
            if key in frame:
                frame.pop(key, None)
                changed = True

        frame["camera_model"] = "PINHOLE"

        if i < 3:
            report_rows.append(
                {
                    "frame": i,
                    "image": rel_path,
                    "actual_size": (actual_w, actual_h),
                    "raw_size": (raw_w, raw_h),
                    "scale": (scale_x, scale_y),
                    "new_intrinsics": {
                        "fl_x": new_fl_x,
                        "fl_y": new_fl_y,
                        "cx": new_cx,
                        "cy": new_cy,
                    },
                }
            )

    # Critical: remove top-level camera keys so ReconfusionParser falls back to per-frame fields.
    for key in DROP_TOP_LEVEL_CAMERA_KEYS:
        if key in metadata:
            metadata.pop(key, None)
            changed = True

    metadata["frames"] = frames

    print(f"\nScene: {scene_dir}")
    for row in report_rows:
        print(row)

    if dry_run:
        print("Dry run only; not writing changes.")
        return

    if changed:
        if backup:
            backup_path = transforms_path.with_suffix(".json.bak")
            if not backup_path.exists():
                shutil.copy2(transforms_path, backup_path)
                print(f"Backed up original to {backup_path}")

        with open(transforms_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Wrote fixed {transforms_path}")
    else:
        print("No changes needed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=Path, default=None)
    parser.add_argument("--scenes_root", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    args = parser.parse_args()

    if args.scene_dir is None and args.scenes_root is None:
        raise ValueError("Pass either --scene_dir or --scenes_root")

    if args.scene_dir is not None:
        fix_scene(args.scene_dir, dry_run=args.dry_run, backup=not args.no_backup)

    if args.scenes_root is not None:
        scene_dirs = sorted(
            p for p in args.scenes_root.iterdir()
            if p.is_dir() and (p / "transforms.json").exists()
        )
        print(f"Found {len(scene_dirs)} scenes under {args.scenes_root}")
        for scene_dir in scene_dirs:
            fix_scene(scene_dir, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    main()