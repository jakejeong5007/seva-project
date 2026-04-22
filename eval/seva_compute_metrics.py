#!/usr/bin/env python3
"""
eval/seva_compute_metrics.py

Compute paper-aligned image metrics for SEVA renders:
  - PSNR ↑
  - SSIM ↑
  - LPIPS ↓

For img2vid:
  demo.py writes final outputs in:
    work_dirs/demo/img2vid/<save_subdir>/<scene_id>/transforms.json

  The final output transforms.json contains one image path per original scene
  frame after demo.py re-inserts the input frames. This script computes metrics
  on TARGET frames only, i.e. all frames except train_ids from
  train_test_split_<P>.json. This matches the trajectory-NVS framing where
  input frames are conditioning views and the remaining frames are generated
  targets.

For img2img:
  metrics are computed over test_ids from train_test_split_<P>.json.

Dependencies:
  pip install pillow numpy pandas scikit-image torch torchvision lpips tqdm
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from tqdm import tqdm


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def resolve_path_maybe_relative(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return base_dir / p


def resize_cover_center_crop(img: Image.Image, out_w: int, out_h: int) -> Image.Image:
    """Match SEVA default tuple-size path: resize to cover, then center crop."""
    w, h = img.size
    scale = max(out_w / w, out_h / h)
    rw = int(math.ceil(w * scale))
    rh = int(math.ceil(h * scale))
    img = img.resize((rw, rh), Image.Resampling.BICUBIC)
    left = max(0, (rw - out_w) // 2)
    top = max(0, (rh - out_h) // 2)
    return img.crop((left, top, left + out_w, top + out_h))


def resize_short_then_center_crop(img: Image.Image, short_side: int, out_w: int, out_h: int) -> Image.Image:
    """Useful for benchmark entries that use --L_short followed by center-crop postprocessing."""
    w, h = img.size
    if w < h:
        rw = short_side
        rh = int(round(short_side * h / w))
    else:
        rh = short_side
        rw = int(round(short_side * w / h))
    img = img.resize((rw, rh), Image.Resampling.BICUBIC)
    left = max(0, (rw - out_w) // 2)
    top = max(0, (rh - out_h) // 2)
    return img.crop((left, top, left + out_w, top + out_h))


def load_rgb_float(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def preprocess_gt(path: Path, mode: str, out_w: int, out_h: int, l_short: Optional[int]) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if mode == "center_crop":
        img = resize_cover_center_crop(img, out_w, out_h)
    elif mode == "none":
        if img.size != (out_w, out_h):
            img = img.resize((out_w, out_h), Image.Resampling.BICUBIC)
    elif mode == "l_short_center_crop":
        if l_short is None:
            raise ValueError("--l_short is required for --gt_preprocess l_short_center_crop")
        img = resize_short_then_center_crop(img, l_short, out_w, out_h)
    else:
        raise ValueError(f"Unknown --gt_preprocess {mode}")
    return np.asarray(img, dtype=np.float32) / 255.0


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(ssim_fn(gt, pred, channel_axis=2, data_range=1.0))


class LPIPSComputer:
    def __init__(self, enabled: bool, device: str):
        self.enabled = enabled
        self.device = device
        self.model = None
        if enabled:
            import torch
            import lpips
            self.torch = torch
            self.model = lpips.LPIPS(net="alex").to(device).eval()

    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
        if not self.enabled:
            return None
        torch = self.torch
        with torch.no_grad():
            p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
            g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
            p = p * 2.0 - 1.0
            g = g * 2.0 - 1.0
            return float(self.model(p, g).item())


def get_scene_ids(args: argparse.Namespace) -> List[str]:
    out = []
    if args.scene_ids:
        out.extend(args.scene_ids)
    if args.scene_list_jsonl:
        with open(args.scene_list_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "scene_id" in rec:
                    out.append(str(rec["scene_id"]))
                elif "scene_dir" in rec:
                    out.append(Path(rec["scene_dir"]).name)
    if not out:
        # Infer from render root directories.
        out = sorted([p.name for p in args.render_root.iterdir() if p.is_dir()])
    return sorted(dict.fromkeys(out))


def get_metric_indices(task: str, num_frames: int, split: Dict) -> List[int]:
    train_ids = set(int(x) for x in split["train_ids"])
    if task == "img2vid":
        return [i for i in range(num_frames) if i not in train_ids]
    if task == "img2img":
        return [int(i) for i in split["test_ids"]]
    raise ValueError(f"Unsupported task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--render_root", type=Path, required=True)
    parser.add_argument("--scene_ids", nargs="*", default=None)
    parser.add_argument("--scene_list_jsonl", type=Path, default=None)

    parser.add_argument("--task", choices=["img2vid", "img2img"], default="img2vid")
    parser.add_argument("--num_inputs", type=int, required=True)
    parser.add_argument("--gt_preprocess", choices=["center_crop", "none", "l_short_center_crop"], default="center_crop")
    parser.add_argument("--l_short", type=int, default=None)

    parser.add_argument("--compute_lpips", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lpips_device", type=str, default="cuda")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_summary", type=Path, default=None)
    args = parser.parse_args()

    lpips_metric = LPIPSComputer(enabled=args.compute_lpips, device=args.lpips_device)

    scene_ids = get_scene_ids(args)
    rows = []

    for scene_id in tqdm(scene_ids, desc="scenes"):
        scene_dir = args.scenes_root / scene_id
        render_dir = args.render_root / scene_id

        scene_tf_path = scene_dir / "transforms.json"
        render_tf_path = render_dir / "transforms.json"
        split_path = scene_dir / f"train_test_split_{args.num_inputs}.json"

        if not render_tf_path.exists():
            print(f"[WARN] missing render transforms: {render_tf_path}")
            continue
        if not split_path.exists():
            print(f"[WARN] missing split: {split_path}")
            continue

        scene_tf = load_json(scene_tf_path)
        render_tf = load_json(render_tf_path)
        split = load_json(split_path)

        scene_frames = scene_tf["frames"]
        render_frames = render_tf["frames"]

        n = min(len(scene_frames), len(render_frames))
        metric_indices = [i for i in get_metric_indices(args.task, n, split) if i < n]

        for frame_idx in tqdm(metric_indices, desc=scene_id, leave=False):
            gt_path = resolve_path_maybe_relative(scene_frames[frame_idx]["file_path"], scene_dir)
            pred_path = resolve_path_maybe_relative(render_frames[frame_idx]["file_path"], render_dir)

            if not pred_path.exists():
                print(f"[WARN] missing pred: {pred_path}")
                continue

            pred = load_rgb_float(pred_path)
            out_h, out_w = pred.shape[:2]
            gt = preprocess_gt(
                gt_path,
                mode=args.gt_preprocess,
                out_w=out_w,
                out_h=out_h,
                l_short=args.l_short,
            )

            if gt.shape != pred.shape:
                raise ValueError(f"Shape mismatch {scene_id} frame {frame_idx}: pred={pred.shape}, gt={gt.shape}")

            row = {
                "scene_id": scene_id,
                "frame_idx": frame_idx,
                "pred_path": str(pred_path),
                "gt_path": str(gt_path),
                "psnr": psnr(pred, gt),
                "ssim": ssim(pred, gt),
            }
            lp = lpips_metric(pred, gt)
            if lp is not None:
                row["lpips"] = lp
            rows.append(row)

    if not rows:
        raise RuntimeError("No metric rows were computed.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for key in ["psnr", "ssim", "lpips"]:
        vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
        if vals:
            summary[key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "n": int(len(vals)),
            }

    per_scene = {}
    for scene_id in sorted(set(r["scene_id"] for r in rows)):
        scene_rows = [r for r in rows if r["scene_id"] == scene_id]
        per_scene[scene_id] = {}
        for key in ["psnr", "ssim", "lpips"]:
            vals = [float(r[key]) for r in scene_rows if key in r and np.isfinite(float(r[key]))]
            if vals:
                per_scene[scene_id][key] = float(np.mean(vals))
        per_scene[scene_id]["num_frames"] = len(scene_rows)

    summary_obj = {"overall": summary, "per_scene": per_scene}
    out_summary = args.out_summary or args.out_csv.with_suffix(".summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary_obj, f, indent=2)

    print("\nOverall summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote frame metrics: {args.out_csv}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
