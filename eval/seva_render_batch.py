#!/usr/bin/env python3
"""
eval/seva_render_batch.py

Batch-render SEVA official and/or fine-tuned checkpoints for evaluation.

This script is intentionally thin: it calls demo.py so the render path remains
the same as the official SEVA evaluation path.

Example official:
  python eval/seva_render_batch.py \
    --repo_root . \
    --scenes_root /path/to/dl3dv_parsed_fixed/11K/scenes \
    --scene_ids 000000 000001 \
    --task img2vid \
    --num_inputs 6 \
    --method_name official_p6 \
    --seed 23

Example fine-tuned:
  python eval/seva_render_batch.py \
    --repo_root . \
    --scenes_root /path/to/dl3dv_parsed_fixed/11K/scenes \
    --scene_ids 000000 000001 \
    --task img2vid \
    --num_inputs 6 \
    --method_name finetuned_p6 \
    --backbone_ckpt runs/MY_RUN/checkpoints/last.pt \
    --seed 23

You can pass --scene_list_jsonl /path/to/splits/test.jsonl instead of --scene_ids.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def read_scene_ids(args: argparse.Namespace) -> List[str]:
    scene_ids: List[str] = []

    if args.scene_ids:
        scene_ids.extend(args.scene_ids)

    if args.scene_list_txt is not None:
        with open(args.scene_list_txt, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    scene_ids.append(line)

    if args.scene_list_jsonl is not None:
        with open(args.scene_list_jsonl, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "scene_id" in rec:
                    scene_ids.append(str(rec["scene_id"]))
                elif "scene_dir" in rec:
                    scene_ids.append(Path(rec["scene_dir"]).name)
                else:
                    raise KeyError(
                        f"JSONL row must contain scene_id or scene_dir: {rec}"
                    )

    # Preserve order, remove duplicates.
    seen = set()
    out = []
    for sid in scene_ids:
        sid = str(sid)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)

    if not out:
        raise ValueError("No scenes specified. Use --scene_ids, --scene_list_txt, or --scene_list_jsonl.")

    return out


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("\n" + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=Path, default=Path("."))
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--scene_ids", nargs="*", default=None)
    parser.add_argument("--scene_list_txt", type=Path, default=None)
    parser.add_argument("--scene_list_jsonl", type=Path, default=None)

    parser.add_argument("--task", choices=["img2vid", "img2img"], default="img2vid")
    parser.add_argument("--num_inputs", type=int, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--video_save_fps", type=float, default=10.0)

    parser.add_argument("--backbone_ckpt", type=Path, default=None)
    parser.add_argument("--checkpoint_strict", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--L_short", type=int, default=None)
    parser.add_argument("--T", type=str, default=None)
    parser.add_argument("--cfg", type=str, default=None)
    parser.add_argument("--camera_scale", type=float, default=None)

    parser.add_argument("--skip_saved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    demo_py = repo_root / "demo.py"
    if not demo_py.exists():
        raise FileNotFoundError(f"Could not find demo.py at {demo_py}")

    scene_ids = read_scene_ids(args)

    for scene_id in scene_ids:
        save_subdir = f"eval_{args.method_name}_task-{args.task}_p{args.num_inputs}_seed{args.seed}"
        cmd = [
            sys.executable,
            str(demo_py),
            "--data_path",
            str(args.scenes_root),
            "--data_items",
            scene_id,
            "--task",
            args.task,
            "--num_inputs",
            str(args.num_inputs),
            "--video_save_fps",
            str(args.video_save_fps),
            "--seed",
            str(args.seed),
            "--save_subdir",
            save_subdir,
        ]

        if args.task == "img2vid":
            cmd += [
                "--replace_or_include_input",
                "True",
                "--use_traj_prior",
                "True",
                "--chunk_strategy",
                "interp",
            ]

        if args.H is not None:
            cmd += ["--H", str(args.H)]
        if args.W is not None:
            cmd += ["--W", str(args.W)]
        if args.L_short is not None:
            cmd += ["--L_short", str(args.L_short)]
        if args.T is not None:
            cmd += ["--T", args.T]
        if args.cfg is not None:
            cmd += ["--cfg", args.cfg]
        if args.camera_scale is not None:
            cmd += ["--camera_scale", str(args.camera_scale)]
        if args.skip_saved:
            cmd += ["--skip_saved", "True"]

        if args.backbone_ckpt is not None:
            cmd += ["--backbone_ckpt", str(args.backbone_ckpt)]
            cmd += ["--checkpoint_strict", "True" if args.checkpoint_strict else "False"]

        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
