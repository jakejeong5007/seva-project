from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run author-style SEVA demo renders for an official-vs-finetuned "
            "comparison on the same scene."
        )
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Parent directory that contains scene folders parsable by ReconfusionParser.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="Scene folder name under --data_path, for example garden_flythrough or 000247.",
    )
    parser.add_argument(
        "--backbone_ckpt",
        type=Path,
        default=None,
        help="Training checkpoint used for the finetuned render.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="quality",
        choices=["quality", "proof"],
        help=(
            "quality: use the strongest available split with author-style demo "
            "settings; proof: prefer the smallest available split to stay closer "
            "to the 1-view training regime."
        ),
    )
    parser.add_argument(
        "--num_inputs",
        type=int,
        default=None,
        help="Override the scene split to use. Must match an available train_test_split_*.json.",
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--l_short", type=int, default=576)
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke demo.py.",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default=None,
        help="Output prefix under work_dirs/demo. Defaults to compare_<scene>_<preset>.",
    )
    parser.add_argument(
        "--dry_run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def available_num_inputs(scene_dir: Path) -> list[int]:
    values: list[int] = []
    for split_file in sorted(scene_dir.glob("train_test_split_*.json")):
        suffix = split_file.stem.removeprefix("train_test_split_")
        try:
            values.append(int(suffix))
        except ValueError:
            continue
    return sorted(set(values))


def choose_num_inputs(
    available: Iterable[int], *, requested: int | None, preset: str
) -> int:
    available = sorted(set(int(x) for x in available))
    if not available:
        raise ValueError("Scene has no train_test_split_*.json files.")

    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested num_inputs={requested} is unavailable. "
                f"Available splits: {available}"
            )
        return requested

    if preset == "proof":
        if 1 in available:
            return 1
        return min(available)

    # quality preset
    return max(available)


def author_style_options(num_inputs: int) -> tuple[str, str]:
    if num_inputs <= 8:
        return "3.0,2.0", "interp-gt"
    return "3.0", "interp"


def build_demo_command(
    *,
    python_exe: str,
    repo_root: Path,
    data_path: Path,
    scene: str,
    num_inputs: int,
    seed: int,
    l_short: int,
    save_subdir: str,
    backbone_ckpt: Path | None,
) -> list[str]:
    cfg, chunk_strategy = author_style_options(num_inputs)
    command = [
        python_exe,
        "demo.py",
        "--data_path",
        str(data_path),
        "--data_items",
        scene,
        "--task",
        "img2trajvid",
        "--num_inputs",
        str(num_inputs),
        "--cfg",
        cfg,
        "--L_short",
        str(l_short),
        "--use_traj_prior",
        "True",
        "--chunk_strategy",
        chunk_strategy,
        "--seed",
        str(seed),
        "--save_subdir",
        save_subdir,
    ]
    if backbone_ckpt is not None:
        command.extend(
            [
                "--backbone_ckpt",
                str(backbone_ckpt),
                "--checkpoint_strict",
                "True",
            ]
        )
    return command


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scene_dir = (args.data_path / args.scene).resolve()
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    splits = available_num_inputs(scene_dir)
    num_inputs = choose_num_inputs(
        splits, requested=args.num_inputs, preset=args.preset
    )
    save_prefix = args.save_prefix or f"compare_{args.scene}_{args.preset}"

    print(f"Scene: {scene_dir}")
    print(f"Available splits: {splits}")
    print(f"Selected num_inputs: {num_inputs}")

    official_command = build_demo_command(
        python_exe=args.python_exe,
        repo_root=repo_root,
        data_path=args.data_path.resolve(),
        scene=args.scene,
        num_inputs=num_inputs,
        seed=args.seed,
        l_short=args.l_short,
        save_subdir=f"{save_prefix}_official",
        backbone_ckpt=None,
    )
    run_command(official_command, cwd=repo_root, dry_run=args.dry_run)

    if args.backbone_ckpt is None:
        print("No --backbone_ckpt provided, skipping finetuned render.")
        return

    finetuned_command = build_demo_command(
        python_exe=args.python_exe,
        repo_root=repo_root,
        data_path=args.data_path.resolve(),
        scene=args.scene,
        num_inputs=num_inputs,
        seed=args.seed,
        l_short=args.l_short,
        save_subdir=f"{save_prefix}_finetuned",
        backbone_ckpt=args.backbone_ckpt.resolve(),
    )
    run_command(finetuned_command, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
