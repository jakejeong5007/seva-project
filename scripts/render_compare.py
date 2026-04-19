from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seva.data_io import ReconfusionParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run official-vs-finetuned SEVA renders on the same source scene. "
            "Supports both scene-folder img2trajvid and single-image "
            "img2trajvid_s-prob comparisons."
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
        "--task",
        type=str,
        default="img2trajvid",
        choices=["img2trajvid", "img2trajvid_s-prob"],
        help=(
            "img2trajvid compares the full scene-folder render; "
            "img2trajvid_s-prob compares a single-image baseline extracted from the scene."
        ),
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
    parser.add_argument(
        "--source_view_rank",
        type=int,
        default=0,
        help=(
            "For img2trajvid_s-prob, choose which training-view index from the "
            "selected split to use as the single-image source. Default: 0."
        ),
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--l_short", type=int, default=576)
    parser.add_argument(
        "--traj_prior",
        type=str,
        default="orbit",
        help="Camera motion preset used for img2trajvid_s-prob.",
    )
    parser.add_argument(
        "--num_targets",
        type=int,
        default=111,
        help="Number of generated target frames for img2trajvid_s-prob.",
    )
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
        help="Output prefix under work_dirs/demo. Defaults to compare_<scene>_<preset>_<task>.",
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

    return max(available)


def author_style_options(num_inputs: int) -> tuple[str, str]:
    if num_inputs <= 8:
        return "3.0,2.0", "interp-gt"
    return "3.0", "interp"


def resolve_single_image_source(
    *,
    scene_dir: Path,
    num_inputs: int,
    source_view_rank: int,
) -> tuple[Path, int, list[int]]:
    parser = ReconfusionParser(str(scene_dir), normalize=False)
    split_dict = parser.splits_per_num_input_frames[int(num_inputs)]
    train_ids = [int(x) for x in split_dict["train_ids"]]
    if not train_ids:
        raise ValueError(
            f"Scene split train_test_split_{num_inputs}.json has no train_ids."
        )
    if not (0 <= source_view_rank < len(train_ids)):
        raise ValueError(
            f"source_view_rank={source_view_rank} is out of range for "
            f"{len(train_ids)} available training views: {train_ids}"
        )
    frame_id = train_ids[source_view_rank]
    return Path(parser.image_paths[frame_id]), frame_id, train_ids


def build_demo_command(
    *,
    python_exe: str,
    task: str,
    scene_dir: Path,
    scene_name: str,
    num_inputs: int,
    seed: int,
    l_short: int,
    save_subdir: str,
    backbone_ckpt: Path | None,
    traj_prior: str,
    num_targets: int,
    source_view_rank: int,
) -> tuple[list[str], dict[str, object]]:
    metadata: dict[str, object] = {
        "scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "task": task,
        "num_inputs": num_inputs,
    }

    if task == "img2trajvid":
        cfg, chunk_strategy = author_style_options(num_inputs)
        command = [
            python_exe,
            "demo.py",
            "--data_path",
            str(scene_dir.parent),
            "--data_items",
            scene_name,
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
        metadata["cfg"] = cfg
        metadata["chunk_strategy"] = chunk_strategy
    elif task == "img2trajvid_s-prob":
        source_image, source_frame_id, train_ids = resolve_single_image_source(
            scene_dir=scene_dir,
            num_inputs=num_inputs,
            source_view_rank=source_view_rank,
        )
        command = [
            python_exe,
            "demo.py",
            "--data_path",
            str(source_image.parent),
            "--data_items",
            source_image.name,
            "--task",
            "img2trajvid_s-prob",
            "--replace_or_include_input",
            "True",
            "--traj_prior",
            traj_prior,
            "--cfg",
            "4.0,2.0",
            "--guider_types",
            "1,2",
            "--num_targets",
            str(num_targets),
            "--L_short",
            str(l_short),
            "--use_traj_prior",
            "True",
            "--chunk_strategy",
            "interp",
            "--seed",
            str(seed),
            "--save_subdir",
            save_subdir,
        ]
        metadata.update(
            {
                "source_image": str(source_image),
                "source_frame_id": source_frame_id,
                "source_view_rank": source_view_rank,
                "available_train_ids": train_ids,
                "traj_prior": traj_prior,
                "num_targets": num_targets,
            }
        )
    else:
        raise ValueError(f"Unsupported task: {task!r}")

    if backbone_ckpt is not None:
        command.extend(
            [
                "--backbone_ckpt",
                str(backbone_ckpt),
                "--checkpoint_strict",
                "True",
            ]
        )
    return command, metadata


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
    task_slug = args.task.replace("-", "_")
    save_prefix = args.save_prefix or f"compare_{args.scene}_{args.preset}_{task_slug}"

    print(f"Scene: {scene_dir}")
    print(f"Task: {args.task}")
    print(f"Available splits: {splits}")
    print(f"Selected num_inputs: {num_inputs}")

    official_command, official_metadata = build_demo_command(
        python_exe=args.python_exe,
        task=args.task,
        scene_dir=scene_dir,
        scene_name=args.scene,
        num_inputs=num_inputs,
        seed=args.seed,
        l_short=args.l_short,
        save_subdir=f"{save_prefix}_official",
        backbone_ckpt=None,
        traj_prior=args.traj_prior,
        num_targets=args.num_targets,
        source_view_rank=args.source_view_rank,
    )
    if "source_image" in official_metadata:
        print(
            "Selected source image: "
            f"{official_metadata['source_image']} "
            f"(frame_id={official_metadata['source_frame_id']}, "
            f"rank={official_metadata['source_view_rank']})"
        )
    run_command(official_command, cwd=repo_root, dry_run=args.dry_run)

    if args.backbone_ckpt is None:
        print("No --backbone_ckpt provided, skipping finetuned render.")
        return

    finetuned_command, _ = build_demo_command(
        python_exe=args.python_exe,
        task=args.task,
        scene_dir=scene_dir,
        scene_name=args.scene,
        num_inputs=num_inputs,
        seed=args.seed,
        l_short=args.l_short,
        save_subdir=f"{save_prefix}_finetuned",
        backbone_ckpt=args.backbone_ckpt.resolve(),
        traj_prior=args.traj_prior,
        num_targets=args.num_targets,
        source_view_rank=args.source_view_rank,
    )
    run_command(finetuned_command, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
