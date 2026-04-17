#!/usr/bin/env python3
"""
build_dl3dv_parsed_seva.py

Parse hashed DL3DV scene zip files into a normalized parser-level dataset while
also writing per-scene train/test split files expected by SEVA's
ReconfusionParser.

Output structure:

<output_dir>/
├── dataset_meta.json
├── zip_index.json
├── parser_warnings.log
├── splits/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── scenes/
    ├── 000000/
    │   ├── scene_meta.json
    │   ├── transforms.json
    │   ├── cameras.npz
    │   ├── train_test_split_1.json
    │   ├── train_test_split_6.json
    │   ├── train_test_split_32.json
    │   └── images/
    │       ├── 000000.jpg
    │       ├── 000001.jpg
    │       └── ...
    └── ...

SEVA-compatible scene requirements are the important part:
- images/
- transforms.json
- train_test_split_*.json  (with keys: train_ids, test_ids)

Requirements:
    pip install pillow numpy
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Tuple
from zipfile import ZipFile

import numpy as np
from PIL import Image


DEFAULT_SCENE_NUM_INPUTS = [1, 6, 32]


@dataclass
class SceneAssignment:
    scene_id: str
    zip_index: int
    zip_name: str
    zip_path: Path
    split: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def clear_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def is_zip_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def log_warning(log_path: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] WARNING: {message}"
    print(line)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def find_single_transforms_path(zf: ZipFile) -> str:
    candidates = [
        name for name in zf.namelist()
        if name.endswith("transforms.json") and not name.endswith("/")
    ]
    if not candidates:
        raise FileNotFoundError("No transforms.json found inside zip.")
    if len(candidates) > 1:
        candidates.sort(key=lambda s: (len(PurePosixPath(s).parts), s))
    return candidates[0]


def load_json_from_zip(zf: ZipFile, member: str) -> Dict[str, Any]:
    with zf.open(member) as f:
        return json.load(f)


def build_intrinsic_matrix(
    fl_x: float,
    fl_y: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    return np.array(
        [
            [fl_x, 0.0, cx],
            [0.0, fl_y, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def safe_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required camera field: {name}")
    return float(value)


def compute_split_counts(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0

    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    while n_test < 0:
        if n_train >= n_val and n_train > 0:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        n_test = n - n_train - n_val

    return n_train, n_val, n_test


def assign_scene_splits(
    zip_paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[SceneAssignment]:
    rng = random.Random(seed)

    indexed = list(enumerate(sorted(zip_paths)))
    shuffled = indexed[:]
    rng.shuffle(shuffled)

    n_train, n_val, n_test = compute_split_counts(
        n=len(shuffled),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    split_labels = (
        ["train"] * n_train +
        ["val"] * n_val +
        ["test"] * n_test
    )

    assignments_by_zip_index: Dict[int, SceneAssignment] = {}
    for (zip_index, zip_path), split in zip(shuffled, split_labels):
        scene_id = f"{zip_index:06d}"
        assignments_by_zip_index[zip_index] = SceneAssignment(
            scene_id=scene_id,
            zip_index=zip_index,
            zip_name=zip_path.name,
            zip_path=zip_path,
            split=split,
        )

    return [assignments_by_zip_index[i] for i, _ in indexed]


def extract_image_as_jpg(
    zf: ZipFile,
    member: str,
    out_path: Path,
    jpg_quality: int,
) -> Tuple[int, int]:
    with zf.open(member) as f:
        img = Image.open(BytesIO(f.read()))
        img = img.convert("RGB")
        width, height = img.size
        img.save(out_path, format="JPEG", quality=jpg_quality)
    return width, height


def resolve_zip_member(
    zf: ZipFile,
    transforms_member: str,
    file_path: str,
) -> str:
    """
    Resolve a frame file_path from transforms.json against the actual zip contents.

    Tries:
    1. file_path as-is
    2. parent(transforms.json) / file_path
    3. top-level folder containing transforms.json / file_path
    4. unique basename match fallback
    """
    names = set(zf.namelist())

    file_path_pp = PurePosixPath(file_path)
    transforms_pp = PurePosixPath(transforms_member)
    transforms_parent = transforms_pp.parent

    candidates: List[str] = []

    candidates.append(file_path_pp.as_posix())

    if str(transforms_parent) not in ("", "."):
        candidates.append((transforms_parent / file_path_pp).as_posix())

    parts = transforms_pp.parts
    if len(parts) >= 2:
        top_dir = PurePosixPath(parts[0])
        candidates.append((top_dir / file_path_pp).as_posix())

    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate in names:
            return candidate

    target_name = file_path_pp.name
    basename_matches = [
        name for name in names
        if not name.endswith("/") and PurePosixPath(name).name == target_name
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]

    raise FileNotFoundError(
        f"Could not resolve file_path={file_path!r} from transforms={transforms_member!r}. "
        f"Tried: {unique_candidates}. "
        f"Basename matches: {basename_matches[:10]}"
    )


def parse_scene_num_inputs(value: str) -> List[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--scene_num_inputs must not be empty")

    result: List[int] = []
    for item in items:
        try:
            parsed = int(item)
        except ValueError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid scene input count {item!r}; expected comma-separated integers"
            ) from e
        if parsed <= 0:
            raise argparse.ArgumentTypeError(
                f"Scene input counts must be positive, got {parsed}"
            )
        result.append(parsed)
    return sorted(set(result))


def select_evenly_spaced_indices(num_frames: int, num_select: int) -> List[int]:
    """
    Select unique, sorted frame indices spread across the full scene.

    This is more reliable than directly rounding linspace because it guarantees
    uniqueness for num_select <= num_frames.
    """
    if num_select <= 0:
        raise ValueError("num_select must be > 0")
    if num_select > num_frames:
        raise ValueError("num_select cannot exceed num_frames")
    if num_select == 1:
        return [0]
    if num_select == num_frames:
        return list(range(num_frames))

    targets = np.linspace(0.0, float(num_frames - 1), num_select)
    available = set(range(num_frames))
    selected: List[int] = []

    for target in targets:
        best_idx = min(available, key=lambda idx: (abs(idx - target), idx))
        selected.append(best_idx)
        available.remove(best_idx)

    selected.sort()
    return selected


def build_scene_train_test_splits(
    scene_dir: Path,
    num_frames: int,
    scene_num_inputs: Iterable[int],
    log_path: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if num_frames < 2:
        log_warning(
            log_path,
            f"{scene_dir.name}: only {num_frames} valid frame(s); cannot create SEVA train/test splits.",
        )
        return records

    all_ids = list(range(num_frames))

    for requested_num_inputs in scene_num_inputs:
        if requested_num_inputs >= num_frames:
            log_warning(
                log_path,
                f"{scene_dir.name}: cannot create train_test_split_{requested_num_inputs}.json "
                f"because num_frames={num_frames}. Need at least num_inputs + 1 frames.",
            )
            continue

        train_ids = select_evenly_spaced_indices(num_frames, requested_num_inputs)
        train_ids_set = set(train_ids)
        test_ids = [idx for idx in all_ids if idx not in train_ids_set]

        split_obj = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "num_frames": num_frames,
            "num_inputs": requested_num_inputs,
            "strategy": "evenly_spaced_inputs_rest_as_targets",
        }
        split_path = scene_dir / f"train_test_split_{requested_num_inputs}.json"
        write_json(split_path, split_obj)

        records.append(
            {
                "file": split_path.name,
                "num_inputs": requested_num_inputs,
                "num_targets": len(test_ids),
                "train_ids": train_ids,
            }
        )

    return records


def parse_single_scene(
    assignment: SceneAssignment,
    out_root: Path,
    jpg_quality: int,
    scene_num_inputs: Iterable[int],
    log_path: Path,
) -> Dict[str, Any] | None:
    scene_dir = out_root / "scenes" / assignment.scene_id
    images_dir = scene_dir / "images"
    ensure_dir(images_dir)

    with ZipFile(assignment.zip_path, "r") as zf:
        try:
            transforms_member = find_single_transforms_path(zf)
            raw_transforms = load_json_from_zip(zf, transforms_member)
        except Exception as e:
            log_warning(
                log_path,
                f"{assignment.zip_name}: failed to load transforms.json: {e}",
            )
            return None

        raw_inner_dir = PurePosixPath(transforms_member).parent.as_posix()
        raw_frames = raw_transforms.get("frames", [])
        if not raw_frames:
            log_warning(
                log_path,
                f"{assignment.zip_name}: transforms.json has no frames. Skipping scene.",
            )
            return None

        raw_w = raw_transforms.get("w")
        raw_h = raw_transforms.get("h")

        normalized_frames: List[Dict[str, Any]] = []
        frame_ids: List[int] = []
        colmap_im_ids: List[int] = []
        intrinsics: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        distortions: List[np.ndarray] = []

        output_w: int | None = None
        output_h: int | None = None

        for frame in raw_frames:
            original_rel_path = frame.get("file_path")
            if not original_rel_path:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: frame missing file_path. Skipping frame.",
                )
                continue

            try:
                zip_image_member = resolve_zip_member(
                    zf=zf,
                    transforms_member=transforms_member,
                    file_path=str(original_rel_path),
                )
            except Exception as e:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: could not resolve frame path "
                    f"{original_rel_path!r}: {e}. Skipping frame.",
                )
                continue

            out_name = f"{len(normalized_frames):06d}.jpg"
            out_image_path = images_dir / out_name

            try:
                width, height = extract_image_as_jpg(
                    zf=zf,
                    member=zip_image_member,
                    out_path=out_image_path,
                    jpg_quality=jpg_quality,
                )
            except Exception as e:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: failed to extract image "
                    f"{zip_image_member!r}: {e}. Skipping frame.",
                )
                continue

            if output_w is None or output_h is None:
                output_w, output_h = width, height
            elif width != output_w or height != output_h:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: inconsistent frame size for "
                    f"{original_rel_path!r}. Expected ({output_w}, {output_h}), "
                    f"got ({width}, {height}). Skipping frame.",
                )
                try:
                    out_image_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            try:
                fl_x = safe_float(frame.get("fl_x", raw_transforms.get("fl_x")), "fl_x")
                fl_y = safe_float(frame.get("fl_y", raw_transforms.get("fl_y")), "fl_y")
                cx = safe_float(frame.get("cx", raw_transforms.get("cx")), "cx")
                cy = safe_float(frame.get("cy", raw_transforms.get("cy")), "cy")

                k1 = float(frame.get("k1", raw_transforms.get("k1", 0.0)))
                k2 = float(frame.get("k2", raw_transforms.get("k2", 0.0)))
                p1 = float(frame.get("p1", raw_transforms.get("p1", 0.0)))
                p2 = float(frame.get("p2", raw_transforms.get("p2", 0.0)))

                intrinsic = build_intrinsic_matrix(fl_x, fl_y, cx, cy)

                transform_matrix = np.asarray(frame["transform_matrix"], dtype=np.float32)
                if transform_matrix.shape != (4, 4):
                    raise ValueError(
                        f"transform_matrix shape {transform_matrix.shape} != (4, 4)"
                    )
            except Exception as e:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: invalid camera data for "
                    f"{original_rel_path!r}: {e}. Skipping frame.",
                )
                try:
                    out_image_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            new_idx = len(normalized_frames)

            frame_ids.append(new_idx)
            colmap_im_ids.append(int(frame.get("colmap_im_id", -1)))
            intrinsics.append(intrinsic)
            extrinsics.append(transform_matrix)
            distortions.append(np.array([k1, k2, p1, p2], dtype=np.float32))

            normalized_frame = dict(frame)
            normalized_frame["file_path"] = f"images/{out_name}"
            normalized_frame["original_file_path"] = str(original_rel_path)
            normalized_frame["resolved_zip_member"] = zip_image_member
            normalized_frame["normalized_frame_id"] = new_idx
            normalized_frames.append(normalized_frame)

        if len(normalized_frames) < 2:
            log_warning(
                log_path,
                f"{assignment.zip_name}: only {len(normalized_frames)} valid frame(s) remained after filtering. "
                f"Skipping scene because SEVA train/test split files require at least 2 frames.",
            )
            return None

        normalized_transforms = {
            k: v for k, v in raw_transforms.items() if k != "frames"
        }
        normalized_transforms["frames"] = normalized_frames
        write_json(scene_dir / "transforms.json", normalized_transforms)

        intrinsics_np = np.stack(intrinsics, axis=0).astype(np.float32)
        extrinsics_np = np.stack(extrinsics, axis=0).astype(np.float32)
        frame_ids_np = np.asarray(frame_ids, dtype=np.int32)
        colmap_im_ids_np = np.asarray(colmap_im_ids, dtype=np.int32)
        distortions_np = np.stack(distortions, axis=0).astype(np.float32)

        assert output_w is not None and output_h is not None
        np.savez_compressed(
            scene_dir / "cameras.npz",
            frame_ids=frame_ids_np,
            colmap_im_ids=colmap_im_ids_np,
            intrinsics=intrinsics_np,
            extrinsics=extrinsics_np,
            distortions=distortions_np,
            image_width=np.asarray(output_w, dtype=np.int32),
            image_height=np.asarray(output_h, dtype=np.int32),
            raw_json_width=np.asarray(-1 if raw_w is None else int(raw_w), dtype=np.int32),
            raw_json_height=np.asarray(-1 if raw_h is None else int(raw_h), dtype=np.int32),
            camera_model=np.asarray(str(raw_transforms.get("camera_model", "UNKNOWN"))),
        )

        split_files = build_scene_train_test_splits(
            scene_dir=scene_dir,
            num_frames=len(normalized_frames),
            scene_num_inputs=scene_num_inputs,
            log_path=log_path,
        )
        if not split_files:
            log_warning(
                log_path,
                f"{assignment.zip_name}: no feasible train_test_split_*.json files were created. Skipping scene.",
            )
            return None

        scene_meta = {
            "scene_id": assignment.scene_id,
            "split": assignment.split,
            "dataset": "dl3dv",
            "source_subset": "10K",
            "source_zip_name": assignment.zip_name,
            "source_zip_index": assignment.zip_index,
            "raw_inner_dir": raw_inner_dir,
            "raw_transforms_member": transforms_member,
            "num_frames": len(normalized_frames),
            "image_width": int(output_w),
            "image_height": int(output_h),
            "raw_json_width": None if raw_w is None else int(raw_w),
            "raw_json_height": None if raw_h is None else int(raw_h),
            "camera_model": raw_transforms.get("camera_model"),
            "camera_convention": "nerfstudio_opengl",
            "images_dir": "images",
            "transforms_file": "transforms.json",
            "cameras_file": "cameras.npz",
            "split_files": split_files,
        }
        write_json(scene_dir / "scene_meta.json", scene_meta)

        return {
            "scene_id": assignment.scene_id,
            "scene_dir": f"scenes/{assignment.scene_id}",
            "split": assignment.split,
            "num_frames": len(normalized_frames),
            "source_zip_name": assignment.zip_name,
            "source_zip_index": assignment.zip_index,
            "scene_split_files": [record["file"] for record in split_files],
            "available_num_inputs": [record["num_inputs"] for record in split_files],
        }


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    jpg_quality: int,
    scene_num_inputs: Iterable[int],
) -> None:
    zip_paths = sorted([p for p in input_dir.iterdir() if is_zip_file(p)])
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found in {input_dir}")

    ensure_dir(output_dir)
    ensure_dir(output_dir / "scenes")
    ensure_dir(output_dir / "splits")

    log_path = output_dir / "parser_warnings.log"
    clear_file(log_path)

    train_jsonl = output_dir / "splits" / "train.jsonl"
    val_jsonl = output_dir / "splits" / "val.jsonl"
    test_jsonl = output_dir / "splits" / "test.jsonl"
    clear_file(train_jsonl)
    clear_file(val_jsonl)
    clear_file(test_jsonl)

    assignments = assign_scene_splits(
        zip_paths=zip_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    zip_index_records: List[Dict[str, Any]] = []
    split_counts = {"train": 0, "val": 0, "test": 0}
    skipped_scenes = 0

    requested_scene_num_inputs = list(scene_num_inputs)

    for assignment in assignments:
        print(f"[{assignment.scene_id}] parsing {assignment.zip_name} -> {assignment.split}")
        scene_record = parse_single_scene(
            assignment=assignment,
            out_root=output_dir,
            jpg_quality=jpg_quality,
            scene_num_inputs=requested_scene_num_inputs,
            log_path=log_path,
        )

        if scene_record is None:
            skipped_scenes += 1
            continue

        split_counts[assignment.split] += 1

        zip_index_records.append(
            {
                "source_zip_index": assignment.zip_index,
                "source_zip_name": assignment.zip_name,
                "scene_id": assignment.scene_id,
                "split": assignment.split,
                "scene_dir": f"scenes/{assignment.scene_id}",
                "available_num_inputs": scene_record["available_num_inputs"],
            }
        )

        jsonl_record = {
            "scene_id": scene_record["scene_id"],
            "scene_dir": scene_record["scene_dir"],
            "num_frames": scene_record["num_frames"],
            "available_num_inputs": scene_record["available_num_inputs"],
            "scene_split_files": scene_record["scene_split_files"],
        }

        if assignment.split == "train":
            append_jsonl(train_jsonl, jsonl_record)
        elif assignment.split == "val":
            append_jsonl(val_jsonl, jsonl_record)
        else:
            append_jsonl(test_jsonl, jsonl_record)

    dataset_meta = {
        "dataset": "dl3dv",
        "source_subset": input_dir.name,
        "num_input_scenes": len(zip_paths),
        "num_parsed_scenes": sum(split_counts.values()),
        "num_skipped_scenes": skipped_scenes,
        "num_train_scenes": split_counts["train"],
        "num_val_scenes": split_counts["val"],
        "num_test_scenes": split_counts["test"],
        "split_scheme": "scene_level_with_per_scene_train_test_files",
        "scene_id_policy": "sorted_zip_order_zero_padded_6_digits",
        "image_format": "jpg",
        "camera_convention": "nerfstudio_opengl",
        "parser_version": "v4_seva_scene_splits",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scene_num_inputs_requested": requested_scene_num_inputs,
        "warning_log": "parser_warnings.log",
        "seva_scene_format": {
            "scene_root": "scenes/<scene_id>",
            "required": [
                "images/",
                "transforms.json",
                "train_test_split_*.json",
            ],
            "extra": [
                "cameras.npz",
                "scene_meta.json",
            ],
        },
    }

    write_json(output_dir / "dataset_meta.json", dataset_meta)
    write_json(output_dir / "zip_index.json", zip_index_records)

    print("\nDone.")
    print(f"Parsed scenes: {sum(split_counts.values())}")
    print(f"Skipped scenes: {skipped_scenes}")
    print(f"Warnings log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build normalized DL3DV parser-level dataset from hashed scene zips "
            "and create SEVA-compatible per-scene train/test split files."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing hashed zip files, e.g. data/10K",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root, e.g. dl3dv_parsed/10K",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scene-level split assignment",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95,
        help="JPEG quality when converting frames",
    )
    parser.add_argument(
        "--scene_num_inputs",
        type=parse_scene_num_inputs,
        default=DEFAULT_SCENE_NUM_INPUTS,
        help=(
            "Comma-separated list of SEVA per-scene input-view counts to create, "
            "for example: 1,6,32"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        jpg_quality=args.jpg_quality,
        scene_num_inputs=args.scene_num_inputs,
    )


if __name__ == "__main__":
    main()
