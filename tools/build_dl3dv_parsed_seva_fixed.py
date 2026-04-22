#!/usr/bin/env python3
"""
build_dl3dv_parsed_seva_fixed.py

Parse DL3DV hashed scene zip files into SEVA/Reconfusion-style scene folders.

This version fixes the two issues that caused poor official-SEVA rendering on
your converted scenes:

1. Intrinsics are scaled to the ACTUAL extracted image size.
   Example: raw metadata 3840x2160, stored image 480x270:
       fl_x, cx are multiplied by 480 / 3840
       fl_y, cy are multiplied by 270 / 2160

2. Per-scene train_test_split_*.json files are author-style:
   - input views are selected by camera-space K-means
   - target views are strided, e.g. 0, 8, 16, ...
   - old behavior "evenly spaced inputs + every remaining target" is avoided

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
    │   ├── train_test_split_3.json
    │   ├── train_test_split_6.json
    │   ├── train_test_split_16.json
    │   ├── train_test_split_32.json
    │   └── images/
    │       ├── 000000.jpg
    │       ├── 000001.jpg
    │       └── ...

Requirements:
    pip install pillow numpy

Example:
    python tools/build_dl3dv_parsed_seva_fixed.py \
      --input_dir /path/to/raw/DL3DV/11K \
      --output_dir /path/to/dataset/dl3dv_parsed_fixed/11K \
      --scene_num_inputs 1 3 6 16 32 \
      --target_stride 8 \
      --image_format jpg \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
from PIL import Image


DEFAULT_SCENE_NUM_INPUTS = [1, 3, 6, 16, 32]

# These top-level fields are dangerous after image extraction/downsampling because
# SEVA's ReconfusionParser may prefer top-level camera metadata over frame-level
# metadata. We remove them and write correct per-frame values instead.
DROP_TOP_LEVEL_CAMERA_KEYS = {
    "w",
    "h",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "k1",
    "k2",
    "k3",
    "k4",
    "p1",
    "p2",
    "camera_model",
}

# Reconfusion/SEVA uses pinhole K in this path. Keep the RGBs as they are, but do
# not keep stale distortion parameters in the normalized transforms.
DROP_FRAME_DISTORTION_KEYS = {"k1", "k2", "k3", "k4", "p1", "p2"}


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


def natural_key(text: str) -> list[Any]:
    """Sort paths with numeric tokens in numeric order."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def find_single_transforms_path(zf: ZipFile) -> str:
    candidates = [
        name
        for name in zf.namelist()
        if name.endswith("transforms.json") and not name.endswith("/")
    ]
    if not candidates:
        raise FileNotFoundError("No transforms.json found inside zip.")
    if len(candidates) > 1:
        # Prefer the shallowest transforms.json. Stable for hashed DL3DV zips.
        candidates.sort(key=lambda s: (len(PurePosixPath(s).parts), s))
    return candidates[0]


def load_json_from_zip(zf: ZipFile, member: str) -> Dict[str, Any]:
    with zf.open(member) as f:
        return json.load(f)


def build_intrinsic_matrix(fl_x: float, fl_y: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def safe_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"Missing required camera field: {name}")
    return float(value)


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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

    n_train, n_val, _n_test = compute_split_counts(
        n=len(shuffled),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    assignments: List[SceneAssignment] = []
    for split_rank, (zip_index, zip_path) in enumerate(shuffled):
        if split_rank < n_train:
            split = "train"
        elif split_rank < n_train + n_val:
            split = "val"
        else:
            split = "test"

        # Scene IDs follow sorted zip order, not shuffled order, so they are stable.
        scene_id = f"{zip_index:06d}"
        assignments.append(
            SceneAssignment(
                scene_id=scene_id,
                zip_index=zip_index,
                zip_name=zip_path.name,
                zip_path=zip_path,
                split=split,
            )
        )

    assignments.sort(key=lambda a: a.zip_index)
    return assignments


def resolve_zip_member(zf: ZipFile, transforms_member: str, file_path: str) -> str:
    """Resolve a frame file_path inside a zip robustly."""
    names = zf.namelist()
    names_set = set(names)

    # Normalize path from transforms.
    clean = file_path.replace("\\", "/").lstrip("./")
    candidates = []

    # Direct path.
    candidates.append(clean)

    # Relative to transforms.json directory.
    transform_dir = PurePosixPath(transforms_member).parent
    if str(transform_dir) not in {"", "."}:
        candidates.append(str(transform_dir / clean))

    # Common case: file_path is relative to scene root and transforms.json is nested.
    clean_name = PurePosixPath(clean).name
    candidates.extend([name for name in names if PurePosixPath(name).name == clean_name])

    for cand in candidates:
        if cand in names_set and not cand.endswith("/"):
            return cand

    # Fallback: basename match, sorted for determinism.
    basename_matches = [
        name
        for name in names
        if PurePosixPath(name).name == clean_name and not name.endswith("/")
    ]
    if basename_matches:
        basename_matches.sort()
        return basename_matches[0]

    raise FileNotFoundError(
        f"Could not resolve image path {file_path!r} inside zip. "
        f"Tried examples: {candidates[:5]}"
    )


def extract_image(
    zf: ZipFile,
    member: str,
    out_path: Path,
    image_format: str,
    jpg_quality: int,
) -> Tuple[int, int]:
    """Extract image to jpg/png and return actual stored width/height."""
    with zf.open(member) as f:
        img = Image.open(f)
        img = img.convert("RGB")

        width, height = img.size
        ensure_dir(out_path.parent)

        if image_format == "jpg":
            img.save(out_path, format="JPEG", quality=jpg_quality, subsampling=0)
        elif image_format == "png":
            img.save(out_path, format="PNG")
        else:
            raise ValueError(f"Unsupported image_format: {image_format}")

    return int(width), int(height)


def camera_features_from_c2ws(
    c2ws: np.ndarray,
    *,
    position_weight: float = 1.0,
    direction_weight: float = 1.0,
) -> np.ndarray:
    """Build normalized camera-position + viewing-direction features."""
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2ws [N,4,4], got {c2ws.shape}")

    centers = c2ws[:, :3, 3].astype(np.float64)

    # OpenGL cameras look along local -Z. Negating all directions would preserve
    # pairwise distances if done consistently, but this is semantically clearer.
    dirs = -c2ws[:, :3, 2].astype(np.float64)
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)

    pos_mean = centers.mean(axis=0, keepdims=True)
    pos_std = centers.std(axis=0, keepdims=True) + 1e-8
    centers_norm = (centers - pos_mean) / pos_std

    return np.concatenate(
        [float(position_weight) * centers_norm, float(direction_weight) * dirs],
        axis=1,
    )


def farthest_point_kmeans_init(features: np.ndarray, k: int) -> np.ndarray:
    """Deterministic k-means initialization using representative/farthest points."""
    n = features.shape[0]
    if k > n:
        raise ValueError(f"k={k} cannot exceed number of points n={n}")

    # First center: point closest to global mean.
    mean = features.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(features - mean[None, :], axis=1)))
    selected = [first]

    while len(selected) < k:
        selected_feats = features[selected]
        dist = np.min(
            np.linalg.norm(features[:, None, :] - selected_feats[None, :, :], axis=2),
            axis=1,
        )
        dist[selected] = -1.0
        selected.append(int(np.argmax(dist)))

    return features[selected].copy()


def simple_kmeans_select_indices(
    features: np.ndarray,
    k: int,
    *,
    seed: int,
    num_iters: int = 50,
) -> List[int]:
    """
    Select k representative frame indices by small NumPy k-means.

    No scikit-learn dependency is required.
    """
    n = features.shape[0]
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)

    centers = farthest_point_kmeans_init(features, k)

    # Add tiny deterministic jitter so exact duplicates do not cause stable bad ties.
    jitter = rng.normal(scale=1e-8, size=features.shape)
    feats = features + jitter

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(num_iters):
        distances = np.linalg.norm(feats[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for ci in range(k):
            mask = new_labels == ci
            if np.any(mask):
                new_centers[ci] = feats[mask].mean(axis=0)
            else:
                # Empty cluster: re-seed with farthest point from existing centers.
                dist_to_centers = np.min(
                    np.linalg.norm(feats[:, None, :] - new_centers[None, :, :], axis=2),
                    axis=1,
                )
                new_centers[ci] = feats[int(np.argmax(dist_to_centers))]

        if np.array_equal(labels, new_labels):
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    selected: List[int] = []
    for ci in range(k):
        mask = np.where(labels == ci)[0]
        if mask.size == 0:
            continue
        distances = np.linalg.norm(feats[mask] - centers[ci][None, :], axis=1)
        selected.append(int(mask[int(np.argmin(distances))]))

    # Dedupe and fill if two cluster centers map to the same nearest frame.
    selected_set = set(selected)
    while len(selected_set) < k:
        selected_feats = feats[sorted(selected_set)] if selected_set else feats[:1]
        dist_to_selected = np.min(
            np.linalg.norm(feats[:, None, :] - selected_feats[None, :, :], axis=2),
            axis=1,
        )
        for idx in selected_set:
            dist_to_selected[idx] = -1.0
        selected_set.add(int(np.argmax(dist_to_selected)))

    return sorted(selected_set)


def build_scene_train_test_splits(
    scene_dir: Path,
    c2ws: np.ndarray,
    scene_num_inputs: Iterable[int],
    log_path: Path,
    *,
    target_stride: int,
    seed: int,
    position_weight: float,
    direction_weight: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    num_frames = int(c2ws.shape[0])

    if num_frames < 2:
        log_warning(
            log_path,
            f"{scene_dir.name}: only {num_frames} valid frame(s); cannot create SEVA train/test splits.",
        )
        return records

    if target_stride <= 0:
        raise ValueError("--target_stride must be positive")

    features = camera_features_from_c2ws(
        c2ws,
        position_weight=position_weight,
        direction_weight=direction_weight,
    )

    for requested_num_inputs in scene_num_inputs:
        requested_num_inputs = int(requested_num_inputs)
        if requested_num_inputs >= num_frames:
            log_warning(
                log_path,
                f"{scene_dir.name}: cannot create train_test_split_{requested_num_inputs}.json "
                f"because num_frames={num_frames}. Need at least num_inputs + 1 frames.",
            )
            continue

        train_ids = simple_kmeans_select_indices(
            features,
            requested_num_inputs,
            seed=seed + requested_num_inputs * 1009,
        )
        train_set = set(train_ids)

        # Author-style: strided target trajectory, not every remaining frame.
        test_ids = [idx for idx in range(0, num_frames, target_stride) if idx not in train_set]

        # Fallback for very short scenes or unlucky overlaps.
        if not test_ids:
            test_ids = [idx for idx in range(num_frames) if idx not in train_set]

        split_obj = {
            "train_ids": train_ids,
            "test_ids": test_ids,
            "num_frames": num_frames,
            "num_inputs": requested_num_inputs,
            "strategy": "kmeans_position_direction_inputs_strided_targets",
            "target_stride": int(target_stride),
            "kmeans_feature": "normalized_camera_center_plus_opengl_forward_direction",
            "seed": int(seed),
        }

        split_path = scene_dir / f"train_test_split_{requested_num_inputs}.json"
        write_json(split_path, split_obj)

        records.append(
            {
                "file": split_path.name,
                "num_inputs": requested_num_inputs,
                "num_targets": len(test_ids),
                "train_ids": train_ids,
                "test_ids_first": test_ids[:20],
                "strategy": split_obj["strategy"],
            }
        )

    return records


def frame_order_key(frame: Dict[str, Any], mode: str) -> Any:
    if mode == "source":
        return 0
    path = str(frame.get("file_path", ""))
    if mode == "sorted_path":
        return path
    if mode == "natural_path":
        return natural_key(path)
    raise ValueError(f"Unknown frame_order mode: {mode}")


def normalize_frame_camera(
    frame: Dict[str, Any],
    raw_transforms: Dict[str, Any],
    *,
    actual_w: int,
    actual_h: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Return normalized frame metadata, scaled intrinsic matrix, c2w, and raw distortion vector.
    """
    raw_w = first_not_none(frame.get("w"), raw_transforms.get("w"), actual_w)
    raw_h = first_not_none(frame.get("h"), raw_transforms.get("h"), actual_h)

    raw_fl_x = first_not_none(frame.get("fl_x"), raw_transforms.get("fl_x"))
    raw_fl_y = first_not_none(frame.get("fl_y"), raw_transforms.get("fl_y"))
    raw_cx = first_not_none(frame.get("cx"), raw_transforms.get("cx"))
    raw_cy = first_not_none(frame.get("cy"), raw_transforms.get("cy"))

    raw_w_f = safe_float(raw_w, "w")
    raw_h_f = safe_float(raw_h, "h")

    if raw_w_f <= 0 or raw_h_f <= 0:
        raise ValueError(f"Invalid raw image size: w={raw_w_f}, h={raw_h_f}")

    scale_x = float(actual_w) / raw_w_f
    scale_y = float(actual_h) / raw_h_f

    fl_x_scaled = safe_float(raw_fl_x, "fl_x") * scale_x
    fl_y_scaled = safe_float(raw_fl_y, "fl_y") * scale_y
    cx_scaled = safe_float(raw_cx, "cx") * scale_x
    cy_scaled = safe_float(raw_cy, "cy") * scale_y

    transform = frame.get("transform_matrix")
    if transform is None:
        raise ValueError("Missing required frame field: transform_matrix")
    c2w = np.asarray(transform, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got {c2w.shape}")

    intrinsic = build_intrinsic_matrix(fl_x_scaled, fl_y_scaled, cx_scaled, cy_scaled)

    distortion = np.array(
        [
            float(first_not_none(frame.get("k1"), raw_transforms.get("k1"), 0.0)),
            float(first_not_none(frame.get("k2"), raw_transforms.get("k2"), 0.0)),
            float(first_not_none(frame.get("p1"), raw_transforms.get("p1"), 0.0)),
            float(first_not_none(frame.get("p2"), raw_transforms.get("p2"), 0.0)),
        ],
        dtype=np.float32,
    )

    normalized = dict(frame)
    normalized["w"] = int(actual_w)
    normalized["h"] = int(actual_h)
    normalized["fl_x"] = float(fl_x_scaled)
    normalized["fl_y"] = float(fl_y_scaled)
    normalized["cx"] = float(cx_scaled)
    normalized["cy"] = float(cy_scaled)
    normalized["camera_model"] = "PINHOLE"

    for key in DROP_FRAME_DISTORTION_KEYS:
        normalized.pop(key, None)

    return normalized, intrinsic, c2w, distortion


def parse_single_scene(
    assignment: SceneAssignment,
    out_root: Path,
    jpg_quality: int,
    image_format: str,
    scene_num_inputs: Iterable[int],
    target_stride: int,
    seed: int,
    frame_order: str,
    applied_transform_policy: str,
    position_weight: float,
    direction_weight: float,
    log_path: Path,
) -> Optional[Dict[str, Any]]:
    scene_dir = out_root / "scenes" / assignment.scene_id
    images_dir = scene_dir / "images"
    ensure_dir(images_dir)

    with ZipFile(assignment.zip_path, "r") as zf:
        try:
            transforms_member = find_single_transforms_path(zf)
            raw_transforms = load_json_from_zip(zf, transforms_member)
        except Exception as e:
            log_warning(log_path, f"{assignment.zip_name}: failed to load transforms.json: {e}")
            return None

        raw_inner_dir = PurePosixPath(transforms_member).parent.as_posix()
        raw_frames = raw_transforms.get("frames", [])
        if not raw_frames:
            log_warning(log_path, f"{assignment.zip_name}: transforms.json has no frames. Skipping scene.")
            return None

        if frame_order != "source":
            raw_frames = sorted(raw_frames, key=lambda fr: frame_order_key(fr, frame_order))

        normalized_frames: List[Dict[str, Any]] = []
        frame_ids: List[int] = []
        intrinsics: List[np.ndarray] = []
        extrinsics: List[np.ndarray] = []
        distortions: List[np.ndarray] = []
        image_sizes: List[Tuple[int, int]] = []
        source_paths: List[str] = []

        output_w: Optional[int] = None
        output_h: Optional[int] = None
        raw_json_w = raw_transforms.get("w")
        raw_json_h = raw_transforms.get("h")

        for raw_idx, frame in enumerate(raw_frames):
            original_rel_path = frame.get("file_path")
            if not original_rel_path:
                log_warning(log_path, f"{assignment.zip_name}: frame missing file_path. Skipping frame.")
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

            out_ext = "jpg" if image_format == "jpg" else "png"
            out_name = f"{len(normalized_frames):06d}.{out_ext}"
            out_image_path = images_dir / out_name

            try:
                width, height = extract_image(
                    zf=zf,
                    member=zip_image_member,
                    out_path=out_image_path,
                    image_format=image_format,
                    jpg_quality=jpg_quality,
                )
            except Exception as e:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: failed to extract image {zip_image_member!r}: {e}. Skipping frame.",
                )
                continue

            if output_w is None or output_h is None:
                output_w, output_h = width, height
            elif width != output_w or height != output_h:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: inconsistent frame size for "
                    f"{original_rel_path!r}. Expected ({output_w}, {output_h}), got ({width}, {height}). "
                    "Skipping frame.",
                )
                try:
                    out_image_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            try:
                normalized_frame, intrinsic, c2w, distortion = normalize_frame_camera(
                    frame,
                    raw_transforms,
                    actual_w=width,
                    actual_h=height,
                )
            except Exception as e:
                log_warning(
                    log_path,
                    f"{assignment.zip_name}: invalid camera metadata for frame "
                    f"{original_rel_path!r}: {e}. Skipping frame.",
                )
                try:
                    out_image_path.unlink(missing_ok=True)
                except Exception:
                    pass
                continue

            normalized_frame["file_path"] = f"images/{out_name}"
            normalized_frame["original_file_path"] = str(original_rel_path)
            normalized_frame["resolved_zip_member"] = zip_image_member
            normalized_frame["source_frame_order_index"] = int(raw_idx)
            normalized_frame["normalized_frame_id"] = len(normalized_frames)

            normalized_frames.append(normalized_frame)
            frame_ids.append(len(normalized_frames) - 1)
            intrinsics.append(intrinsic)
            extrinsics.append(c2w)
            distortions.append(distortion)
            image_sizes.append((width, height))
            source_paths.append(str(original_rel_path))

    if len(normalized_frames) < 2:
        log_warning(
            log_path,
            f"{assignment.zip_name}: only {len(normalized_frames)} valid frames after parsing. Skipping scene.",
        )
        return None

    assert output_w is not None and output_h is not None

    # Build normalized transforms.json.
    normalized_transforms = {
        k: v
        for k, v in raw_transforms.items()
        if k not in DROP_TOP_LEVEL_CAMERA_KEYS and k != "frames"
    }

    if applied_transform_policy == "drop":
        normalized_transforms.pop("applied_transform", None)
    elif applied_transform_policy == "keep":
        pass
    else:
        raise ValueError(f"Unknown applied_transform_policy: {applied_transform_policy}")

    normalized_transforms["frames"] = normalized_frames
    write_json(scene_dir / "transforms.json", normalized_transforms)

    intrinsics_np = np.stack(intrinsics, axis=0).astype(np.float32)
    extrinsics_np = np.stack(extrinsics, axis=0).astype(np.float32)
    distortions_np = np.stack(distortions, axis=0).astype(np.float32)
    image_sizes_np = np.asarray(image_sizes, dtype=np.int32)

    np.savez_compressed(
        scene_dir / "cameras.npz",
        frame_ids=np.asarray(frame_ids, dtype=np.int32),
        intrinsics=intrinsics_np,
        extrinsics=extrinsics_np,
        distortions=distortions_np,
        image_sizes=image_sizes_np,
        image_width=np.asarray([int(output_w)], dtype=np.int32),
        image_height=np.asarray([int(output_h)], dtype=np.int32),
    )

    split_files = build_scene_train_test_splits(
        scene_dir=scene_dir,
        c2ws=extrinsics_np,
        scene_num_inputs=scene_num_inputs,
        log_path=log_path,
        target_stride=target_stride,
        seed=seed + assignment.zip_index * 7919,
        position_weight=position_weight,
        direction_weight=direction_weight,
    )
    if not split_files:
        log_warning(
            log_path,
            f"{assignment.zip_name}: no feasible train_test_split_*.json files were created. Skipping scene.",
        )
        return None

    first = normalized_frames[0]
    scene_meta = {
        "scene_id": assignment.scene_id,
        "split": assignment.split,
        "dataset": "dl3dv",
        "source_subset": assignment.zip_path.parent.name,
        "source_zip_name": assignment.zip_name,
        "source_zip_index": assignment.zip_index,
        "raw_inner_dir": raw_inner_dir,
        "raw_transforms_member": transforms_member,
        "num_frames": len(normalized_frames),
        "image_width": int(output_w),
        "image_height": int(output_h),
        "raw_json_width": None if raw_json_w is None else int(raw_json_w),
        "raw_json_height": None if raw_json_h is None else int(raw_json_h),
        "camera_model": "PINHOLE",
        "source_camera_model": raw_transforms.get("camera_model"),
        "camera_convention": "nerfstudio_opengl",
        "images_dir": "images",
        "transforms_file": "transforms.json",
        "cameras_file": "cameras.npz",
        "split_files": split_files,
        "parser_fixes": {
            "scaled_intrinsics_to_extracted_image_size": True,
            "removed_top_level_camera_metadata": True,
            "frame_order": frame_order,
            "split_strategy": "kmeans_position_direction_inputs_strided_targets",
            "target_stride": target_stride,
            "applied_transform_policy": applied_transform_policy,
        },
        "first_frame_intrinsics": {
            "w": first["w"],
            "h": first["h"],
            "fl_x": first["fl_x"],
            "fl_y": first["fl_y"],
            "cx": first["cx"],
            "cy": first["cy"],
            "cx_over_w": float(first["cx"]) / float(first["w"]),
            "cy_over_h": float(first["cy"]) / float(first["h"]),
        },
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
        "image_width": int(output_w),
        "image_height": int(output_h),
    }


def discover_zip_paths(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = sorted(p for p in input_dir.rglob("*.zip") if is_zip_file(p))
    else:
        paths = sorted(p for p in input_dir.iterdir() if is_zip_file(p))
    return paths


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    jpg_quality: int,
    image_format: str,
    scene_num_inputs: Iterable[int],
    target_stride: int,
    frame_order: str,
    applied_transform_policy: str,
    position_weight: float,
    direction_weight: float,
    recursive: bool,
    max_scenes: Optional[int],
    overwrite: bool,
) -> None:
    zip_paths = discover_zip_paths(input_dir, recursive=recursive)
    if not zip_paths:
        raise FileNotFoundError(f"No .zip files found in {input_dir}")

    if max_scenes is not None:
        zip_paths = zip_paths[: int(max_scenes)]

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite or choose a new --output_dir."
            )

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

    requested_scene_num_inputs = [int(v) for v in scene_num_inputs]

    for assignment in assignments:
        print(f"[{assignment.scene_id}] parsing {assignment.zip_name} -> {assignment.split}")
        scene_record = parse_single_scene(
            assignment=assignment,
            out_root=output_dir,
            jpg_quality=jpg_quality,
            image_format=image_format,
            scene_num_inputs=requested_scene_num_inputs,
            target_stride=target_stride,
            seed=seed,
            frame_order=frame_order,
            applied_transform_policy=applied_transform_policy,
            position_weight=position_weight,
            direction_weight=direction_weight,
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
                "image_width": scene_record["image_width"],
                "image_height": scene_record["image_height"],
            }
        )

        jsonl_record = {
            "scene_id": scene_record["scene_id"],
            "scene_dir": scene_record["scene_dir"],
            "num_frames": scene_record["num_frames"],
            "available_num_inputs": scene_record["available_num_inputs"],
            "scene_split_files": scene_record["scene_split_files"],
            "image_width": scene_record["image_width"],
            "image_height": scene_record["image_height"],
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
        "image_format": image_format,
        "camera_convention": "nerfstudio_opengl",
        "parser_version": "v5_scaled_intrinsics_kmeans_strided_targets",
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "scene_num_inputs_requested": requested_scene_num_inputs,
        "target_stride": target_stride,
        "frame_order": frame_order,
        "applied_transform_policy": applied_transform_policy,
        "warning_log": "parser_warnings.log",
        "critical_fixes": [
            "scale intrinsics to actual extracted image size",
            "remove top-level camera metadata from normalized transforms.json",
            "write per-frame w/h/fl_x/fl_y/cx/cy",
            "K-means camera-aware train_ids",
            "strided test_ids",
        ],
        "seva_scene_format": {
            "scene_root": "scenes/<scene_id>",
            "required": ["images/", "transforms.json", "train_test_split_*.json"],
            "extra": ["cameras.npz", "scene_meta.json"],
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
            "with SEVA-compatible transforms and author-style train/test split files."
        )
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing hashed zip files.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root, e.g. dl3dv_parsed_fixed/11K.")

    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scene_num_inputs",
        type=int,
        nargs="+",
        default=DEFAULT_SCENE_NUM_INPUTS,
        help="Per-scene train_test_split_N.json files to create.",
    )
    parser.add_argument(
        "--target_stride",
        type=int,
        default=8,
        help="Stride for author-style target IDs, e.g. 0,8,16,...",
    )

    parser.add_argument("--image_format", type=str, default="jpg", choices=["jpg", "png"])
    parser.add_argument("--jpg_quality", type=int, default=95)

    parser.add_argument(
        "--frame_order",
        type=str,
        default="sorted_path",
        choices=["source", "sorted_path", "natural_path"],
        help=(
            "How to order frames before assigning new 000000 image IDs. "
            "sorted_path is closest to the SEVA benchmark expectation when filenames are zero-padded."
        ),
    )
    parser.add_argument(
        "--applied_transform_policy",
        type=str,
        default="keep",
        choices=["keep", "drop"],
        help=(
            "Whether to keep top-level applied_transform in normalized transforms.json. "
            "If official SEVA still looks wrong after fixing intrinsics, A/B test --applied_transform_policy drop."
        ),
    )
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--direction_weight", type=float, default=1.0)

    parser.add_argument("--recursive", action="store_true", help="Search for zip files recursively under input_dir.")
    parser.add_argument("--max_scenes", type=int, default=None, help="Parse only the first N sorted zip files.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output_dir before writing.")

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
        image_format=args.image_format,
        scene_num_inputs=args.scene_num_inputs,
        target_stride=args.target_stride,
        frame_order=args.frame_order,
        applied_transform_policy=args.applied_transform_policy,
        position_weight=args.position_weight,
        direction_weight=args.direction_weight,
        recursive=args.recursive,
        max_scenes=args.max_scenes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
