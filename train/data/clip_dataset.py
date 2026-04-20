# File: clip_dataset.py
# Description: Dataset wrapper for SEVA-style training clips built on top of
#              the official ReconfusionParser scene format.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from seva.data_io import ReconfusionParser
except ImportError as e:
    raise ImportError(
        "Could not import ReconfusionParser from seva.data_io. "
        "Make sure this file is placed inside the SEVA project or that the "
        "SEVA repo is on PYTHONPATH."
    ) from e


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _find_scene_dirs(dataset_root: Path, split: Optional[str]) -> List[Path]:
    """
    Find scene directories from either:
    1. dataset_root/splits/<split>.jsonl and scene_dir fields, or
    2. all directories containing transforms.json.

    Supports both:
      parsed_root/
        scenes/<scene_id>/...
        splits/train.jsonl

    and:
      scenes_root/
        <scene_id>/...
    """
    dataset_root = dataset_root.resolve()

    splits_dir = dataset_root / "splits"
    if split is not None and splits_dir.exists():
        jsonl_path = splits_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            records = _load_jsonl(jsonl_path)
            scene_dirs: List[Path] = []
            for rec in records:
                rel_scene_dir = rec.get("scene_dir")
                if rel_scene_dir is None:
                    continue
                scene_dir = (dataset_root / rel_scene_dir).resolve()
                if (scene_dir / "transforms.json").exists():
                    scene_dirs.append(scene_dir)
            if scene_dirs:
                return sorted(scene_dirs)

    scenes_dir = dataset_root / "scenes"
    search_root = scenes_dir if scenes_dir.exists() else dataset_root

    scene_dirs = [
        p for p in sorted(search_root.iterdir())
        if p.is_dir() and (p / "transforms.json").exists()
    ]
    return scene_dirs


def _filter_scene_dirs(
    scene_dirs: Sequence[Path],
    scene_names: Optional[Sequence[str]],
) -> List[Path]:
    if scene_names is None:
        return list(scene_dirs)

    requested = {str(name) for name in scene_names}
    filtered = [scene_dir for scene_dir in scene_dirs if scene_dir.name in requested]
    if not filtered:
        raise FileNotFoundError(
            f"None of the requested scenes {sorted(requested)} were found."
        )
    return filtered


def find_scene_dir(
    dataset_root: str | Path,
    scene_name: str,
    *,
    split: Optional[str] = None,
) -> Path:
    scene_dirs = _find_scene_dirs(Path(dataset_root), split=split)
    filtered = _filter_scene_dirs(scene_dirs, [scene_name])
    if len(filtered) != 1:
        raise FileNotFoundError(f"Could not uniquely resolve scene {scene_name!r}.")
    return filtered[0]


def choose_num_input_views_for_preset(
    available: Sequence[int],
    *,
    preset: Literal["quality", "proof"],
    requested: Optional[int] = None,
) -> int:
    values = sorted({int(x) for x in available})
    if not values:
        raise ValueError("Scene has no train_test_split_*.json files.")

    if requested is not None:
        requested = int(requested)
        if requested not in values:
            raise ValueError(
                f"Requested num_input_views={requested} is unavailable. "
                f"Available splits: {values}"
            )
        return requested

    if preset == "proof":
        if 1 in values:
            return 1
        return min(values)

    return max(values)


def inspect_scene_split(
    scene_dir: str | Path,
    *,
    num_input_views: Optional[int] = None,
    normalize_world: bool = False,
) -> Dict[str, Any]:
    parser = ReconfusionParser(str(Path(scene_dir)), normalize=normalize_world)
    available_num_input_views = sorted(
        int(key) for key in parser.splits_per_num_input_frames.keys()
    )
    train_ids: List[int] = []
    test_ids: List[int] = []
    if num_input_views is not None:
        split_dict = parser.splits_per_num_input_frames[int(num_input_views)]
        train_ids = [int(x) for x in split_dict["train_ids"]]
        test_ids = [int(x) for x in split_dict["test_ids"]]
    return {
        "scene_dir": str(Path(scene_dir)),
        "scene_name": Path(scene_dir).name,
        "available_num_input_views": available_num_input_views,
        "num_input_views": (
            None if num_input_views is None else int(num_input_views)
        ),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "num_total_frames": len(train_ids) + len(test_ids),
    }


def _get_wh_with_fixed_shortest_side(w: int, h: int, size: int) -> tuple[int, int]:
    if size <= 0:
        return w, h
    if w < h:
        return size, int(round(size * h / w))
    return int(round(size * w / h)), size


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    tensor = tensor / 127.5 - 1.0
    return tensor.contiguous()


def _resize_image_and_K(
    image_path: Path,
    K: np.ndarray,
    target_hw: Optional[Tuple[int, int]],
    target_short_side: Optional[int],
    normalize_intrinsics: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size

        K_out = np.array(K, dtype=np.float32, copy=True)

        if target_short_side is not None:
            target_w, target_h = _get_wh_with_fixed_shortest_side(
                orig_w,
                orig_h,
                int(target_short_side),
            )
            if (orig_h, orig_w) != (target_h, target_w):
                scale_x = float(target_w) / float(orig_w)
                scale_y = float(target_h) / float(orig_h)
                K_out[0, :] *= scale_x
                K_out[1, :] *= scale_y
                img = img.resize((target_w, target_h), resample=Image.BILINEAR) # type: ignore
        elif target_hw is not None:
            target_h, target_w = target_hw
            if (orig_h, orig_w) != (target_h, target_w):
                scale_x = float(target_w) / float(orig_w)
                scale_y = float(target_h) / float(orig_h)
                K_out[0, :] *= scale_x
                K_out[1, :] *= scale_y
                img = img.resize((target_w, target_h), resample=Image.BILINEAR) # type: ignore
            else:
                target_h, target_w = orig_h, orig_w
        else:
            target_h, target_w = orig_h, orig_w

        if normalize_intrinsics:
            K_out[0, :] /= float(target_w)
            K_out[1, :] /= float(target_h)

        img_tensor = _pil_to_tensor(img)
        K_tensor = torch.from_numpy(K_out).float()
        return img_tensor, K_tensor


class SevaClipDataset(Dataset):
    """
    Minimal SEVA-style clip dataset.

    Each item is one scene-conditioned training example containing:
      - imgs:        [T, 3, H, W]   in [-1, 1]
      - c2ws:        [T, 4, 4]
      - Ks:          [T, 3, 3]
      - input_mask:  [T] bool
      - frame_ids:   [T] frame indices inside the scene

    The dataset reuses the official ReconfusionParser. By default it samples
    paper-style random T-frame clips and chooses M input views uniformly from
    1..T-1. For render-matching or benchmark checks, set
    training_sample_mode="benchmark_split" to reuse train_test_split_*.json.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        split: Optional[str] = "train",
        num_input_views: int | Sequence[int] = (1, 6),
        total_frames: Optional[int] = 8,
        target_hw: Optional[Tuple[int, int]] = (576, 576),
        target_short_side: Optional[int] = None,
        normalize_world: bool = False,
        normalize_intrinsics: bool = True,
        shuffle_test_frames: bool = True,
        frame_selection_mode: Literal["clip", "img2trajvid"] = "clip",
        training_sample_mode: Literal["paper", "benchmark_split"] = "paper",
        small_stride_prob: float = 0.2,
        clips_per_scene_per_epoch: int = 16,
        scene_names: Optional[Sequence[str]] = None,
        parser_cache_size: int = 16,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.scene_dirs = _filter_scene_dirs(
            _find_scene_dirs(self.dataset_root, split=split),
            scene_names,
        )
        if not self.scene_dirs:
            raise FileNotFoundError(
                f"No SEVA scene directories found under {self.dataset_root}"
            )

        if isinstance(num_input_views, int):
            self.num_input_views_options = [num_input_views]
        else:
            self.num_input_views_options = sorted({int(x) for x in num_input_views})
        if not self.num_input_views_options:
            raise ValueError("num_input_views must contain at least one option.")

        self.total_frames = total_frames
        self.target_hw = target_hw
        self.target_short_side = target_short_side
        self.normalize_world = normalize_world
        self.normalize_intrinsics = normalize_intrinsics
        self.shuffle_test_frames = shuffle_test_frames
        self.frame_selection_mode = frame_selection_mode
        if training_sample_mode not in {"paper", "benchmark_split"}:
            raise ValueError(
                "training_sample_mode must be either 'paper' or 'benchmark_split', "
                f"got {training_sample_mode!r}."
            )
        if training_sample_mode == "paper" and self.total_frames is None:
            raise ValueError("training_sample_mode='paper' requires total_frames to be set.")
        self.training_sample_mode = training_sample_mode
        self.small_stride_prob = float(small_stride_prob)
        if not (0.0 <= self.small_stride_prob <= 1.0):
            raise ValueError(f"small_stride_prob must be in [0, 1], got {small_stride_prob}")
        self.clips_per_scene_per_epoch = max(int(clips_per_scene_per_epoch), 1)
        self.epoch = 0
        self.parser_cache_size = max(int(parser_cache_size), 0)
        self.seed = int(seed)
        self._parser_cache: Dict[str, ReconfusionParser] = {}
        self._parser_cache_order: List[str] = []

    def __len__(self) -> int:
        return len(self.scene_dirs) * self.clips_per_scene_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch seed so each scene yields fresh sampled clips."""
        self.epoch = int(epoch)

    def _get_parser(self, scene_dir: Path) -> ReconfusionParser:
        key = str(scene_dir.resolve())
        if key in self._parser_cache:
            return self._parser_cache[key]

        parser = ReconfusionParser(str(scene_dir), normalize=self.normalize_world)

        if self.parser_cache_size > 0:
            if len(self._parser_cache_order) >= self.parser_cache_size:
                oldest_key = self._parser_cache_order.pop(0)
                self._parser_cache.pop(oldest_key, None)
            self._parser_cache[key] = parser
            self._parser_cache_order.append(key)

        return parser

    def _make_rng(self, idx: int) -> random.Random:
        return random.Random(self.seed + self.epoch * 1_000_003 + idx)

    def _choose_num_inputs(
        self,
        parser: ReconfusionParser,
        rng: random.Random,
    ) -> int:
        available = []
        for k in parser.splits_per_num_input_frames.keys():
            try:
                k_int = int(k)
            except (TypeError, ValueError):
                continue
            available.append(k_int)
        available = sorted(set(available))
        if not available:
            raise ValueError(
                "Scene has no available train_test_split_*.json files."
            )

        feasible = []
        for k in available:
            if k not in self.num_input_views_options:
                continue
            if self.total_frames is not None and k >= self.total_frames:
                continue  # need at least one target frame for training
            feasible.append(k)

        if not feasible:
            raise ValueError(
                f"No feasible num_input_views for scene. Requested one of "
                f"{self.num_input_views_options}, available={available}, "
                f"total_frames={self.total_frames}."
            )

        return rng.choice(feasible)

    def _select_frame_ids(
        self,
        parser: ReconfusionParser,
        num_inputs: int,
        rng: random.Random,
    ) -> Tuple[List[int], List[int], List[int], List[bool]]:
        split_dict = parser.splits_per_num_input_frames[num_inputs]
        input_ids = [int(x) for x in split_dict["train_ids"]]
        test_ids = [int(x) for x in split_dict["test_ids"]]

        if len(test_ids) == 0:
            raise ValueError(
                f"Scene split train_test_split_{num_inputs}.json has no test_ids."
            )

        if self.total_frames is None:
            selected_test_ids = list(test_ids)
        else:
            num_targets = max(1, self.total_frames - len(input_ids))
            if len(test_ids) <= num_targets:
                selected_test_ids = list(test_ids)
            else:
                if self.frame_selection_mode == "img2trajvid":
                    selected_test_ids = list(test_ids[:num_targets])
                elif self.shuffle_test_frames:
                    selected_test_ids = rng.sample(test_ids, k=num_targets)
                else:
                    selected_test_ids = list(test_ids[:num_targets])

        if self.frame_selection_mode == "img2trajvid":
            all_ids = list(input_ids) + list(selected_test_ids)
            input_mask = [True] * len(input_ids) + [False] * len(selected_test_ids)
        else:
            all_ids = sorted(input_ids + selected_test_ids)
            input_id_set = set(input_ids)
            input_mask = [frame_id in input_id_set for frame_id in all_ids]

        return all_ids, input_ids, selected_test_ids, input_mask

    def _select_frame_ids_paper(
        self,
        parser: ReconfusionParser,
        rng: random.Random,
    ) -> Tuple[List[int], List[int], List[int], List[bool]]:
        """Sample a paper-style T-frame context and random input/target split.

        The SEVA paper samples a context window T=M+N, chooses M uniformly from
        1..T-1, and includes a small-stride branch with probability 0.2.
        """
        if self.total_frames is None:
            raise ValueError("paper sampling requires total_frames to be set.")

        T = int(self.total_frames)
        if T < 2:
            raise ValueError(f"paper sampling requires total_frames >= 2, got {T}.")
        all_ids = list(range(len(parser.image_paths)))
        if len(all_ids) < T:
            raise ValueError(f"Scene has only {len(all_ids)} frames, need T={T}.")

        if rng.random() < self.small_stride_prob:
            max_stride = max(1, (len(all_ids) - 1) // max(T - 1, 1))
            # Keep the stride small to bias this branch toward nearby views.
            stride = rng.randint(1, min(2, max_stride))
            max_start = len(all_ids) - 1 - stride * (T - 1)
            start = rng.randint(0, max_start)
            frame_ids = [all_ids[start + i * stride] for i in range(T)]
        else:
            frame_ids = sorted(rng.sample(all_ids, k=T))

        num_inputs = rng.randint(1, T - 1)
        input_positions = set(rng.sample(range(T), k=num_inputs))
        input_mask = [i in input_positions for i in range(T)]

        input_ids = [frame_id for i, frame_id in enumerate(frame_ids) if input_mask[i]]
        target_ids = [frame_id for i, frame_id in enumerate(frame_ids) if not input_mask[i]]
        return frame_ids, input_ids, target_ids, input_mask

    def _load_all_c2ws(self, parser: ReconfusionParser) -> torch.Tensor:
        all_ids = list(range(len(parser.image_paths)))
        return torch.stack(
            [
                torch.from_numpy(
                    np.asarray(parser.camtoworlds[frame_id], dtype=np.float32)
                )
                for frame_id in all_ids
            ],
            dim=0,
        )

    def _load_clip(
        self,
        parser: ReconfusionParser,
        scene_dir: Path,
        frame_ids: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs: List[torch.Tensor] = []
        Ks: List[torch.Tensor] = []
        c2ws: List[torch.Tensor] = []

        for frame_id in frame_ids:
            image_path = Path(parser.image_paths[frame_id])
            K = parser.Ks_dict[frame_id]
            c2w = parser.camtoworlds[frame_id]

            img_tensor, K_tensor = _resize_image_and_K(
                image_path=image_path,
                K=K,
                target_hw=self.target_hw,
                target_short_side=self.target_short_side,
                normalize_intrinsics=self.normalize_intrinsics,
            )
            imgs.append(img_tensor)
            Ks.append(K_tensor)
            c2ws.append(torch.from_numpy(np.asarray(c2w, dtype=np.float32)))

        return (
            torch.stack(imgs, dim=0),
            torch.stack(Ks, dim=0),
            torch.stack(c2ws, dim=0),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_dir = self.scene_dirs[idx % len(self.scene_dirs)]
        parser = self._get_parser(scene_dir)
        rng = self._make_rng(idx)

        if self.training_sample_mode == "paper":
            frame_ids, input_ids, target_ids, input_mask = self._select_frame_ids_paper(
                parser=parser,
                rng=rng,
            )
            num_inputs = len(input_ids)
        else:
            num_inputs = self._choose_num_inputs(parser, rng)
            frame_ids, input_ids, target_ids, input_mask = self._select_frame_ids(
                parser=parser,
                num_inputs=num_inputs,
                rng=rng,
            )

        imgs, Ks, c2ws = self._load_clip(
            parser=parser,
            scene_dir=scene_dir,
            frame_ids=frame_ids,
        )
        all_c2ws = self._load_all_c2ws(parser)

        input_mask_tensor = torch.tensor(input_mask, dtype=torch.bool)
        input_indices_tensor = torch.nonzero(input_mask_tensor, as_tuple=False).squeeze(-1)
        target_indices_tensor = torch.nonzero(~input_mask_tensor, as_tuple=False).squeeze(-1)

        sample: Dict[str, Any] = {
            "scene_name": scene_dir.name,
            "scene_dir": str(scene_dir),
            "imgs": imgs,                                  # [T, 3, H, W]
            "Ks": Ks,                                      # [T, 3, 3]
            "c2ws": c2ws,                                  # [T, 4, 4]
            "all_c2ws": all_c2ws,                          # [N_scene, 4, 4]
            "input_mask": input_mask_tensor,               # [T]
            "input_indices": input_indices_tensor,         # [M]
            "target_indices": target_indices_tensor,       # [N]
            "frame_ids": torch.tensor(frame_ids, dtype=torch.long),
            "source_input_frame_ids": torch.tensor(input_ids, dtype=torch.long),
            "source_target_frame_ids": torch.tensor(target_ids, dtype=torch.long),
            "num_input_views": int(num_inputs),
            "scene_scale": float(getattr(parser, "scene_scale", 1.0)),
        }
        return sample


def seva_clip_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple collate function.

    This assumes every sample in the batch has the same T, H, and W.
    That is usually true when:
      - total_frames is fixed, and
      - target_hw is fixed.
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    out: Dict[str, Any] = {
        "scene_name": [item["scene_name"] for item in batch],
        "scene_dir": [item["scene_dir"] for item in batch],
        "imgs": torch.stack([item["imgs"] for item in batch], dim=0),
        "Ks": torch.stack([item["Ks"] for item in batch], dim=0),
        "c2ws": torch.stack([item["c2ws"] for item in batch], dim=0),
        # Keep all scene cameras as a list because scenes can have different lengths.
        "all_c2ws": [item["all_c2ws"] for item in batch],
        "input_mask": torch.stack([item["input_mask"] for item in batch], dim=0),
        "frame_ids": torch.stack([item["frame_ids"] for item in batch], dim=0),
        "num_input_views": torch.tensor(
            [item["num_input_views"] for item in batch], dtype=torch.long
        ),
        "scene_scale": torch.tensor(
            [item["scene_scale"] for item in batch], dtype=torch.float32
        ),
        # Kept as lists because lengths can vary with different num_input_views.
        "input_indices": [item["input_indices"] for item in batch],
        "target_indices": [item["target_indices"] for item in batch],
        "source_input_frame_ids": [item["source_input_frame_ids"] for item in batch],
        "source_target_frame_ids": [item["source_target_frame_ids"] for item in batch],
    }
    return out


if __name__ == "__main__":
    # Tiny smoke-test example.
    # Update dataset_root to your parsed dataset path before running.
    dataset_root = Path("dl3dv_parsed/10K")
    if dataset_root.exists():
        ds = SevaClipDataset(
            dataset_root=dataset_root,
            split="train",
            num_input_views=(1, 6),
            total_frames=8,
            target_hw=(576, 576),
            normalize_world=False,
            normalize_intrinsics=True,
        )
        sample = ds[0]
        print("scene:", sample["scene_name"])
        print("imgs:", tuple(sample["imgs"].shape))
        print("Ks:", tuple(sample["Ks"].shape))
        print("c2ws:", tuple(sample["c2ws"].shape))
        print("input_mask:", tuple(sample["input_mask"].shape))
        print("frame_ids:", sample["frame_ids"].tolist())
        print("num_input_views:", sample["num_input_views"])
    else:
        print(
            "Set dataset_root in clip_dataset.py to your parsed scene directory "
            "for the smoke test."
        )
