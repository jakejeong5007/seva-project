from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seva.modules.autoencoder import AutoEncoder as SevaAutoEncoder
from train.data.clip_dataset import (
    SevaClipDataset,
    choose_num_input_views_for_preset,
    find_scene_dir,
    inspect_scene_split,
    seva_clip_collate,
)
from train.training.diffusion_loss import compute_seva_diffusion_loss
try:
    from train.training.epipolar_loss import (
        EpipolarLossConfig,
        compute_visibility_gated_epipolar_loss,
        should_apply_epipolar_loss,
    )
except Exception:  # pragma: no cover - optional research loss.
    EpipolarLossConfig = None  # type: ignore[assignment]
    compute_visibility_gated_epipolar_loss = None  # type: ignore[assignment]
    should_apply_epipolar_loss = None  # type: ignore[assignment]

from train.training.model_factory import (
    SevaBundle,
    build_seva_bundle,
    infer_batch_shape,
    summarize_bundle,
)
from train.training.validate import ValidationConfig, evaluate
from train.utils.checkpointing import (
    ResumeState,
    initialize_backbone_weights,
    load_checkpoint,
    save_checkpoint,
)
from train.utils.logging_utils import append_jsonl, ensure_dir, save_json, to_jsonable

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
except Exception:  # pragma: no cover - FSDP is optional for CPU-only syntax checks.
    FullyShardedDataParallel = None
    MixedPrecision = None
    ShardingStrategy = None
    size_based_auto_wrap_policy = None


DEFAULT_SCRATCH_LR = 1e-4
DEFAULT_FINETUNE_LR = 2e-5


class DummyAE(nn.Module):
    """Random latent encoder used only for smoke tests."""

    def __init__(self, latent_channels: int = 4, downsample_factor: int = 8) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.downsample_factor = int(downsample_factor)

    def encode(self, x: torch.Tensor, encoding_t: int = 1) -> torch.Tensor:
        del encoding_t
        if x.dim() != 4:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(x.shape)}")
        n, _, h, w = x.shape
        if h % self.downsample_factor != 0 or w % self.downsample_factor != 0:
            raise ValueError(
                f"Input spatial size {(h, w)} must be divisible by downsample factor "
                f"{self.downsample_factor}."
            )
        return torch.randn(
            n,
            self.latent_channels,
            h // self.downsample_factor,
            w // self.downsample_factor,
            device=x.device,
            dtype=x.dtype,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical SEVA training entrypoint for the local training stack."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("runs"))
    parser.add_argument("--run_name", type=str, default="seva_stage1")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--total_frames", type=int, default=8)
    parser.add_argument("--num_input_views", type=str, default="1,6")
    parser.add_argument(
        "--training_sample_mode",
        type=str,
        default="paper",
        choices=["paper", "benchmark_split"],
        help=(
            "paper samples random T-frame contexts and random M input views. "
            "benchmark_split reuses train_test_split_*.json for render matching."
        ),
    )
    parser.add_argument("--small_stride_prob", type=float, default=0.2)
    parser.add_argument("--clips_per_scene_per_epoch", type=int, default=16)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument(
        "--l_short",
        type=int,
        default=None,
        help=(
            "Resize the shortest image side to this value while preserving aspect ratio. "
            "When set, overrides the fixed --height/--width resize used by training."
        ),
    )
    parser.add_argument(
        "--normalize_intrinsics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize K by image width/height inside the dataset.",
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--distributed_strategy",
        type=str,
        default="auto",
        choices=["auto", "none", "ddp", "fsdp"],
        help=(
            "Multi-GPU strategy. Launch ddp/fsdp with torchrun. "
            "auto selects fsdp when WORLD_SIZE>1, otherwise none."
        ),
    )
    parser.add_argument(
        "--dist_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="torch.distributed backend used for multi-GPU launch.",
    )
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Passed to DistributedDataParallel when --distributed_strategy ddp.",
    )
    parser.add_argument(
        "--fsdp_min_num_params",
        type=int,
        default=20_000_000,
        help=(
            "FSDP size-based auto-wrap threshold. Lower values shard more "
            "submodules and may reduce peak memory, at the cost of more communication."
        ),
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
        help="FSDP sharding strategy.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--autocast_dtype",
        type=str,
        default="bfloat16",
        choices=["none", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--sdpa_backend",
        type=str,
        default="math",
        choices=["auto", "math"],
        help="Force a PyTorch SDPA backend during training and validation.",
    )

    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--latent_downsample_factor", type=int, default=8)
    parser.add_argument("--latent_scaling_factor", type=float, default=1.0)
    parser.add_argument("--encoding_t", type=int, default=1)
    parser.add_argument("--sample_posterior", action="store_true", default=False)

    parser.add_argument(
        "--schedule",
        type=str,
        default="seva_ddpm",
        choices=["seva_ddpm", "cosine_vp", "rf_linear"],
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="epsilon",
        choices=["epsilon", "x0", "v", "velocity"],
    )
    parser.add_argument("--camera_scale", type=float, default=2.0)
    parser.add_argument("--target_frame_weight", type=float, default=1.0)
    parser.add_argument("--input_frame_weight", type=float, default=0.0)
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.0)

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Defaults to 1e-4 for scratch and 2e-5 for pretrained initialization.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--betas", type=str, default="0.9,0.999")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=4)
    parser.add_argument(
        "--overfit_one_batch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache the first training batch and reuse it every step for debugging.",
    )
    parser.add_argument(
        "--match_render_scene",
        type=str,
        default=None,
        help=(
            "Overfit a single scene using the same scene split/order as "
            "scripts/render_compare.py."
        ),
    )
    parser.add_argument(
        "--match_render_task",
        type=str,
        default="img2vid",
        choices=["img2vid", "img2trajvid"],
        help=(
            "Which inference ordering to mirror in --match_render_scene mode. "
            "Use img2vid for benchmark-style trajectory evaluation and img2trajvid "
            "only when you explicitly want input-views-first ordering."
        ),
    )
    parser.add_argument(
        "--match_render_preset",
        type=str,
        default="quality",
        choices=["quality", "proof"],
        help="Split-selection preset used together with --match_render_scene.",
    )
    parser.add_argument(
        "--match_render_num_inputs",
        type=int,
        default=None,
        help="Override the split selected by --match_render_preset.",
    )
    parser.add_argument(
        "--match_render_train_frames",
        type=int,
        default=None,
        help=(
            "Maximum number of frames to use for training when --match_render_scene "
            "is enabled. Defaults to --total_frames instead of the full render trajectory."
        ),
    )

    parser.add_argument(
        "--train_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train_conditioner",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--train_ae",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--use_dummy_ae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a random latent encoder for pipeline smoke tests only.",
    )
    parser.add_argument(
        "--ae_factory",
        type=str,
        default=None,
        help="Import path 'package.module:function_name' used to build a custom AE.",
    )
    parser.add_argument(
        "--ae_kwargs_json",
        type=str,
        default="{}",
        help="JSON kwargs passed to either the custom AE factory or SEVA AutoEncoder.",
    )

    parser.add_argument(
        "--init_backbone_mode",
        type=str,
        default="official",
        choices=["scratch", "resume", "official", "local_pretrained"],
        help="How to initialize the SEVA backbone before optimizer creation.",
    )
    parser.add_argument(
        "--official_model_version",
        type=float,
        default=1.1,
        help="Official SEVA checkpoint version used when --init_backbone_mode official.",
    )
    parser.add_argument(
        "--official_pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-virtual-camera",
        help="Hugging Face repo id or local directory passed to seva.utils.load_model.",
    )
    parser.add_argument(
        "--official_weight_name",
        type=str,
        default="model.safetensors",
        help="Weight filename passed to seva.utils.load_model.",
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=Path,
        default=None,
        help="Checkpoint or state_dict used when --init_backbone_mode local_pretrained.",
    )
    parser.add_argument(
        "--pretrained_strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use strict=True when loading pretrained backbone weights.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Training checkpoint path to resume from.",
    )
    parser.add_argument(
        "--debug_fixed_noise_seed",
        type=int,
        default=None,
        help=(
            "Deprecated no-op in the current checked-in diffusion_loss.py. "
            "Keep only for backwards CLI compatibility unless diffusion_loss.py "
            "is extended to accept deterministic-noise debug inputs."
        ),
    )
    parser.add_argument(
        "--debug_fixed_noise_idx",
        type=int,
        default=None,
        help=(
            "Deprecated no-op in the current checked-in diffusion_loss.py. "
            "Keep only for backwards CLI compatibility unless diffusion_loss.py "
            "is extended to accept deterministic-noise debug inputs."
        ),
    )
    parser.add_argument(
        "--activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use torch.utils.checkpoint around the SEVA backbone forward. "
            "Reduces activation memory at the cost of extra compute."
        ),
    )

    parser.add_argument(
        "--offload_frozen_encoders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Move the frozen CLIP conditioner and VAE to CPU after conditioning "
            "is built, before the trainable SEVA backbone forward."
        ),
    )

    # ------------------------------------------------------------------
    # SEVA/DDPM timestep sampling
    # ------------------------------------------------------------------
    parser.add_argument(
        "--seva_noise_idx_min",
        type=int,
        default=0,
        help="Inclusive minimum SEVA/DDPM noise index. Low index = low noise.",
    )
    parser.add_argument(
        "--seva_noise_idx_max",
        type=int,
        default=999,
        help=(
            "Inclusive maximum SEVA/DDPM noise index. Use e.g. 100 or 200 "
            "for low-noise fine-tuning / epipolar-loss ablations."
        ),
    )
    parser.add_argument(
        "--seva_timestep_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "low_noise_beta", "fixed"],
        help=(
            "How to sample SEVA/DDPM noise indices. 'uniform' preserves the "
            "default behavior; 'low_noise_beta' biases toward lower-noise "
            "indices inside the selected range; 'fixed' is deterministic."
        ),
    )
    parser.add_argument(
        "--seva_timestep_beta_alpha",
        type=float,
        default=0.7,
        help="Beta alpha for --seva_timestep_sampling low_noise_beta.",
    )
    parser.add_argument(
        "--seva_timestep_beta_beta",
        type=float,
        default=3.0,
        help="Beta beta for --seva_timestep_sampling low_noise_beta.",
    )
    parser.add_argument(
        "--seva_fixed_noise_idx",
        type=int,
        default=None,
        help=(
            "Fixed SEVA/DDPM noise index for deterministic debugging. "
            "Only used when --seva_timestep_sampling fixed."
        ),
    )

    # ------------------------------------------------------------------
    # Research: visibility-gated epipolar loss
    # ------------------------------------------------------------------
    parser.add_argument("--epi_loss_weight", type=float, default=0.0)
    parser.add_argument("--epi_start_step", type=int, default=1000)
    parser.add_argument("--epi_warmup_steps", type=int, default=2000)
    parser.add_argument("--epi_every", type=int, default=2)
    parser.add_argument("--epi_prediction_type", type=str, default="epsilon", choices=["epsilon", "x0"])
    parser.add_argument("--epi_target_frames_per_clip", type=int, default=1)
    parser.add_argument("--epi_sources_per_target", type=int, default=1)
    parser.add_argument("--epi_res", type=int, default=128)
    parser.add_argument("--epi_pixels", type=int, default=512)
    parser.add_argument("--epi_line_samples", type=int, default=32)
    parser.add_argument("--epi_textured_fraction", type=float, default=0.8)
    parser.add_argument("--epi_tau", type=float, default=0.07)
    parser.add_argument("--epi_conf_min", type=float, default=0.10)
    parser.add_argument("--epi_match_logit_center", type=float, default=0.30)
    parser.add_argument("--epi_match_logit_scale", type=float, default=0.10)
    parser.add_argument("--epi_max_sigma", type=float, default=1.0)
    parser.add_argument("--epi_sigma_softness", type=float, default=0.25)
    parser.add_argument("--epi_min_epipolar_baseline", type=float, default=1e-4)
    parser.add_argument("--epi_use_rotation_h_fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epi_min_rotation_for_h_deg", type=float, default=2.0)
    parser.add_argument("--epi_min_valid_ratio", type=float, default=0.15)
    parser.add_argument("--epi_feature_mode", type=str, default="rgb_sobel", choices=["rgb_sobel"])
    parser.add_argument("--epi_ae_decode_chunk_size", type=int, default=1)
    parser.add_argument("--epi_auto_move_ae_to_device", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def parse_num_input_views(text: str) -> tuple[int, ...]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--num_input_views must contain at least one integer.")
    return tuple(values)


def resolve_dtype(name: str) -> Optional[torch.dtype]:
    mapping: dict[str, Optional[torch.dtype]] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "none": None,
    }
    return mapping[name]


def parse_betas(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("--betas must be of the form '0.9,0.999'")
    return float(parts[0]), float(parts[1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_sdpa_backends_for_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    try:
        torch.backends.cuda.enable_cudnn_sdp(True)
    except Exception:
        pass


def make_sdpa_context(mode: str):
    if mode == "auto" or sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    if mode == "math":
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def import_factory(factory_path: str) -> Any:
    if ":" not in factory_path:
        raise ValueError(
            f"Factory path must look like 'package.module:function_name', got {factory_path!r}"
        )
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def build_ae_from_args(args: argparse.Namespace) -> nn.Module:
    if args.use_dummy_ae:
        return DummyAE(
            latent_channels=args.latent_channels,
            downsample_factor=args.latent_downsample_factor,
        )

    ae_kwargs = json_loads_dict(args.ae_kwargs_json)
    if args.ae_factory is not None:
        factory = import_factory(args.ae_factory)
        ae = factory(**ae_kwargs)
        if not isinstance(ae, nn.Module):
            raise TypeError(f"AE factory must return an nn.Module, got {type(ae)}")
        return ae

    ae = SevaAutoEncoder(**ae_kwargs)
    if not isinstance(ae, nn.Module):
        raise TypeError(f"SEVA AutoEncoder must be an nn.Module, got {type(ae)}")
    return ae


def json_loads_dict(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object, got {type(payload)}")
    return payload


def build_dataloader(
    *,
    dataset_root: Path,
    split: Optional[str],
    num_input_views: tuple[int, ...],
    total_frames: int,
    height: int,
    width: int,
    l_short: Optional[int],
    normalize_intrinsics: bool,
    shuffle: bool,
    seed: int,
    batch_size: int,
    num_workers: int,
    frame_selection_mode: str = "clip",
    training_sample_mode: str = "paper",
    small_stride_prob: float = 0.2,
    clips_per_scene_per_epoch: int = 16,
    scene_names: Optional[tuple[str, ...]] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    dataset = SevaClipDataset(
        dataset_root=dataset_root,
        split=split,
        num_input_views=num_input_views,
        total_frames=total_frames,
        target_hw=None if l_short is not None else (height, width),
        target_short_side=l_short,
        normalize_world=False,
        normalize_intrinsics=normalize_intrinsics,
        shuffle_test_frames=True,
        frame_selection_mode=frame_selection_mode,  # type: ignore[arg-type]
        training_sample_mode=training_sample_mode,  # type: ignore[arg-type]
        small_stride_prob=small_stride_prob,
        clips_per_scene_per_epoch=clips_per_scene_per_epoch,
        scene_names=scene_names,
        seed=seed,
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=seva_clip_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def resolve_render_match_overrides(
    *,
    dataset_root: Path,
    split: Optional[str],
    scene_name: str,
    preset: str,
    requested_num_inputs: Optional[int],
    task: str = "img2vid",
) -> dict[str, Any]:
    scene_dir = find_scene_dir(dataset_root, scene_name, split=split)
    metadata_default = inspect_scene_split(scene_dir)
    available = metadata_default["available_num_input_views"]
    selected_num_inputs = choose_num_input_views_for_preset(
        available,
        preset=preset,  # type: ignore[arg-type]
        requested=requested_num_inputs,
    )
    metadata = inspect_scene_split(scene_dir, num_input_views=selected_num_inputs)
    return {
        "scene_names": (scene_name,),
        # img2vid keeps trajectory order; img2trajvid reorders inputs-first.
        "frame_selection_mode": "img2trajvid" if task == "img2trajvid" else "clip",
        "num_input_views": (selected_num_inputs,),
        "total_frames": metadata["num_total_frames"],
        "resolved_scene_dir": scene_dir,
        "resolved_scene_name": scene_name,
        "resolved_num_input_views": selected_num_inputs,
        "resolved_available_num_input_views": available,
        "resolved_task": task,
    }


def resolve_effective_init_mode(args: argparse.Namespace) -> str:
    if args.resume is not None:
        if args.init_backbone_mode != "resume":
            print(
                f"Resume checkpoint provided; overriding init_backbone_mode="
                f"{args.init_backbone_mode!r} with 'resume'."
            )
        return "resume"
    if args.init_backbone_mode == "resume":
        raise ValueError("--resume is required when --init_backbone_mode resume")
    return args.init_backbone_mode


def resolve_learning_rate(user_lr: Optional[float], init_mode: str) -> float:
    if user_lr is not None:
        return float(user_lr)
    if init_mode in {"official", "local_pretrained"}:
        return DEFAULT_FINETUNE_LR
    return DEFAULT_SCRATCH_LR


def build_epipolar_config_from_args(args: argparse.Namespace) -> Optional[Any]:
    if float(args.epi_loss_weight) <= 0.0:
        return None
    if EpipolarLossConfig is None:
        raise RuntimeError(
            "--epi_loss_weight was set, but train.training.epipolar_loss could not be imported."
        )
    return EpipolarLossConfig(
        loss_weight=float(args.epi_loss_weight),
        start_step=int(args.epi_start_step),
        warmup_steps=int(args.epi_warmup_steps),
        every=int(args.epi_every),
        prediction_type=str(args.epi_prediction_type),
        target_frames_per_clip=int(args.epi_target_frames_per_clip),
        sources_per_target=int(args.epi_sources_per_target),
        feature_res=int(args.epi_res),
        pixels_per_pair=int(args.epi_pixels),
        line_samples=int(args.epi_line_samples),
        textured_fraction=float(args.epi_textured_fraction),
        tau=float(args.epi_tau),
        confidence_min=float(args.epi_conf_min),
        match_logit_center=float(args.epi_match_logit_center),
        match_logit_scale=float(args.epi_match_logit_scale),
        max_sigma=float(args.epi_max_sigma),
        sigma_softness=float(args.epi_sigma_softness),
        min_epipolar_baseline=float(args.epi_min_epipolar_baseline),
        use_rotation_h_fallback=bool(args.epi_use_rotation_h_fallback),
        min_rotation_for_h_deg=float(args.epi_min_rotation_for_h_deg),
        min_valid_ratio=float(args.epi_min_valid_ratio),
        feature_mode=str(args.epi_feature_mode),
        ae_decode_chunk_size=(
            None if args.epi_ae_decode_chunk_size is None or int(args.epi_ae_decode_chunk_size) <= 0
            else int(args.epi_ae_decode_chunk_size)
        ),
        auto_move_ae_to_device=bool(args.epi_auto_move_ae_to_device),
    )


def get_optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> list[float]:
    learning_rates: list[float] = []
    for param_group in optimizer.param_groups:
        lr = float(param_group["lr"])
        if lr not in learning_rates:
            learning_rates.append(lr)
    return learning_rates


def format_learning_rates(learning_rates: list[float]) -> str:
    return ", ".join(f"{lr:.2e}" for lr in learning_rates)


def set_module_modes(bundle: SevaBundle) -> None:
    if bundle.train_backbone:
        bundle.wrapper.train()
    else:
        bundle.wrapper.eval()

    if bundle.conditioner is not None:
        if bundle.train_conditioner:
            bundle.conditioner.train()
        else:
            bundle.conditioner.eval()

    if bundle.ae is not None:
        if bundle.train_ae:
            bundle.ae.train()
        else:
            bundle.ae.eval()


def count_finite_grads(module: nn.Module) -> tuple[int, int]:
    total = 0
    finite = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += 1
        if torch.isfinite(parameter.grad).all():
            finite += 1
    return finite, total


@dataclass(frozen=True)
class DistributedContext:
    strategy: str
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if distributed_is_initialized():
        return int(dist.get_rank())
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def main_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)


def resolve_distributed_context(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    requested = str(args.distributed_strategy)
    if requested == "auto":
        strategy = "fsdp" if world_size > 1 else "none"
    else:
        strategy = requested

    distributed = world_size > 1 and strategy != "none"
    if strategy in {"ddp", "fsdp"} and world_size <= 1:
        raise ValueError(
            f"--distributed_strategy {strategy!r} requires torchrun with WORLD_SIZE>1. "
            "Use: torchrun --standalone --nproc_per_node=4 -m train.train ..."
        )

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SEVA training currently expects CUDA GPUs.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://")
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank)))

    return DistributedContext(
        strategy=strategy,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=(rank == 0),
    )


def resolve_device_for_process(args: argparse.Namespace, dist_ctx: DistributedContext) -> torch.device:
    if dist_ctx.distributed:
        return torch.device("cuda", dist_ctx.local_rank)
    if args.device is not None:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_ddp_module(module: nn.Module) -> bool:
    return isinstance(module, DistributedDataParallel)


def is_fsdp_module(module: nn.Module) -> bool:
    return FullyShardedDataParallel is not None and isinstance(
        module, FullyShardedDataParallel
    )


def is_parallel_wrapper(module: nn.Module) -> bool:
    return is_ddp_module(module) or is_fsdp_module(module)


def trainable_backbone_container(bundle: SevaBundle) -> nn.Module:
    if is_parallel_wrapper(bundle.wrapper):
        return bundle.wrapper
    return bundle.backbone


def collect_trainable_parameters(bundle: SevaBundle) -> list[nn.Parameter]:
    modules: Iterable[Optional[nn.Module]] = (
        trainable_backbone_container(bundle),
        bundle.conditioner,
        bundle.ae,
    )
    params: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            ident = id(parameter)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(parameter)
    return params


def build_optimizer_from_bundle(
    bundle: SevaBundle,
    *,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> torch.optim.Optimizer:
    params = collect_trainable_parameters(bundle)
    if not params:
        raise ValueError("No trainable parameters found. Check train_backbone/train_conditioner/train_ae.")
    return torch.optim.AdamW(
        [{"params": params, "lr": lr}],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )


def make_fsdp_mixed_precision(dtype: Optional[torch.dtype]) -> Any:
    if MixedPrecision is None:
        return None
    if dtype not in {torch.float16, torch.bfloat16}:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def wrap_bundle_for_distributed(
    bundle: SevaBundle,
    *,
    args: argparse.Namespace,
    device: torch.device,
    dtype: Optional[torch.dtype],
    dist_ctx: DistributedContext,
) -> None:
    if not dist_ctx.distributed:
        return

    if bundle.train_conditioner or bundle.train_ae:
        raise NotImplementedError(
            "Distributed training currently wraps only the SEVA backbone/SGMWrapper. "
            "Keep --no-train_conditioner and --no-train_ae for multi-GPU runs."
        )

    if dist_ctx.strategy == "ddp":
        bundle.wrapper = DistributedDataParallel(
            bundle.wrapper,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )
        return

    if dist_ctx.strategy == "fsdp":
        if FullyShardedDataParallel is None or ShardingStrategy is None:
            raise RuntimeError("PyTorch FSDP is not available in this environment.")

        auto_wrap_policy = None
        if args.fsdp_min_num_params is not None and int(args.fsdp_min_num_params) > 0:
            if size_based_auto_wrap_policy is None:
                raise RuntimeError("size_based_auto_wrap_policy is not available.")
            auto_wrap_policy = partial(
                size_based_auto_wrap_policy,
                min_num_params=int(args.fsdp_min_num_params),
            )

        sharding_strategy = getattr(ShardingStrategy, args.fsdp_sharding_strategy)
        bundle.wrapper = FullyShardedDataParallel(
            bundle.wrapper,
            auto_wrap_policy=auto_wrap_policy,
            device_id=device if device.type == "cuda" else None,
            sharding_strategy=sharding_strategy,
            mixed_precision=make_fsdp_mixed_precision(dtype),
            use_orig_params=True,
            limit_all_gathers=True,
        )
        return

    raise ValueError(f"Unknown distributed strategy: {dist_ctx.strategy!r}")


def maybe_set_distributed_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def maybe_no_sync_context(module: nn.Module, *, should_sync: bool):
    if should_sync:
        return nullcontext()
    # DDP no_sync saves communication. FSDP no_sync can increase memory because
    # it delays gradient sharding, so keep FSDP synchronized for memory safety.
    if is_fsdp_module(module):
        return nullcontext()
    no_sync = getattr(module, "no_sync", None)
    if callable(no_sync):
        return no_sync()
    return nullcontext()


def clip_grad_norm_for_bundle(
    bundle: SevaBundle,
    trainable_parameters: list[nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    if is_fsdp_module(bundle.wrapper):
        return bundle.wrapper.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=max_norm)


def reduce_mean_float(value: float, device: torch.device) -> float:
    if not distributed_is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(dist.get_world_size())
    return float(tensor.detach().cpu().item())


def distributed_barrier() -> None:
    if distributed_is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if distributed_is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    dist_ctx = resolve_distributed_context(args)
    set_seed(args.seed + dist_ctx.rank)

    num_input_views = parse_num_input_views(args.num_input_views)
    dtype = resolve_dtype(args.dtype)
    autocast_dtype = resolve_dtype(args.autocast_dtype)
    betas = parse_betas(args.betas)
    effective_init_mode = resolve_effective_init_mode(args)
    bootstrap_lr = resolve_learning_rate(args.lr, effective_init_mode)

    if effective_init_mode == "resume" and args.lr is not None:
        main_print(
            "Resume mode restores optimizer state from the checkpoint; "
            "--lr is only used to bootstrap the temporary optimizer and will "
            "be overwritten by the checkpoint state."
        )
    if args.overfit_one_batch and args.log_every != 1:
        main_print("Overfit-one-batch mode active; forcing log_every=1.")
        args.log_every = 1

    if args.debug_fixed_noise_seed is not None or args.debug_fixed_noise_idx is not None:
        main_print(
            "Warning: --debug_fixed_noise_seed / --debug_fixed_noise_idx are currently "
            "no-op compatibility flags in the checked-in diffusion_loss.py. "
            "They will not make training deterministic unless diffusion_loss.py "
            "is extended to accept them."
        )

    if args.l_short is None and args.height == args.width:
        main_print(
            "Note: current clip_dataset.py uses direct resize to HxW, not official "
            "resize-cover + center-crop. For benchmark-aligned training, keep eval "
            "preprocessing matched or patch clip_dataset.py with a true center-crop mode."
        )

    device = resolve_device_for_process(args, dist_ctx)
    enable_sdpa_backends_for_cuda()

    dataset_scene_names: Optional[tuple[str, ...]] = None
    frame_selection_mode = "clip"
    training_sample_mode = args.training_sample_mode
    total_frames = int(args.total_frames)
    if args.match_render_scene is not None:
        render_match = resolve_render_match_overrides(
            dataset_root=args.dataset_root,
            split=args.train_split,
            scene_name=args.match_render_scene,
            preset=args.match_render_preset,
            requested_num_inputs=args.match_render_num_inputs,
            task=args.match_render_task,
        )
        dataset_scene_names = render_match["scene_names"]
        frame_selection_mode = render_match["frame_selection_mode"]
        training_sample_mode = "benchmark_split"
        num_input_views = render_match["num_input_views"]
        render_total_frames = int(render_match["total_frames"])
        requested_train_frames = (
            args.match_render_train_frames
            if args.match_render_train_frames is not None
            else args.total_frames
        )
        requested_train_frames = int(requested_train_frames)
        min_required_frames = int(render_match["resolved_num_input_views"]) + 1
        if requested_train_frames < min_required_frames:
            raise ValueError(
                "match-render training needs at least num_input_views + 1 frames. "
                f"num_input_views={render_match['resolved_num_input_views']}, "
                f"requested_train_frames={requested_train_frames}, "
                f"minimum={min_required_frames}."
            )
        total_frames = min(requested_train_frames, render_total_frames)
        main_print(
            "Match-render mode: "
            f"scene={render_match['resolved_scene_name']} "
            f"num_input_views={render_match['resolved_num_input_views']} "
            f"train_total_frames={total_frames} "
            f"render_total_frames={render_total_frames} "
            f"available_splits={render_match['resolved_available_num_input_views']}"
        )
        if args.match_render_task == "img2trajvid" and args.l_short is None:
            args.l_short = 576
            main_print(
                "Match-render mode: defaulting --l_short to 576 because "
                "--match_render_task img2trajvid usually uses shortest-side resize."
            )

    run_dir = args.output_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    if dist_ctx.is_main:
        ensure_dir(run_dir)
        ensure_dir(checkpoint_dir)
    distributed_barrier()

    train_loader = build_dataloader(
        dataset_root=args.dataset_root,
        split=args.train_split,
        num_input_views=num_input_views,
        total_frames=total_frames,
        height=args.height,
        width=args.width,
        l_short=args.l_short,
        normalize_intrinsics=args.normalize_intrinsics,
        shuffle=args.shuffle,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        frame_selection_mode=frame_selection_mode,
        training_sample_mode=training_sample_mode,
        small_stride_prob=args.small_stride_prob,
        clips_per_scene_per_epoch=args.clips_per_scene_per_epoch,
        scene_names=dataset_scene_names,
        distributed=dist_ctx.distributed and not args.overfit_one_batch,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
    )

    val_loader: Optional[DataLoader]
    try:
        val_loader = build_dataloader(
            dataset_root=args.dataset_root,
            split=args.val_split,
            num_input_views=num_input_views,
            total_frames=total_frames,
            height=args.height,
            width=args.width,
            l_short=args.l_short,
            normalize_intrinsics=args.normalize_intrinsics,
            shuffle=False,
            seed=args.seed + 1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            frame_selection_mode=frame_selection_mode,
            training_sample_mode=training_sample_mode,
            small_stride_prob=args.small_stride_prob,
            clips_per_scene_per_epoch=args.clips_per_scene_per_epoch,
            scene_names=dataset_scene_names,
            distributed=False,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
    except Exception as exc:
        main_print(f"Validation loader disabled: {exc}")
        val_loader = None

    first_batch = next(iter(train_loader))
    _, num_frames, _, _ = infer_batch_shape(first_batch)
    overfit_batch = first_batch if args.overfit_one_batch else None
    if overfit_batch is not None:
        scene_names = overfit_batch.get("scene_name")
        main_print(f"Overfit-one-batch mode: caching first training batch from {scene_names}")

    ae = build_ae_from_args(args)
    bundle = build_seva_bundle(
        device=device,
        dtype=dtype if dtype is not None else torch.float32,
        num_frames=num_frames,
        build_conditioner=True,
        train_backbone=args.train_backbone,
        train_conditioner=args.train_conditioner,
        ae=ae,
        train_ae=args.train_ae,
    )

    if effective_init_mode != "resume":
        initialize_backbone_weights(
            bundle=bundle,
            init_mode=effective_init_mode,
            official_model_version=args.official_model_version,
            official_pretrained_model_name_or_path=args.official_pretrained_model_name_or_path,
            official_weight_name=args.official_weight_name,
            pretrained_ckpt=args.pretrained_ckpt,
            pretrained_strict=args.pretrained_strict,
        )

    if effective_init_mode == "resume" and dist_ctx.strategy == "fsdp":
        raise NotImplementedError(
            "FSDP resume is not implemented in this lightweight trainer. "
            "Use --init_backbone_mode local_pretrained for a saved backbone, "
            "or resume in single-GPU/DDP mode."
        )

    wrap_bundle_for_distributed(
        bundle,
        args=args,
        device=device,
        dtype=dtype,
        dist_ctx=dist_ctx,
    )

    optimizer = build_optimizer_from_bundle(
        bundle,
        lr=bootstrap_lr,
        weight_decay=args.weight_decay,
        betas=betas,
        eps=args.eps,
    )
    optimizer.zero_grad(set_to_none=True)

    resume_state = ResumeState(epoch=0, global_step=0, best_val_loss=math.inf)
    if effective_init_mode == "resume":
        if args.resume is None:
            raise ValueError("Resume mode requires --resume.")
        resume_state = load_checkpoint(
            path=args.resume,
            bundle=bundle,
            optimizer=optimizer,
            device=device,
        )
        main_print(
            f"Resumed from {args.resume} at epoch={resume_state.epoch}, "
            f"step={resume_state.global_step}"
        )

    active_learning_rates = get_optimizer_learning_rates(optimizer)
    config_payload = dict(vars(args))
    config_payload["effective_init_backbone_mode"] = effective_init_mode
    config_payload["bootstrap_lr"] = bootstrap_lr
    config_payload["optimizer_learning_rates"] = active_learning_rates
    config_payload["resolved_num_input_views"] = list(num_input_views)
    config_payload["resolved_total_frames"] = total_frames
    config_payload["resolved_frame_selection_mode"] = frame_selection_mode
    config_payload["resolved_training_sample_mode"] = training_sample_mode
    config_payload["resolved_scene_names"] = (
        list(dataset_scene_names) if dataset_scene_names is not None else None
    )
    config_payload["distributed"] = {
        "strategy": dist_ctx.strategy,
        "world_size": dist_ctx.world_size,
        "rank": dist_ctx.rank,
        "local_rank": dist_ctx.local_rank,
    }
    if dist_ctx.is_main:
        save_json(run_dir / "config.json", to_jsonable(config_payload))

    set_module_modes(bundle)

    if dist_ctx.is_main:
        print("Bundle summary:")
        for key, value in summarize_bundle(bundle).items():
            print(f"  {key}: {value}")
        print(f"Resolved init mode: {effective_init_mode}")
        print(
            f"Distributed strategy: {dist_ctx.strategy} "
            f"world_size={dist_ctx.world_size}"
        )
        if effective_init_mode == "resume":
            print(f"Bootstrap optimizer learning rate: {bootstrap_lr:.2e}")
        print(f"Active optimizer learning rate(s): {format_learning_rates(active_learning_rates)}")

    log_path = run_dir / "train_log.jsonl"
    if dist_ctx.is_main and resume_state.global_step == 0 and log_path.exists():
        log_path.unlink()

    target_total_steps = max(int(args.max_steps), 1)
    if dist_ctx.is_main:
        if args.overfit_one_batch:
            print(f"Training for up to {target_total_steps} optimizer steps on one cached train batch.")
        else:
            print(
                f"Training for up to {target_total_steps} optimizer steps; "
                f"{len(train_loader)} train batches/epoch per rank."
            )

    validation_config = ValidationConfig(
        latent_downsample_factor=args.latent_downsample_factor,
        latent_scaling_factor=args.latent_scaling_factor,
        encoding_t=args.encoding_t,
        sample_posterior=args.sample_posterior,
        camera_scale=args.camera_scale,
        schedule=args.schedule,
        reference_c2ws_key="all_c2ws",
        objective=args.objective,
        target_frame_weight=args.target_frame_weight,
        input_frame_weight=args.input_frame_weight,
        val_batches=args.val_batches,
        autocast_dtype=autocast_dtype,
    )
    epi_config = build_epipolar_config_from_args(args)

    trainable_parameters = collect_trainable_parameters(bundle)
    if not trainable_parameters:
        raise ValueError("No trainable parameters found in the configured bundle.")

    global_step = resume_state.global_step
    best_val_loss = resume_state.best_val_loss
    accum_counter = 0
    stop_training = False
    train_start_time = time.time()
    last_epoch = resume_state.epoch
    first_step_after_resume = resume_state.global_step + 1

    for epoch in range(resume_state.epoch, args.epochs):
        last_epoch = epoch
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        maybe_set_distributed_epoch(train_loader, epoch)
        if args.overfit_one_batch:
            batch_iterator = enumerate(repeat(overfit_batch))
        else:
            batch_iterator = enumerate(train_loader)

        for batch_idx, batch in batch_iterator:
            iter_start = time.time()
            with make_sdpa_context(args.sdpa_backend):
                loss_out = compute_seva_diffusion_loss(
                    bundle=bundle,
                    batch=batch,
                    device=device,
                    latent_downsample_factor=args.latent_downsample_factor,
                    latent_scaling_factor=args.latent_scaling_factor,
                    encoding_t=args.encoding_t,
                    sample_posterior=args.sample_posterior,
                    camera_scale=args.camera_scale,
                    reference_c2ws_key="all_c2ws",
                    schedule=args.schedule,
                    objective=args.objective,
                    target_frame_weight=args.target_frame_weight,
                    input_frame_weight=args.input_frame_weight,
                    cfg_dropout_prob=args.cfg_dropout_prob,
                    return_unconditional=True,
                    conditioner_no_grad=not bundle.train_conditioner,
                    encoder_no_grad=not bundle.train_ae,
                    autocast_dtype=autocast_dtype,
                    include_replace_in_conditioning=True,
                    use_activation_checkpointing=args.activation_checkpointing,
                    offload_frozen_encoders=args.offload_frozen_encoders,
                    seva_noise_idx_min=args.seva_noise_idx_min,
                    seva_noise_idx_max=args.seva_noise_idx_max,
                    seva_timestep_sampling=args.seva_timestep_sampling,
                    seva_timestep_beta_alpha=args.seva_timestep_beta_alpha,
                    seva_timestep_beta_beta=args.seva_timestep_beta_beta,
                    seva_fixed_noise_idx=args.seva_fixed_noise_idx,
                )

            epi_out = None
            total_loss = loss_out.loss
            if epi_config is not None:
                if should_apply_epipolar_loss is None or compute_visibility_gated_epipolar_loss is None:
                    raise RuntimeError(
                        "Epipolar loss was requested, but train.training.epipolar_loss "
                        "could not be imported."
                    )
                if should_apply_epipolar_loss(global_step, epi_config):
                    epi_out = compute_visibility_gated_epipolar_loss(
                        bundle=bundle,
                        batch=batch,
                        loss_out=loss_out,
                        config=epi_config,
                        global_step=global_step,
                    )
                    total_loss = total_loss + epi_out.loss

            loss = total_loss / float(args.grad_accum_steps)
            will_step_after_backward = accum_counter + 1 >= args.grad_accum_steps
            with maybe_no_sync_context(bundle.wrapper, should_sync=will_step_after_backward):
                loss.backward()
            accum_counter += 1

            optimizer_stepped = False
            grad_norm_value: Optional[float] = None
            finite_grads, total_grads = count_finite_grads(trainable_backbone_container(bundle))
            if accum_counter >= args.grad_accum_steps:
                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    grad_norm = clip_grad_norm_for_bundle(
                        bundle,
                        trainable_parameters,
                        max_norm=float(args.grad_clip_norm),
                    )
                    grad_norm_value = float(grad_norm.detach().cpu().item())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                global_step += 1
                optimizer_stepped = True

            iter_time = time.time() - iter_start
            should_log_step = optimizer_stepped and (
                global_step == 1
                or global_step == first_step_after_resume
                or (args.log_every > 0 and global_step % args.log_every == 0)
            )
            if should_log_step and dist_ctx.is_main:
                batch_size, frames, height, width = infer_batch_shape(batch)
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "loss": total_loss.detach().float().item(),
                    "loss_diffusion": loss_out.loss.detach().float().item(),
                    "mse_mean": loss_out.mse_per_item.detach().float().mean().item(),
                    "weights_mean": loss_out.weights.detach().float().mean().item(),
                    "noise_idx_min": float(loss_out.timesteps.detach().float().min().item()),
                    "noise_idx_max": float(loss_out.timesteps.detach().float().max().item()),
                    "noise_idx_mean": float(loss_out.timesteps.detach().float().mean().item()),
                    "sigma_min": float(loss_out.sigma.detach().float().min().item()),
                    "sigma_max": float(loss_out.sigma.detach().float().max().item()),
                    "sigma_mean": float(loss_out.sigma.detach().float().mean().item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "batch_size": batch_size,
                    "num_frames": frames,
                    "height": height,
                    "width": width,
                    "iter_time_sec": iter_time,
                    "finite_backbone_grads": finite_grads,
                    "total_backbone_grads": total_grads,
                    "overfit_one_batch": bool(args.overfit_one_batch),
                }
                if "scene_name" in batch:
                    record["scene_name"] = batch["scene_name"]
                if epi_out is not None:
                    record["loss_epipolar"] = epi_out.loss.detach().float().item()
                    record["epi_raw_loss"] = epi_out.raw_loss.detach().float().item()
                    record["epi_warmup_factor"] = float(epi_out.warmup_factor)
                    record["epi_mean_sigma_gate"] = epi_out.mean_sigma_gate.detach().float().item()
                    record["epi_mean_confidence"] = epi_out.mean_confidence.detach().float().item()
                    record["epi_mean_valid_ratio"] = epi_out.mean_valid_ratio.detach().float().item()
                    record["epi_mean_baseline"] = epi_out.mean_baseline.detach().float().item()
                    record["epi_mean_rotation_deg"] = epi_out.mean_rotation_deg.detach().float().item()
                    record["epi_num_pairs"] = int(epi_out.num_pairs)
                    record["epi_num_target_frames"] = int(epi_out.num_target_frames)
                    record["epi_num_pixels"] = int(epi_out.num_pixels)
                    record["epi_num_epipolar_pairs"] = int(epi_out.num_epipolar_pairs)
                    record["epi_num_homography_pairs"] = int(epi_out.num_homography_pairs)
                    record["epi_num_skipped_pairs"] = int(epi_out.num_skipped_pairs)
                if grad_norm_value is not None:
                    record["grad_norm"] = grad_norm_value
                if device.type == "cuda":
                    record["cuda_mem_alloc_mb"] = (
                        torch.cuda.memory_allocated(device) / (1024 ** 2)
                    )
                    record["cuda_mem_reserved_mb"] = (
                        torch.cuda.memory_reserved(device) / (1024 ** 2)
                    )
                append_jsonl(log_path, record)
                elapsed_minutes = (time.time() - train_start_time) / 60.0
                epi_fragment = (
                    f" epi={record['loss_epipolar']:.6f}"
                    if "loss_epipolar" in record
                    else ""
                )
                print(
                    f"step={global_step:06d} epoch={epoch:03d} batch={batch_idx:05d} "
                    f"loss={record['loss']:.6f}{epi_fragment} lr={record['lr']:.2e} "
                    f"iter={iter_time:.2f}s elapsed={elapsed_minutes:.1f}m"
                )

            if optimizer_stepped and args.val_every > 0 and global_step % args.val_every == 0:
                val_loss = evaluate(
                    bundle=bundle,
                    loader=val_loader,
                    device=device,
                    config=validation_config,
                    sdpa_context_factory=lambda: make_sdpa_context(args.sdpa_backend),
                )
                if val_loss is not None:
                    val_loss = reduce_mean_float(float(val_loss), device)
                    if dist_ctx.is_main:
                        print(f"validation: step={global_step:06d} val_loss={val_loss:.6f}")
                        append_jsonl(
                            log_path,
                            {
                                "type": "val",
                                "epoch": epoch,
                                "global_step": global_step,
                                "val_loss": float(val_loss),
                            },
                        )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            path=checkpoint_dir / "best.pt",
                            bundle=bundle,
                            optimizer=optimizer,
                            epoch=epoch,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            args=config_payload,
                        )
                        if dist_ctx.is_main:
                            print(f"saved new best checkpoint: {checkpoint_dir / 'best.pt'}")

            if optimizer_stepped and args.save_every > 0 and global_step % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(
                    path=checkpoint_path,
                    bundle=bundle,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    args=config_payload,
                )
                if dist_ctx.is_main:
                    print(f"saved checkpoint: {checkpoint_path}")

            if optimizer_stepped and global_step >= target_total_steps:
                stop_training = True
                break

        if stop_training:
            break

    final_checkpoint = checkpoint_dir / "last.pt"
    save_checkpoint(
        path=final_checkpoint,
        bundle=bundle,
        optimizer=optimizer,
        epoch=last_epoch,
        global_step=global_step,
        best_val_loss=best_val_loss,
        args=config_payload,
    )
    if dist_ctx.is_main:
        print(f"saved final checkpoint: {final_checkpoint}")
        print("training complete")
    cleanup_distributed()


if __name__ == "__main__":
    main()
