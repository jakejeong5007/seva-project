from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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
from train.training.model_factory import (
    SevaBundle,
    build_optimizer,
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

    parser.add_argument("--epochs", type=int, default=1)
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
        help="Debug only: fix Gaussian noise sampling for deterministic overfit tests.",
    )

    parser.add_argument(
        "--debug_fixed_noise_idx",
        type=int,
        default=None,
        help="Debug only: fix SEVA/DDPM noise index for deterministic overfit tests.",
    )
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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
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
        "frame_selection_mode": "img2trajvid",
        "num_input_views": (selected_num_inputs,),
        "total_frames": metadata["num_total_frames"],
        "resolved_scene_dir": scene_dir,
        "resolved_scene_name": scene_name,
        "resolved_num_input_views": selected_num_inputs,
        "resolved_available_num_input_views": available,
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
        bundle.backbone.train()
    else:
        bundle.backbone.eval()

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


def collect_trainable_parameters(bundle: SevaBundle) -> list[nn.Parameter]:
    modules: Iterable[Optional[nn.Module]] = (
        bundle.backbone,
        bundle.conditioner,
        bundle.ae,
    )
    params: list[nn.Parameter] = []
    for module in modules:
        if module is None:
            continue
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return params


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    num_input_views = parse_num_input_views(args.num_input_views)
    dtype = resolve_dtype(args.dtype)
    autocast_dtype = resolve_dtype(args.autocast_dtype)
    betas = parse_betas(args.betas)
    effective_init_mode = resolve_effective_init_mode(args)
    bootstrap_lr = resolve_learning_rate(args.lr, effective_init_mode)

    if effective_init_mode == "resume" and args.lr is not None:
        print(
            "Resume mode restores optimizer state from the checkpoint; "
            "--lr is only used to bootstrap the temporary optimizer and will "
            "be overwritten by the checkpoint state."
        )
    if args.overfit_one_batch and args.log_every != 1:
        print("Overfit-one-batch mode active; forcing log_every=1.")
        args.log_every = 1

    device = torch.device(args.device) if args.device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
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
        )
        dataset_scene_names = render_match["scene_names"]
        frame_selection_mode = render_match["frame_selection_mode"]
        training_sample_mode = "benchmark_split"
        num_input_views = render_match["num_input_views"]
        total_frames = int(render_match["total_frames"])
        print(
            "Match-render mode: "
            f"scene={render_match['resolved_scene_name']} "
            f"num_input_views={render_match['resolved_num_input_views']} "
            f"total_frames={total_frames} "
            f"available_splits={render_match['resolved_available_num_input_views']}"
        )
        if args.l_short is None:
            args.l_short = 576
            print("Match-render mode: defaulting --l_short to 576 to mirror demo rendering.")

    run_dir = args.output_dir / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(checkpoint_dir)

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
        )
    except Exception as exc:
        print(f"Validation loader disabled: {exc}")
        val_loader = None

    first_batch = next(iter(train_loader))
    _, num_frames, _, _ = infer_batch_shape(first_batch)
    overfit_batch = first_batch if args.overfit_one_batch else None
    if overfit_batch is not None:
        scene_names = overfit_batch.get("scene_name")
        print(f"Overfit-one-batch mode: caching first training batch from {scene_names}")

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

    optimizer = build_optimizer(
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
        print(
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
    save_json(run_dir / "config.json", to_jsonable(config_payload))

    set_module_modes(bundle)

    print("Bundle summary:")
    for key, value in summarize_bundle(bundle).items():
        print(f"  {key}: {value}")
    print(f"Resolved init mode: {effective_init_mode}")
    if effective_init_mode == "resume":
        print(f"Bootstrap optimizer learning rate: {bootstrap_lr:.2e}")
    print(f"Active optimizer learning rate(s): {format_learning_rates(active_learning_rates)}")

    log_path = run_dir / "train_log.jsonl"
    if resume_state.global_step == 0 and log_path.exists():
        log_path.unlink()

    target_total_steps = max(int(args.max_steps), 1)
    if args.overfit_one_batch:
        print(f"Training for up to {target_total_steps} optimizer steps on one cached train batch.")
    else:
        print(
            f"Training for up to {target_total_steps} optimizer steps; "
            f"{len(train_loader)} train batches/epoch."
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
                    debug_fixed_noise_seed=args.debug_fixed_noise_seed,
                    debug_fixed_noise_idx=args.debug_fixed_noise_idx,
                )

            loss = loss_out.loss / float(args.grad_accum_steps)
            loss.backward()
            accum_counter += 1

            optimizer_stepped = False
            grad_norm_value: Optional[float] = None
            finite_grads, total_grads = count_finite_grads(bundle.backbone)
            if accum_counter >= args.grad_accum_steps:
                if args.grad_clip_norm is not None and args.grad_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
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
            if should_log_step:
                batch_size, frames, height, width = infer_batch_shape(batch)
                record: dict[str, Any] = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "global_step": global_step,
                    "loss": loss_out.loss.detach().float().item(),
                    "mse_mean": loss_out.mse_per_item.detach().float().mean().item(),
                    "weights_mean": loss_out.weights.detach().float().mean().item(),
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
                print(
                    f"step={global_step:06d} epoch={epoch:03d} batch={batch_idx:05d} "
                    f"loss={record['loss']:.6f} lr={record['lr']:.2e} "
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
    print(f"saved final checkpoint: {final_checkpoint}")
    print("training complete")


if __name__ == "__main__":
    main()
