from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from train.training.model_factory import SevaBundle
from train.utils.logging_utils import ensure_dir, to_jsonable

try:
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel,
        StateDictType,
    )
except Exception:  # pragma: no cover
    FullStateDictConfig = None
    FullyShardedDataParallel = None
    StateDictType = None


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_val_loss: float


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_rank() -> int:
    if distributed_is_initialized():
        return int(dist.get_rank())
    return 0


def is_main_process() -> bool:
    return distributed_rank() == 0


def is_ddp_module(module: torch.nn.Module) -> bool:
    return isinstance(module, DistributedDataParallel)


def is_fsdp_module(module: torch.nn.Module) -> bool:
    return FullyShardedDataParallel is not None and isinstance(
        module, FullyShardedDataParallel
    )


def move_optimizer_state_(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _strip_uniform_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _strip_known_wrapper_prefixes(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for prefix in (
        "module.module.",      # DDP(SGMWrapper(Seva))
        "module.",             # SGMWrapper(Seva) or DDP raw
        "wrapper.",
        "backbone.",
        "model.",
        "_fsdp_wrapped_module.module.",
        "_fsdp_wrapped_module.",
    ):
        state_dict = _strip_uniform_prefix(state_dict, prefix)
    return state_dict


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "backbone", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                return extract_state_dict(value)
        if payload and all(isinstance(key, str) for key in payload):
            return payload  # type: ignore[return-value]
    raise TypeError("Could not extract a state_dict from the provided checkpoint payload.")


def _raw_backbone_state_dict_from_bundle(bundle: SevaBundle) -> dict[str, torch.Tensor]:
    """Return a raw SEVA backbone state_dict suitable for demo.py --backbone_ckpt.

    In FSDP mode every rank must enter this function because FULL_STATE_DICT
    collection uses distributed collectives. Only rank 0 receives a populated
    dictionary when rank0_only=True.
    """
    wrapper = bundle.wrapper

    if is_fsdp_module(wrapper):
        if FullStateDictConfig is None or StateDictType is None:
            raise RuntimeError("FSDP state_dict support is not available.")
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(  # type: ignore[union-attr]
            wrapper,
            StateDictType.FULL_STATE_DICT,
            cfg,
        ):
            state = dict(wrapper.state_dict())
        if not state:
            return {}
        return _strip_known_wrapper_prefixes(state)

    if is_ddp_module(wrapper):
        # DDP.module is SGMWrapper, whose .module is the raw Seva backbone.
        ddp_inner = wrapper.module
        if hasattr(ddp_inner, "module"):
            return ddp_inner.module.state_dict()
        return _strip_known_wrapper_prefixes(dict(ddp_inner.state_dict()))

    return dict(bundle.backbone.state_dict())


def load_backbone_checkpoint_into_bundle(
    *,
    bundle: SevaBundle,
    checkpoint_path: Path,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _strip_known_wrapper_prefixes(extract_state_dict(payload))
    missing, unexpected = bundle.backbone.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def initialize_backbone_weights(
    *,
    bundle: SevaBundle,
    init_mode: str,
    official_model_version: float,
    official_pretrained_model_name_or_path: str,
    official_weight_name: str,
    pretrained_ckpt: Optional[Path],
    pretrained_strict: bool,
) -> None:
    if init_mode == "scratch":
        print("Backbone init: scratch")
        return

    if init_mode == "official":
        from seva.utils import load_model as load_official_seva_model

        print(
            f"Backbone init: official SEVA weights v{official_model_version} "
            f"from {official_pretrained_model_name_or_path}"
        )
        official_model = load_official_seva_model(
            model_version=official_model_version,
            pretrained_model_name_or_path=official_pretrained_model_name_or_path,
            weight_name=official_weight_name,
            device="cpu",
            verbose=True,
        )
        missing, unexpected = bundle.backbone.load_state_dict(
            official_model.state_dict(),
            strict=pretrained_strict,
        )
        print(
            f"Official load complete: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(" first missing keys:", missing[:10])
        if unexpected:
            print(" first unexpected keys:", unexpected[:10])
        del official_model
        return

    if init_mode == "local_pretrained":
        if pretrained_ckpt is None:
            raise ValueError(
                "--pretrained_ckpt is required when --init_backbone_mode local_pretrained"
            )
        print(f"Backbone init: local pretrained weights from {pretrained_ckpt}")
        missing, unexpected = load_backbone_checkpoint_into_bundle(
            bundle=bundle,
            checkpoint_path=pretrained_ckpt,
            strict=pretrained_strict,
        )
        print(
            f"Local pretrained load complete: missing={len(missing)} unexpected={len(unexpected)}"
        )
        if missing:
            print(" first missing keys:", missing[:10])
        if unexpected:
            print(" first unexpected keys:", unexpected[:10])
        return

    if init_mode == "resume":
        return

    raise ValueError(f"Unknown init mode: {init_mode!r}")


def save_checkpoint(
    *,
    path: Path,
    bundle: SevaBundle,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: Mapping[str, Any],
) -> None:
    # FSDP full-state collection is collective; every rank must call this.
    fsdp_mode = is_fsdp_module(bundle.wrapper)
    if not fsdp_mode and not is_main_process():
        return

    backbone_state = _raw_backbone_state_dict_from_bundle(bundle)

    if not is_main_process():
        return

    ensure_dir(path.parent)
    checkpoint: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "backbone": backbone_state,
        "args": to_jsonable(dict(args)),
    }

    # For FSDP, this lightweight trainer saves an inference/fine-tune friendly
    # full backbone checkpoint. Full optimizer resume for sharded optimizer state
    # is intentionally omitted to avoid fragile, version-specific state dict code.
    checkpoint["optimizer"] = None if fsdp_mode else optimizer.state_dict()

    if bundle.conditioner is not None:
        checkpoint["conditioner"] = bundle.conditioner.state_dict()

    if bundle.ae is not None:
        try:
            checkpoint["ae"] = bundle.ae.state_dict()
        except Exception:
            pass

    torch.save(checkpoint, path)


def load_checkpoint(
    *,
    path: Path,
    bundle: SevaBundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> ResumeState:
    if is_fsdp_module(bundle.wrapper):
        raise NotImplementedError(
            "FSDP resume is not implemented in this lightweight trainer. "
            "Use --init_backbone_mode local_pretrained with a saved checkpoint, "
            "or resume in single-GPU/DDP mode."
        )

    checkpoint = torch.load(path, map_location=device)
    backbone_state = _strip_known_wrapper_prefixes(checkpoint["backbone"])
    bundle.backbone.load_state_dict(backbone_state, strict=True)

    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        move_optimizer_state_(optimizer, device)

    if bundle.conditioner is not None and "conditioner" in checkpoint:
        bundle.conditioner.load_state_dict(checkpoint["conditioner"], strict=False)

    if bundle.ae is not None and "ae" in checkpoint:
        try:
            bundle.ae.load_state_dict(checkpoint["ae"], strict=False)
        except Exception:
            pass

    return ResumeState(
        epoch=int(checkpoint.get("epoch", 0)),
        global_step=int(checkpoint.get("global_step", 0)),
        best_val_loss=float(checkpoint.get("best_val_loss", float("inf"))),
    )
