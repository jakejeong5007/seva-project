from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from train.training.model_factory import SevaBundle
from train.utils.logging_utils import ensure_dir, to_jsonable


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    global_step: int
    best_val_loss: float


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


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "backbone", "module"):
            value = payload.get(key)
            if isinstance(value, dict):
                return extract_state_dict(value)
        if payload and all(isinstance(key, str) for key in payload):
            return payload  # type: ignore[return-value]
    raise TypeError("Could not extract a state_dict from the provided checkpoint payload.")


def load_backbone_checkpoint_into_bundle(
    *,
    bundle: SevaBundle,
    checkpoint_path: Path,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(payload)
    for prefix in ("module.", "wrapper.", "backbone.", "model."):
        state_dict = _strip_uniform_prefix(state_dict, prefix)
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
            print("  first missing keys:", missing[:10])
        if unexpected:
            print("  first unexpected keys:", unexpected[:10])
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
            print("  first missing keys:", missing[:10])
        if unexpected:
            print("  first unexpected keys:", unexpected[:10])
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
    ensure_dir(path.parent)
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": float(best_val_loss),
        "backbone": bundle.backbone.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": to_jsonable(dict(args)),
    }
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
    checkpoint = torch.load(path, map_location=device)
    bundle.backbone.load_state_dict(checkpoint["backbone"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
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
