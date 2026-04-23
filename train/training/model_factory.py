# File: model_factory.py
# Description: Utilities for instantiating the released SEVA backbone,
#              optional CLIP image conditioner, and a forward-pass smoke test.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn

try:
    from seva.model import Seva, SevaParams, SGMWrapper
    from seva.modules.conditioner import CLIPConditioner
except ImportError as e:
    raise ImportError(
        "Could not import SEVA modules. Make sure this file is inside the "
        "stable-virtual-camera project or that the repo is on PYTHONPATH."
    ) from e


@dataclass
class SevaBundle:
    """Container for the main training-time modules.

    Attributes:
        params:          The instantiated SevaParams object.
        backbone:        The raw Seva backbone.
        wrapper:         SGMWrapper around the backbone.
        conditioner:     Optional CLIP image conditioner.
        ae:              Optional autoencoder / VAE module.
        device:          Device on which modules were placed.
        dtype:           Main dtype used for the backbone.
        train_backbone:  Whether the backbone is trainable.
        train_conditioner: Whether the conditioner is trainable.
        train_ae:        Whether the autoencoder is trainable.
    """

    params: SevaParams
    backbone: Seva
    wrapper: SGMWrapper
    conditioner: Optional[nn.Module]
    ae: Optional[nn.Module]
    device: torch.device
    dtype: torch.dtype
    train_backbone: bool
    train_conditioner: bool
    train_ae: bool


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _freeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    module.eval()
    module.requires_grad_(False)


def _unfreeze_module(module: Optional[nn.Module]) -> None:
    if module is None:
        return
    module.train()
    module.requires_grad_(True)


def _to_dtype(module: Optional[nn.Module], dtype: torch.dtype) -> None:
    if module is None:
        return
    module.to(dtype=dtype)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    total = 0
    for p in module.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def build_seva_params(
    *,
    num_frames: int = 21,
    overrides: Optional[Mapping[str, Any]] = None,
) -> SevaParams:
    """Build a SevaParams object with safe override handling.

    Examples:
        params = build_seva_params(num_frames=8)
        params = build_seva_params(overrides={"model_channels": 192})
    """
    kwargs: Dict[str, Any] = {"num_frames": int(num_frames)}
    if overrides is not None:
        kwargs.update(dict(overrides))
    return SevaParams(**kwargs)


def build_seva_bundle(
    *,
    device: Optional[torch.device | str] = None,
    dtype: torch.dtype = torch.float32,
    num_frames: int = 21,
    model_overrides: Optional[Mapping[str, Any]] = None,
    build_conditioner: bool = True,
    train_backbone: bool = True,
    train_conditioner: bool = False,
    ae: Optional[nn.Module] = None,
    train_ae: bool = False,
) -> SevaBundle:
    """Instantiate the released SEVA backbone and optional helper modules.

    This is the main entrypoint you should call from train.py later.
    It does not assume any particular autoencoder implementation; you can pass
    one in when you have chosen your VAE wrapper.
    """
    resolved_device = _resolve_device(device)
    params = build_seva_params(num_frames=num_frames, overrides=model_overrides)

    backbone = Seva(params)
    wrapper = SGMWrapper(backbone)

    backbone.to(device=resolved_device, dtype=dtype)
    wrapper.to(device=resolved_device, dtype=dtype)

    if train_backbone:
        _unfreeze_module(backbone)
        wrapper.train()
    else:
        _freeze_module(backbone)
        wrapper.eval()

    conditioner: Optional[nn.Module]
    if build_conditioner:
        conditioner = CLIPConditioner().to(device=resolved_device)
        if train_conditioner:
            _unfreeze_module(conditioner)
        else:
            _freeze_module(conditioner)
    else:
        conditioner = None

    if ae is not None:
        ae.to(device=resolved_device)
        if train_ae:
            _unfreeze_module(ae)
        else:
            _freeze_module(ae)

    return SevaBundle(
        params=params,
        backbone=backbone,
        wrapper=wrapper,
        conditioner=conditioner,
        ae=ae,
        device=resolved_device,
        dtype=dtype,
        train_backbone=bool(train_backbone),
        train_conditioner=bool(train_conditioner),
        train_ae=bool(train_ae),
    )


def summarize_bundle(bundle: SevaBundle) -> Dict[str, Any]:
    """Return a compact summary that is easy to print while debugging."""
    conditioner_params = 0 if bundle.conditioner is None else count_parameters(bundle.conditioner)
    ae_params = 0 if bundle.ae is None else count_parameters(bundle.ae)

    return {
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "num_frames": bundle.params.num_frames,
        "backbone_params_total": count_parameters(bundle.backbone),
        "backbone_params_trainable": count_parameters(bundle.backbone, trainable_only=True),
        "conditioner_present": bundle.conditioner is not None,
        "conditioner_params_total": conditioner_params,
        "ae_present": bundle.ae is not None,
        "ae_params_total": ae_params,
        "train_backbone": bundle.train_backbone,
        "train_conditioner": bundle.train_conditioner,
        "train_ae": bundle.train_ae,
    }


def build_optimizer(
    bundle: SevaBundle,
    *,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """Create a simple AdamW optimizer over all trainable parameters."""
    param_groups = []

    backbone_params = [p for p in bundle.backbone.parameters() if p.requires_grad]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr})

    if bundle.conditioner is not None:
        cond_params = [p for p in bundle.conditioner.parameters() if p.requires_grad]
        if cond_params:
            param_groups.append({"params": cond_params, "lr": lr})

    if bundle.ae is not None:
        ae_params = [p for p in bundle.ae.parameters() if p.requires_grad]
        if ae_params:
            param_groups.append({"params": ae_params, "lr": lr})

    if not param_groups:
        raise ValueError("No trainable parameters found. Check train_backbone/train_conditioner/train_ae.")

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def infer_batch_shape(batch: Dict[str, torch.Tensor]) -> tuple[int, int, int, int]:
    """Infer [B, T, H, W] from a clip batch."""
    if "imgs" not in batch:
        raise KeyError("Batch must contain 'imgs'.")
    imgs = batch["imgs"]
    if imgs.dim() == 4:
        T, _, H, W = imgs.shape
        return 1, T, H, W
    if imgs.dim() == 5:
        B, T, _, H, W = imgs.shape
        return B, T, H, W
    raise ValueError(f"Expected imgs to have shape [T,3,H,W] or [B,T,3,H,W], got {tuple(imgs.shape)}")


def make_random_latents_for_batch(
    batch: Dict[str, torch.Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    latent_channels: int = 4,
    latent_downsample_factor: int = 8,
) -> torch.Tensor:
    """Create random latent tensors matching the batch's spatial size."""
    B, T, H, W = infer_batch_shape(batch)
    if H % latent_downsample_factor != 0 or W % latent_downsample_factor != 0:
        raise ValueError(
            f"Image size {(H, W)} must be divisible by latent_downsample_factor={latent_downsample_factor}."
        )
    h = H // latent_downsample_factor
    w = W // latent_downsample_factor
    return torch.randn(B * T, latent_channels, h, w, device=device, dtype=dtype)


def make_random_timesteps(
    num_items: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    """Create a simple continuous timestep / sigma tensor for smoke tests."""
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    return torch.rand(num_items, device=device, dtype=dtype) * (high - low) + low


def smoke_test_forward(
    *,
    bundle: SevaBundle,
    batch: Dict[str, torch.Tensor],
    flat_conditioning: Dict[str, torch.Tensor],
    latent_downsample_factor: int = 8,
    latent_channels: int = 4,
    timestep_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run one real forward pass through SGMWrapper.

    Args:
        bundle:
            Output of build_seva_bundle(...).
        batch:
            Real clip batch from clip_dataset.py or its collate function.
        flat_conditioning:
            Output of flatten_conditioning_for_model(...).
        latent_downsample_factor:
            Spatial downsample used by the latent model, usually 8.
        latent_channels:
            Latent channel count, usually 4.
    Returns:
        The predicted latent tensor of shape [B*T, 4, h, w].
    """
    B, T, H, W = infer_batch_shape(batch)
    num_items = B * T
    device = bundle.device

    x = make_random_latents_for_batch(
        batch,
        device=device,
        dtype=bundle.dtype,
        latent_channels=latent_channels,
        latent_downsample_factor=latent_downsample_factor,
    )
    t = make_random_timesteps(
        num_items,
        device=device,
        dtype=timestep_dtype or bundle.dtype,
    )

    moved_c: Dict[str, torch.Tensor] = {}
    for key, value in flat_conditioning.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Conditioning entry {key!r} must be a tensor.")
        moved_c[key] = value.to(device=device, dtype=bundle.dtype)

    required_keys = {"crossattn", "concat", "dense_vector"}
    missing = required_keys - set(moved_c.keys())
    if missing:
        raise KeyError(f"flat_conditioning is missing required keys: {sorted(missing)}")

    bundle.wrapper.eval()
    with torch.no_grad():
        pred = bundle.wrapper(x, t, c=moved_c, num_frames=T)

    h = H // latent_downsample_factor
    w = W // latent_downsample_factor
    expected_shape = (num_items, latent_channels, h, w)
    if tuple(pred.shape) != expected_shape:
        raise RuntimeError(
            f"SEVA forward produced shape {tuple(pred.shape)} but expected {expected_shape}."
        )
    return pred


if __name__ == "__main__":
    bundle = build_seva_bundle(
        device=None,
        dtype=torch.float32,
        num_frames=8,
        build_conditioner=False,
        train_backbone=False,
    )
    print(summarize_bundle(bundle))
