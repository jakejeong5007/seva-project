# File: conditioning.py
# Description: SEVA training-time conditioning utilities built on top of the
#              released SEVA geometry and CLIP image-conditioner modules.

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from seva.geometry import get_plucker_coordinates
    from seva.modules.conditioner import CLIPConditioner
except ImportError as e:
    raise ImportError(
        "Could not import SEVA modules. Make sure this file is placed inside the "
        "SEVA project or that the SEVA repo is on PYTHONPATH."
    ) from e


TensorDict = Dict[str, torch.Tensor]


@dataclass
class ConditioningOutput:
    """
    Container for training-time SEVA conditioning.

    Attributes:
        c:
            Conditional conditioning dict for the model. Tensors are batched as
            [B, T, ...] unless flatten_conditioning_for_model() is called.
        uc:
            Optional unconditional conditioning dict, useful for validation or
            classifier-free guidance style sampling.
        value_dict:
            A batched training-side mirror of the released eval.py value_dict.
        additional_model_inputs:
            Extra kwargs to pass into the model, most importantly num_frames=T.
    """

    c: TensorDict
    uc: Optional[TensorDict]
    value_dict: TensorDict
    additional_model_inputs: Dict[str, Any]


def _require_tensor(batch: Dict[str, Any], key: str) -> torch.Tensor:
    if key not in batch:
        raise KeyError(f"Batch is missing required key: {key!r}")
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Batch[{key!r}] must be a torch.Tensor, got {type(value)}")
    return value


def _to_hom_pose_any(pose: torch.Tensor) -> torch.Tensor:
    """
    Convert (..., 3, 4) poses to (..., 4, 4). Leave (..., 4, 4) unchanged.

    The released seva.geometry.to_hom_pose helper handles [T, 3, 4], but here
    we also want to support [B, T, 3, 4].
    """
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected pose shape (..., 3, 4) or (..., 4, 4), got {tuple(pose.shape)}"
        )

    out = torch.zeros(*pose.shape[:-2], 4, 4, dtype=pose.dtype, device=pose.device)
    out[..., :3, :] = pose
    out[..., 3, 3] = 1.0
    return out


def _maybe_unsqueeze_batch(batch: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """
    Accept either a single sample from SevaClipDataset or a collated batch.

    Single sample:
        imgs: [T, 3, H, W]

    Collated batch:
        imgs: [B, T, 3, H, W]
    """
    imgs = _require_tensor(batch, "imgs")
    single_sample = imgs.dim() == 4

    if not single_sample:
        return batch, False

    expanded: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if key in {"imgs", "Ks", "c2ws", "input_mask", "frame_ids"}:
                expanded[key] = value.unsqueeze(0)
            elif key in {"scene_scale", "num_input_views"}:
                expanded[key] = value.reshape(1)
            else:
                expanded[key] = value
        else:
            expanded[key] = value
    return expanded, True


def _maybe_squeeze_conditioning_dict(c: TensorDict, squeeze_batch: bool) -> TensorDict:
    if not squeeze_batch:
        return c
    out: TensorDict = {}
    for key, value in c.items():
        out[key] = value.squeeze(0)
    return out


def _maybe_squeeze_value_dict(value_dict: TensorDict, squeeze_batch: bool) -> TensorDict:
    if not squeeze_batch:
        return value_dict
    out: TensorDict = {}
    for key, value in value_dict.items():
        if torch.is_tensor(value):
            out[key] = value.squeeze(0)
        else:
            out[key] = value
    return out


def _resolve_device(batch: Dict[str, Any], device: Optional[torch.device | str]) -> torch.device:
    imgs = _require_tensor(batch, "imgs")
    if device is None:
        return imgs.device
    return torch.device(device)


def _move_batch_tensors(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _choose_camera_scale(
    batch: Dict[str, Any],
    camera_scale: Optional[float],
) -> float:
    if camera_scale is not None:
        return float(camera_scale)

    # Fallback only. The released eval path typically uses a constant option,
    # commonly 2.0, rather than the parser's scene_scale.
    if "scene_scale" in batch and isinstance(batch["scene_scale"], torch.Tensor):
        if batch["scene_scale"].numel() > 0:
            return float(batch["scene_scale"].reshape(-1)[0].item())
    return 2.0


def _build_value_dict_single(
    imgs: torch.Tensor,
    Ks: torch.Tensor,
    c2ws: torch.Tensor,
    input_mask: torch.Tensor,
    camera_scale: float,
    latent_downsample_factor: int,
    reference_c2ws: Optional[torch.Tensor] = None,
) -> TensorDict:
    """
    Build one sample's value_dict in the style of the released eval pipeline.

    Args:
        imgs: [T, 3, H, W] in [-1, 1]
        Ks: [T, 3, 3]
        c2ws: [T, 3, 4] or [T, 4, 4]
        input_mask: [T] bool
        camera_scale: target translation normalization scale
        latent_downsample_factor: usually 8 for SD-style VAEs
        reference_c2ws: optional reference camera set for centering statistics.
            When omitted, the current clip cameras are used.
    """
    if imgs.dim() != 4:
        raise ValueError(f"imgs must have shape [T, 3, H, W], got {tuple(imgs.shape)}")
    if Ks.dim() != 3 or Ks.shape[-2:] != (3, 3):
        raise ValueError(f"Ks must have shape [T, 3, 3], got {tuple(Ks.shape)}")
    if input_mask.dim() != 1:
        raise ValueError(f"input_mask must have shape [T], got {tuple(input_mask.shape)}")
    if imgs.shape[0] != Ks.shape[0] or imgs.shape[0] != c2ws.shape[0]:
        raise ValueError(
            "imgs, Ks, and c2ws must agree on T. "
            f"Got T={imgs.shape[0]}, {Ks.shape[0]}, {c2ws.shape[0]}"
        )
    if imgs.shape[0] != input_mask.shape[0]:
        raise ValueError(
            f"imgs and input_mask must agree on T. Got {imgs.shape[0]} and {input_mask.shape[0]}"
        )
    if not torch.any(input_mask):
        raise ValueError("input_mask must contain at least one True entry.")

    T, _, H, W = imgs.shape
    if H % latent_downsample_factor != 0 or W % latent_downsample_factor != 0:
        raise ValueError(
            f"Image size {(H, W)} must be divisible by latent_downsample_factor="
            f"{latent_downsample_factor}."
        )

    c2w = _to_hom_pose_any(c2ws.float())
    w2c = torch.linalg.inv(c2w)

    ref_c2ws = c2w if reference_c2ws is None else _to_hom_pose_any(reference_c2ws.float())

    ref_centers = ref_c2ws[:, :3, 3]
    camera_dist_2med = torch.norm(
        ref_centers - ref_centers.median(0, keepdim=True).values,
        dim=-1,
    )
    valid_mask = camera_dist_2med <= torch.clamp(
        torch.quantile(camera_dist_2med, 0.97) * 10,
        max=1e6,
    )
    if torch.any(valid_mask):
        center = ref_centers[valid_mask].mean(0, keepdim=True)
    else:
        center = ref_centers.mean(0, keepdim=True)

    c2w = c2w.clone()
    c2w[:, :3, 3] -= center
    w2c = torch.linalg.inv(c2w)

    first_input_idx = int(torch.nonzero(input_mask, as_tuple=False)[0].item())
    camera_dists = c2w[:, :3, 3].clone()
    anchor_norm = torch.norm(camera_dists[first_input_idx])
    if torch.isclose(anchor_norm, torch.zeros(1, device=anchor_norm.device), atol=1e-5).any():
        translation_scaling_factor = float(camera_scale)
    else:
        translation_scaling_factor = float(camera_scale) / float(anchor_norm.item())

    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    Ks_norm = _ensure_normalized_intrinsics(Ks, (H, W))

    pp = Ks_norm[..., :2, 2]
    if not (torch.all(pp >= 0) and torch.all(pp <= 1)):
        raise ValueError(
            "Intrinsics normalization failed. "
            f"Principal points range: min={pp.min().item():.6f}, max={pp.max().item():.6f}"
        )

    plucker_coordinate = get_plucker_coordinates(
        extrinsics_src=w2c[first_input_idx],
        extrinsics=w2c,
        intrinsics=Ks_norm,
        target_size=(H // latent_downsample_factor, W // latent_downsample_factor),
    )

    return {
        "cond_frames": imgs,
        "cond_frames_mask": input_mask.bool(),
        "cond_aug": imgs.new_tensor(0.0),
        "plucker_coordinate": plucker_coordinate,
        "c2w": c2w,
        "K": Ks.float(),
        "camera_mask": input_mask.bool(),
    }

def _ensure_normalized_intrinsics(
    K: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Convert intrinsics to resolution-independent normalized image coordinates.

    Expected input:
        K: [..., 3, 3]
        image_hw: (H, W) of the CURRENT loaded/resized images

    Output:
        K_norm: same shape as K, with:
            fx normalized by W
            fy normalized by H
            cx normalized by W
            cy normalized by H
    """
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape [..., 3, 3], got {tuple(K.shape)}")

    H, W = image_hw
    K = K.clone().float()

    principal_points = K[..., :2, 2]

    # If already normalized, keep as-is.
    if torch.all(principal_points >= 0) and torch.all(principal_points <= 1):
        return K

    K[..., 0, :] /= float(W)
    K[..., 1, :] /= float(H)
    return K


def build_value_dict_from_batch(
    batch: Dict[str, Any],
    camera_scale: Optional[float] = 2.0,
    latent_downsample_factor: int = 8,
    device: Optional[torch.device | str] = None,
    reference_c2ws_key: Optional[str] = None,
) -> TensorDict:
    """
    Build a batched value_dict from a clip batch.

    Expected batch keys from clip_dataset.py:
        imgs: [B, T, 3, H, W] or [T, 3, H, W]
        Ks: [B, T, 3, 3] or [T, 3, 3]
        c2ws: [B, T, 4, 4] / [B, T, 3, 4] or unbatched equivalent
        input_mask: [B, T] or [T]

    Optional:
        reference_c2ws_key: if provided and present in batch, use that tensor as
        the centering reference instead of the current clip's c2ws.
    """
    batch, squeeze_batch = _maybe_unsqueeze_batch(batch)
    device_obj = _resolve_device(batch, device)
    batch = _move_batch_tensors(batch, device_obj)

    imgs = _require_tensor(batch, "imgs")
    Ks = _require_tensor(batch, "Ks")
    c2ws = _require_tensor(batch, "c2ws")
    input_mask = _require_tensor(batch, "input_mask").bool()

    if imgs.dim() != 5:
        raise ValueError(f"imgs must have shape [B, T, 3, H, W], got {tuple(imgs.shape)}")
    if Ks.dim() != 4:
        raise ValueError(f"Ks must have shape [B, T, 3, 3], got {tuple(Ks.shape)}")
    if c2ws.dim() != 4:
        raise ValueError(
            f"c2ws must have shape [B, T, 3, 4] or [B, T, 4, 4], got {tuple(c2ws.shape)}"
        )
    if input_mask.dim() != 2:
        raise ValueError(f"input_mask must have shape [B, T], got {tuple(input_mask.shape)}")

    B, T = imgs.shape[:2]
    if Ks.shape[:2] != (B, T) or c2ws.shape[:2] != (B, T) or input_mask.shape[:2] != (B, T):
        raise ValueError("imgs, Ks, c2ws, and input_mask must agree on [B, T].")

    reference_c2ws = None
    if reference_c2ws_key is not None and reference_c2ws_key in batch:
        reference_c2ws = batch[reference_c2ws_key]
        if not isinstance(reference_c2ws, torch.Tensor):
            raise TypeError(
                f"batch[{reference_c2ws_key!r}] must be a torch.Tensor, got {type(reference_c2ws)}"
            )
        reference_c2ws = reference_c2ws.to(device_obj)
        if reference_c2ws.dim() == 3:
            reference_c2ws = reference_c2ws.unsqueeze(0)
        if reference_c2ws.shape[0] not in {1, B}:
            raise ValueError(
                f"reference_c2ws batch dimension must be 1 or B={B}, got {reference_c2ws.shape[0]}"
            )
        if reference_c2ws.shape[0] == 1 and B > 1:
            reference_c2ws = reference_c2ws.expand(B, *reference_c2ws.shape[1:])

    resolved_camera_scale = _choose_camera_scale(batch, camera_scale)

    value_dicts = []
    for b in range(B):
        ref_c2ws_b = None if reference_c2ws is None else reference_c2ws[b]
        value_dicts.append(
            _build_value_dict_single(
                imgs=imgs[b],
                Ks=Ks[b],
                c2ws=c2ws[b],
                input_mask=input_mask[b],
                camera_scale=resolved_camera_scale,
                latent_downsample_factor=latent_downsample_factor,
                reference_c2ws=ref_c2ws_b,
            )
        )

    batched: TensorDict = {}
    for key in value_dicts[0].keys():
        batched[key] = torch.stack([vd[key] for vd in value_dicts], dim=0)

    return _maybe_squeeze_value_dict(batched, squeeze_batch)


def _encode_images_with_ae(
    ae: nn.Module,
    x: torch.Tensor,
    encoding_t: int = 1,
) -> torch.Tensor:
    """
    Encode images to latent space while handling slight encode() signature
    variations.
    """
    try:
        latents = ae.encode(x, encoding_t) # type: ignore
    except TypeError:
        latents = ae.encode(x) # type: ignore

    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"ae.encode(...) must return a torch.Tensor, got {type(latents)}")
    if latents.dim() != 4:
        raise ValueError(f"ae.encode(...) must return [N, C, h, w], got {tuple(latents.shape)}")
    return latents


def _build_crossattn(
    cond_frames: torch.Tensor,
    cond_frames_mask: torch.Tensor,
    conditioner: nn.Module,
    no_grad: bool,
) -> torch.Tensor:
    B, T = cond_frames.shape[:2]
    outputs = []

    grad_context = torch.no_grad if no_grad else nullcontext
    with grad_context():
        for b in range(B):
            masked_imgs = cond_frames[b, cond_frames_mask[b]]
            if masked_imgs.numel() == 0:
                raise ValueError(f"Sample {b} has no input frames selected.")
            embed = conditioner(masked_imgs).mean(0)  # [D]
            outputs.append(embed.view(1, 1, -1).repeat(T, 1, 1))
    return torch.stack(outputs, dim=0)  # [B, T, 1, D]


def _build_replace(
    cond_frames: torch.Tensor,
    cond_frames_mask: torch.Tensor,
    ae: nn.Module,
    encoding_t: int,
    no_grad: bool,
) -> torch.Tensor:
    B, T = cond_frames.shape[:2]
    outputs = []

    grad_context = torch.no_grad if no_grad else nullcontext
    with grad_context():
        for b in range(B):
            masked_imgs = cond_frames[b, cond_frames_mask[b]]
            latents = _encode_images_with_ae(ae, masked_imgs, encoding_t=encoding_t)
            # Match the released eval path exactly: append a constant channel.
            latents = F.pad(latents, (0, 0, 0, 0, 0, 1), value=1.0)
            replace = latents.new_zeros((T, *latents.shape[1:]))
            replace[cond_frames_mask[b]] = latents
            outputs.append(replace)
    return torch.stack(outputs, dim=0)  # [B, T, C+1, h, w]


def build_conditioning_from_value_dict(
    value_dict: TensorDict,
    conditioner: nn.Module,
    ae: Optional[nn.Module] = None,
    encoding_t: int = 1,
    return_unconditional: bool = True,
    conditioner_no_grad: bool = True,
    encoder_no_grad: bool = True,
) -> tuple[TensorDict, Optional[TensorDict]]:
    """
    Build SEVA conditioning dicts from a value_dict.

    Conditional dict keys:
        crossattn:   [B, T, 1, D]
        replace:     [B, T, C+1, h, w] if ae is provided
        concat:      [B, T, 7, h, w]   (1 input-mask + 6 plucker)
        dense_vector:[B, T, 6, h, w]

    Unconditional dict mirrors eval.py:
        - zero crossattn
        - zero replace (if present)
        - zero input-mask channel in concat
        - same plucker dense conditioning
    """
    cond_frames = value_dict["cond_frames"]
    cond_frames_mask = value_dict["cond_frames_mask"].bool()
    pluckers = value_dict["plucker_coordinate"]

    squeeze_batch = cond_frames.dim() == 4
    if squeeze_batch:
        cond_frames = cond_frames.unsqueeze(0)
        cond_frames_mask = cond_frames_mask.unsqueeze(0)
        pluckers = pluckers.unsqueeze(0)

    if cond_frames.dim() != 5:
        raise ValueError(
            f"value_dict['cond_frames'] must have shape [B, T, 3, H, W], got {tuple(cond_frames.shape)}"
        )
    if pluckers.dim() != 5:
        raise ValueError(
            f"value_dict['plucker_coordinate'] must have shape [B, T, 6, h, w], got {tuple(pluckers.shape)}"
        )

    B, T = cond_frames.shape[:2]
    h, w = pluckers.shape[-2:]

    c_crossattn = _build_crossattn(
        cond_frames=cond_frames,
        cond_frames_mask=cond_frames_mask,
        conditioner=conditioner,
        no_grad=conditioner_no_grad,
    )

    c_concat = torch.cat(
        [
            cond_frames_mask.view(B, T, 1, 1, 1).expand(B, T, 1, h, w).to(pluckers.dtype),
            pluckers,
        ],
        dim=2,
    )

    c_dense_vector = pluckers
    c: TensorDict = {
        "crossattn": c_crossattn,
        "concat": c_concat,
        "dense_vector": c_dense_vector,
    }

    if ae is not None:
        c["replace"] = _build_replace(
            cond_frames=cond_frames,
            cond_frames_mask=cond_frames_mask,
            ae=ae,
            encoding_t=encoding_t,
            no_grad=encoder_no_grad,
        )

    uc: Optional[TensorDict] = None
    if return_unconditional:
        uc = {
            "crossattn": torch.zeros_like(c_crossattn),
            "concat": torch.cat(
                [pluckers.new_zeros(B, T, 1, h, w), pluckers],
                dim=2,
            ),
            "dense_vector": c_dense_vector,
        }
        if "replace" in c:
            uc["replace"] = torch.zeros_like(c["replace"])

    c = _maybe_squeeze_conditioning_dict(c, squeeze_batch)
    uc = None if uc is None else _maybe_squeeze_conditioning_dict(uc, squeeze_batch)
    return c, uc


def build_seva_conditioning(
    batch: Dict[str, Any],
    conditioner: nn.Module,
    ae: Optional[nn.Module] = None,
    encoding_t: int = 1,
    camera_scale: Optional[float] = 2.0,
    latent_downsample_factor: int = 8,
    device: Optional[torch.device | str] = None,
    reference_c2ws_key: Optional[str] = None,
    return_unconditional: bool = True,
    conditioner_no_grad: bool = True,
    encoder_no_grad: bool = True,
) -> ConditioningOutput:
    """
    One-stop helper for turning a clip batch into training-time SEVA
    conditioning.

    Typical usage:
        output = build_seva_conditioning(batch, conditioner, ae)
        c = flatten_conditioning_for_model(output.c)
        x = latents.flatten(0, 1)
        pred = model(x, sigma.flatten(0, 1), c, num_frames=T)
    """
    batch_batched, squeeze_batch = _maybe_unsqueeze_batch(batch)
    value_dict = build_value_dict_from_batch(
        batch=batch_batched,
        camera_scale=camera_scale,
        latent_downsample_factor=latent_downsample_factor,
        device=device,
        reference_c2ws_key=reference_c2ws_key,
    )
    c, uc = build_conditioning_from_value_dict(
        value_dict=value_dict,
        conditioner=conditioner,
        ae=ae,
        encoding_t=encoding_t,
        return_unconditional=return_unconditional,
        conditioner_no_grad=conditioner_no_grad,
        encoder_no_grad=encoder_no_grad,
    )

    cond_frames = value_dict["cond_frames"]
    T = cond_frames.shape[0] if squeeze_batch else cond_frames.shape[1]

    if squeeze_batch:
        value_dict_out = _maybe_squeeze_value_dict(value_dict, True)
    else:
        value_dict_out = value_dict

    return ConditioningOutput(
        c=c,
        uc=uc,
        value_dict=value_dict_out,
        additional_model_inputs={
            "num_frames": T,
            "input_frame_mask": value_dict_out["cond_frames_mask"],
        },
    )


def flatten_time_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Flatten [B, T, ...] -> [B*T, ...]. Leave non-[B, T, ...] tensors untouched.
    """
    if x.dim() < 2:
        return x
    return x.flatten(0, 1)


def flatten_conditioning_for_model(c: TensorDict) -> TensorDict:
    """
    Flatten a batched conditioning dict from [B, T, ...] to [B*T, ...].

    This is usually what the released Seva/SGMWrapper expects at training time,
    because x is typically flattened the same way before the forward pass.
    """
    out: TensorDict = {}
    for key, value in c.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Conditioning entry {key!r} must be a torch.Tensor")
        out[key] = flatten_time_batch(value) if value.dim() >= 2 else value
    return out


class SevaConditioningBuilder(nn.Module):
    """
    Thin nn.Module wrapper around the functional helpers above.

    This is useful when you want a reusable object in train.py / validate.py that
    holds the frozen CLIP image conditioner and optional VAE encoder.
    """

    def __init__(
        self,
        conditioner: Optional[nn.Module] = None,
        ae: Optional[nn.Module] = None,
        camera_scale: float = 2.0,
        latent_downsample_factor: int = 8,
        encoding_t: int = 1,
        return_unconditional: bool = True,
        conditioner_no_grad: bool = True,
        encoder_no_grad: bool = True,
    ) -> None:
        super().__init__()
        self.conditioner = conditioner if conditioner is not None else CLIPConditioner()
        self.ae = ae
        self.camera_scale = float(camera_scale)
        self.latent_downsample_factor = int(latent_downsample_factor)
        self.encoding_t = int(encoding_t)
        self.return_unconditional = bool(return_unconditional)
        self.conditioner_no_grad = bool(conditioner_no_grad)
        self.encoder_no_grad = bool(encoder_no_grad)

    def forward(
        self,
        batch: Dict[str, Any],
        device: Optional[torch.device | str] = None,
        reference_c2ws_key: Optional[str] = None,
    ) -> ConditioningOutput:
        return build_seva_conditioning(
            batch=batch,
            conditioner=self.conditioner,
            ae=self.ae,
            encoding_t=self.encoding_t,
            camera_scale=self.camera_scale,
            latent_downsample_factor=self.latent_downsample_factor,
            device=device,
            reference_c2ws_key=reference_c2ws_key,
            return_unconditional=self.return_unconditional,
            conditioner_no_grad=self.conditioner_no_grad,
            encoder_no_grad=self.encoder_no_grad,
        )


if __name__ == "__main__":
    # Minimal shape smoke test using fake data only.
    B, T, H, W = 2, 8, 576, 576
    batch = {
        "imgs": torch.randn(B, T, 3, H, W),
        "Ks": torch.eye(3).view(1, 1, 3, 3).repeat(B, T, 1, 1),
        "c2ws": torch.eye(4).view(1, 1, 4, 4).repeat(B, T, 1, 1),
        "input_mask": torch.tensor(
            [[True, False, False, True, False, False, False, False]] * B,
            dtype=torch.bool,
        ),
    }

    value_dict = build_value_dict_from_batch(batch)
    print("value_dict keys:", sorted(value_dict.keys()))
    print("cond_frames:", tuple(value_dict["cond_frames"].shape))
    print("plucker_coordinate:", tuple(value_dict["plucker_coordinate"].shape))
