
"""
train/training/epipolar_loss.py

Visibility-Gated Epipolar Distribution Loss (VG-EDL) for SEVA.

Design goals
------------
1. Keep the SEVA architecture unchanged.
2. Use only data you already have in training:
     - imgs
     - Ks
     - c2ws
     - input_mask
     - diffusion loss output (pred, noisy_latents, sigma)
3. Avoid requiring depth or a learned matcher.
4. Be robust to occlusion / ambiguous texture:
     - teacher is a *distribution* along the source epipolar line
     - low-confidence or no-overlap pixels are skipped
5. Support arbitrary 3D camera rotation because the geometry is built from the
   full relative rotation matrices, not Euler angles.

High-level idea
---------------
For a target frame t and an input/source frame s:

  1. Reconstruct the predicted clean latent x0_hat from the current diffusion
     prediction and decode it to a target RGB image.
  2. For a set of target pixels p_t, compute the epipolar line l_s in the
     source image induced by the known cameras.
  3. Sample K points along that line.
  4. Compare:
       - GT target feature at p_t  -> teacher distribution over the K source points
       - Pred target feature at p_t -> student distribution over the K source points
  5. Minimize KL(teacher || student), but only where the teacher is confident.

Why this is useful
------------------
This does *not* force direct RGB reconstruction. Instead it says:

    "When a target pixel has a confident, visible match somewhere along the
     correct source epipolar line, the generated target should prefer the same
     source-line locations that the real target prefers."

That makes it more geometry-specific than RGB or LPIPS and more robust than a
hard point correspondence.

Practical note
--------------
This file is written to match the current training code discussed in this
project:
  - compute_seva_diffusion_loss(...) returns pred, noisy_latents, sigma
  - batch contains imgs, Ks, c2ws, input_mask
  - the SEVA autoencoder exposes decode(z, chunk_size=None)

The loss is intentionally conservative and light:
  - one target frame per clip by default
  - one source frame per target by default
  - low-resolution descriptors (128x128 by default)
  - cheap RGB+Sobel features
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import math

import torch
import torch.nn.functional as F
from torch import nn


PairMode = Literal["epipolar", "rotation_homography", "skip"]
PredictionType = Literal["epsilon", "x0"]


# -----------------------------------------------------------------------------
# Configuration and outputs
# -----------------------------------------------------------------------------

@dataclass
class EpipolarLossConfig:
    """
    Configuration for the visibility-gated epipolar loss.

    The defaults are chosen to be safe for a first experiment rather than
    maximally strong.

    loss_weight:
        Final scalar multiplier applied to the geometric loss.
    start_step / warmup_steps / every:
        Training schedule controls. Start the loss after diffusion has settled a
        bit, then ramp in gradually.
    prediction_type:
        How to reconstruct x0_hat from the diffusion model output.
        For the current SEVA/DDPM training path this should be "epsilon".
    target_frames_per_clip:
        How many target frames to supervise per batch item. Start with 1 to keep
        memory small.
    sources_per_target:
        How many input views to pair with each target frame. Start with 1.
    feature_res:
        Descriptor resolution. Smaller is cheaper; 128 is a good first value.
    pixels_per_pair:
        Number of target pixels sampled for one source-target pair.
    line_samples:
        Number of sampled points along each epipolar line.
    textured_fraction:
        Fraction of sampled target pixels drawn from high-gradient regions. The
        remaining fraction is sampled uniformly.
    tau:
        Softmax temperature used to form teacher/student line distributions.
    confidence_min:
        Minimum confidence below which a pixel is ignored entirely.
    match_logit_center / match_logit_scale:
        Convert teacher best-similarity into a soft confidence gate.
    max_sigma / sigma_softness:
        Sigma-based gate. The geometry loss is down-weighted on very noisy steps.
    min_epipolar_baseline:
        Below this translation baseline, epipolar geometry becomes degenerate.
        If use_rotation_h_fallback=True, we switch to a rotation-homography loss.
    use_rotation_h_fallback:
        If True, use a simple rotation-only homography feature loss when the
        translation baseline is tiny.
    min_rotation_for_h_deg:
        Minimum relative rotation required before using the homography fallback.
    min_valid_ratio:
        Minimum fraction of sampled pixels that must produce a valid epipolar
        segment (or valid homography projection) or the pair is skipped.
    feature_mode:
        Descriptor type. "rgb_sobel" is intentionally cheap and stable.
    ae_decode_chunk_size:
        Chunk size passed to AE.decode(). Keep this small.
    auto_move_ae_to_device:
        If the training loss previously offloaded the frozen AE to CPU, move it
        back to the target device before decoding.
    """

    loss_weight: float = 1e-3
    start_step: int = 1000
    warmup_steps: int = 2000
    every: int = 1

    prediction_type: PredictionType = "epsilon"

    target_frames_per_clip: int = 1
    sources_per_target: int = 1

    feature_res: int = 128
    pixels_per_pair: int = 512
    line_samples: int = 32
    textured_fraction: float = 0.8

    tau: float = 0.07
    confidence_min: float = 0.10
    match_logit_center: float = 0.30
    match_logit_scale: float = 0.10

    max_sigma: float = 1.0
    sigma_softness: float = 0.25

    min_epipolar_baseline: float = 1e-4
    use_rotation_h_fallback: bool = True
    min_rotation_for_h_deg: float = 2.0
    min_valid_ratio: float = 0.15

    feature_mode: str = "rgb_sobel"
    ae_decode_chunk_size: Optional[int] = 1
    auto_move_ae_to_device: bool = True


@dataclass
class EpipolarLossOutput:
    """
    Output container for logging/debugging.

    loss:
        Final weighted loss that should be added to the diffusion loss.
    raw_loss:
        Unweighted mean pair loss before loss_weight, warmup, and sigma gating.
    warmup_factor:
        Step-based warmup multiplier in [0, 1].
    mean_sigma_gate:
        Mean sigma gate applied to used target frames.
    mean_confidence:
        Mean confidence across used pixels.
    mean_valid_ratio:
        Mean fraction of sampled pixels that produced valid geometry.
    mean_baseline:
        Mean source-target translation baseline.
    mean_rotation_deg:
        Mean full relative rotation angle in degrees.
    num_pairs:
        Number of source-target pairs that contributed.
    num_target_frames:
        Number of target frames that contributed.
    num_pixels:
        Number of pixels used after confidence filtering.
    num_epipolar_pairs:
        Number of pairs using the epipolar branch.
    num_homography_pairs:
        Number of pairs using the rotation-homography fallback.
    num_skipped_pairs:
        Pairs skipped due to low overlap / invalid geometry / no confident pixels.
    pair_modes:
        String list for debug. Usually not logged every step, but helpful.
    """

    loss: torch.Tensor
    raw_loss: torch.Tensor
    warmup_factor: float
    mean_sigma_gate: torch.Tensor
    mean_confidence: torch.Tensor
    mean_valid_ratio: torch.Tensor
    mean_baseline: torch.Tensor
    mean_rotation_deg: torch.Tensor

    num_pairs: int
    num_target_frames: int
    num_pixels: int
    num_epipolar_pairs: int
    num_homography_pairs: int
    num_skipped_pairs: int
    pair_modes: List[str] = field(default_factory=list)


@dataclass
class PairLossStats:
    """
    Internal per-pair statistics.

    loss:
        Mean pair loss before outer weighting.
    mean_confidence:
        Mean confidence over valid pixels.
    valid_ratio:
        Fraction of sampled pixels that produced valid geometry.
    num_pixels_used:
        Number of pixels with confidence > 0.
    mode:
        "epipolar" or "rotation_homography".
    baseline:
        Translation baseline between source and target cameras.
    rotation_deg:
        Full relative rotation angle between source and target cameras.
    """

    loss: torch.Tensor
    mean_confidence: torch.Tensor
    valid_ratio: torch.Tensor
    num_pixels_used: int
    mode: PairMode
    baseline: torch.Tensor
    rotation_deg: torch.Tensor


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------

def _as_batched(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either a single clip sample or a collated batch.

    Expected batched shapes:
        imgs:       [B, T, 3, H, W]
        Ks:         [B, T, 3, 3]
        c2ws:       [B, T, 3, 4] or [B, T, 4, 4]
        input_mask: [B, T]
    """
    imgs = batch["imgs"]
    if imgs.dim() == 5:
        return batch
    if imgs.dim() != 4:
        raise ValueError(f"batch['imgs'] must have 4 or 5 dims, got {tuple(imgs.shape)}")

    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key in {"imgs", "Ks", "c2ws", "input_mask", "frame_ids"}:
            out[key] = value.unsqueeze(0)
        elif isinstance(value, torch.Tensor) and key in {"scene_scale", "num_input_views"}:
            out[key] = value.reshape(1)
        else:
            out[key] = value
    return out


def _to_homogeneous_pose(c2w: torch.Tensor) -> torch.Tensor:
    """
    Convert [3,4] or [4,4] pose to [4,4].
    """
    if c2w.shape == (4, 4):
        return c2w
    if c2w.shape == (3, 4):
        bottom = c2w.new_tensor([[0.0, 0.0, 0.0, 1.0]])
        return torch.cat([c2w, bottom], dim=0)
    raise ValueError(f"Expected pose [3,4] or [4,4], got {tuple(c2w.shape)}")


def _skew(v: torch.Tensor) -> torch.Tensor:
    """
    Return the skew-symmetric matrix [v]_x used in epipolar geometry.
    """
    return torch.stack(
        [
            torch.stack([v.new_tensor(0.0), -v[2], v[1]]),
            torch.stack([v[2], v.new_tensor(0.0), -v[0]]),
            torch.stack([-v[1], v[0], v.new_tensor(0.0)]),
        ],
        dim=0,
    )


def _pixel_to_grid(points_xy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Convert pixel coordinates to grid_sample coordinates in [-1, 1].

    points_xy:
        [..., 2] with x in [0, W-1], y in [0, H-1]
    """
    if W <= 1 or H <= 1:
        raise ValueError("Feature map must have both H and W > 1 for grid sampling.")
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    xg = 2.0 * x / float(W - 1) - 1.0
    yg = 2.0 * y / float(H - 1) - 1.0
    return torch.stack([xg, yg], dim=-1)


def _sample_feature_points(feat: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample a single feature map at N pixel coordinates.

    feat:
        [1, C, H, W]
    points_xy:
        [N, 2] pixel coordinates

    Returns:
        [N, C]
    """
    if feat.dim() != 4 or feat.shape[0] != 1:
        raise ValueError(f"feat must have shape [1,C,H,W], got {tuple(feat.shape)}")
    H, W = feat.shape[-2:]
    grid = _pixel_to_grid(points_xy, H, W).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1, C, N, 1]
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()  # [N, C]


def _sample_feature_line_points(feat: torch.Tensor, line_points_xy: torch.Tensor) -> torch.Tensor:
    """
    Bilinearly sample a single feature map at N x K pixel coordinates.

    feat:
        [1, C, H, W]
    line_points_xy:
        [N, K, 2]

    Returns:
        [N, K, C]
    """
    if feat.dim() != 4 or feat.shape[0] != 1:
        raise ValueError(f"feat must have shape [1,C,H,W], got {tuple(feat.shape)}")
    N, K = line_points_xy.shape[:2]
    H, W = feat.shape[-2:]
    grid = _pixel_to_grid(line_points_xy, H, W).view(1, N, K, 2)
    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # [1, C, N, K]
    return sampled[0].permute(1, 2, 0).contiguous()  # [N, K, C]


def _rgb01_to_gray(img01: torch.Tensor) -> torch.Tensor:
    """
    Convert [B,3,H,W] RGB image in [0,1] to [B,1,H,W] grayscale.
    """
    if img01.dim() != 4 or img01.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(img01.shape)}")
    r, g, b = img01[:, 0:1], img01[:, 1:2], img01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _sobel(gray: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Sobel x/y responses for [B,1,H,W] grayscale input.
    """
    if gray.dim() != 4 or gray.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(gray.shape)}")

    kx = gray.new_tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]]]
    ).unsqueeze(1)  # [1,1,3,3]

    ky = gray.new_tensor(
        [[[-1.0, -2.0, -1.0],
          [ 0.0,  0.0,  0.0],
          [ 1.0,  2.0,  1.0]]]
    ).unsqueeze(1)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy


def build_rgb_sobel_descriptor(img01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Cheap local descriptor used for the first epipolar experiments.

    Descriptor channels:
        3 x RGB
        1 x Sobel-x on grayscale
        1 x Sobel-y on grayscale

    Then L2-normalize across channels.

    Why this descriptor?
        - very cheap
        - fully differentiable
        - enough to test the geometry idea
        - avoids bringing in DINO / VGG memory at the first stage
    """
    gray = _rgb01_to_gray(img01)
    gx, gy = _sobel(gray)
    feat = torch.cat([img01, gx, gy], dim=1)
    feat = F.normalize(feat, dim=1, eps=eps)
    return feat


def _resize_img01(img01: torch.Tensor, size: int) -> torch.Tensor:
    """
    Resize [1,3,H,W] image in [0,1] to [1,3,size,size].
    """
    return F.interpolate(img01, size=(size, size), mode="bilinear", align_corners=False)


def _ensure_img01(img: torch.Tensor) -> torch.Tensor:
    """
    Convert image to [0,1] range and keep shape [1,3,H,W].
    """
    if img.dim() != 4 or img.shape[1] != 3:
        raise ValueError(f"Expected [1,3,H,W], got {tuple(img.shape)}")
    # Most training images here are already in [-1,1].
    img01 = img * 0.5 + 0.5
    return img01.clamp(0.0, 1.0)


def _looks_normalized_intrinsics(K: torch.Tensor) -> bool:
    """
    Heuristic:
      normalized intrinsics often have fx, fy, cx, cy on the order of 0..2
      pixel intrinsics are much larger.
    """
    return (
        float(K[0, 0].abs()) <= 10.0
        and float(K[1, 1].abs()) <= 10.0
        and float(K[0, 2].abs()) <= 2.0
        and float(K[1, 2].abs()) <= 2.0
    )


def K_to_feature_pixels(
    K: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    Convert either pixel-space or normalized intrinsics to the feature-map's
    pixel coordinates.

    This helper is necessary because the batch may contain:
      - raw pixel intrinsics
      - normalized intrinsics
    and the epipolar geometry is computed at feature resolution.
    """
    Kf = K.to(dtype=torch.float32).clone()

    if _looks_normalized_intrinsics(Kf):
        Kf[0, :] *= float(feat_w)
        Kf[1, :] *= float(feat_h)
    else:
        Kf[0, :] *= float(feat_w) / float(image_w)
        Kf[1, :] *= float(feat_h) / float(image_h)

    Kf[2, :] = Kf.new_tensor([0.0, 0.0, 1.0])
    return Kf


def rotation_angle_degrees(R_rel: torch.Tensor) -> torch.Tensor:
    """
    Full SO(3) rotation angle from a relative rotation matrix.

    This uses the matrix trace, so it naturally handles arbitrary 3D rotations,
    including combined yaw/pitch/roll and wrap-around through 360 degrees.
    """
    cos_theta = (torch.trace(R_rel) - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def pose_pair_stats(c2w_s: torch.Tensor, c2w_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Relative pose diagnostics for one source-target pair.

    Returns:
        baseline:
            ||C_t - C_s||
        rotation_deg:
            full relative rotation angle
        view_angle_deg:
            angle between camera forward directions
        up_angle_deg:
            angle between camera up directions
        R_rel:
            full relative rotation matrix
    """
    c2w_s = _to_homogeneous_pose(c2w_s).float()
    c2w_t = _to_homogeneous_pose(c2w_t).float()

    R_s = c2w_s[:3, :3]
    R_t = c2w_t[:3, :3]
    C_s = c2w_s[:3, 3]
    C_t = c2w_t[:3, 3]

    R_rel = R_s.transpose(0, 1) @ R_t
    rotation_deg = rotation_angle_degrees(R_rel)

    baseline = torch.linalg.norm(C_t - C_s)

    # OpenGL forward = -Z, up = +Y
    f_s = -R_s[:, 2]
    f_t = -R_t[:, 2]
    f_s = f_s / (torch.linalg.norm(f_s) + 1e-8)
    f_t = f_t / (torch.linalg.norm(f_t) + 1e-8)
    view_cos = torch.clamp(torch.dot(f_s, f_t), -1.0, 1.0)
    view_angle_deg = torch.rad2deg(torch.acos(view_cos))

    u_s = R_s[:, 1]
    u_t = R_t[:, 1]
    u_s = u_s / (torch.linalg.norm(u_s) + 1e-8)
    u_t = u_t / (torch.linalg.norm(u_t) + 1e-8)
    up_cos = torch.clamp(torch.dot(u_s, u_t), -1.0, 1.0)
    up_angle_deg = torch.rad2deg(torch.acos(up_cos))

    return {
        "baseline": baseline,
        "rotation_deg": rotation_deg,
        "view_angle_deg": view_angle_deg,
        "up_angle_deg": up_angle_deg,
        "R_rel": R_rel,
    }


def _warmup_factor(global_step: Optional[int], config: EpipolarLossConfig) -> float:
    """
    Step-based warmup multiplier in [0, 1].
    """
    if global_step is None:
        return 1.0
    if global_step < config.start_step:
        return 0.0
    if config.warmup_steps <= 0:
        return 1.0
    return min(1.0, float(global_step - config.start_step + 1) / float(config.warmup_steps))


def should_apply_epipolar_loss(global_step: Optional[int], config: EpipolarLossConfig) -> bool:
    """
    Convenience gate for train.py.

    Example:
        if should_apply_epipolar_loss(global_step, epi_cfg):
            epi_out = compute_visibility_gated_epipolar_loss(...)
            loss = loss + epi_out.loss
    """
    if config.loss_weight <= 0.0:
        return False
    if global_step is None:
        return True
    if global_step < config.start_step:
        return False
    if config.every > 1 and (global_step % config.every != 0):
        return False
    return True


# -----------------------------------------------------------------------------
# Geometry: fundamental matrix and rotation homography
# -----------------------------------------------------------------------------

def fundamental_matrix_from_cameras(
    K_t: torch.Tensor,
    K_s: torch.Tensor,
    c2w_t: torch.Tensor,
    c2w_s: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    Compute F_{s<-t} such that a target pixel p_t induces a source epipolar line:

        l_s = F_{s<-t} p_t

    The full relative rotation matrix is used, so this handles arbitrary 3D
    rotations and camera attitudes.
    """
    c2w_t = _to_homogeneous_pose(c2w_t).to(dtype=torch.float32)
    c2w_s = _to_homogeneous_pose(c2w_s).to(dtype=torch.float32)

    K_t_f = K_to_feature_pixels(K_t, image_h=image_h, image_w=image_w, feat_h=feat_h, feat_w=feat_w)
    K_s_f = K_to_feature_pixels(K_s, image_h=image_h, image_w=image_w, feat_h=feat_h, feat_w=feat_w)

    R_t = c2w_t[:3, :3]
    R_s = c2w_s[:3, :3]
    C_t = c2w_t[:3, 3]
    C_s = c2w_s[:3, 3]

    # Source camera receives the target point:
    # x_s = R_st x_t + t_st
    R_st = R_s.transpose(0, 1) @ R_t
    t_st = R_s.transpose(0, 1) @ (C_t - C_s)

    E = _skew(t_st) @ R_st
    Fm = torch.linalg.inv(K_s_f).transpose(0, 1) @ E @ torch.linalg.inv(K_t_f)
    return Fm


def rotation_homography_from_cameras(
    K_t: torch.Tensor,
    K_s: torch.Tensor,
    c2w_t: torch.Tensor,
    c2w_s: torch.Tensor,
    *,
    image_h: int,
    image_w: int,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    Rotation-only homography fallback for near-pure-rotation pairs.

    When the translation baseline is very small, epipolar geometry becomes
    degenerate. For those cases a rotation homography gives a more useful
    signal than forcing an epipolar line.
    """
    c2w_t = _to_homogeneous_pose(c2w_t).to(dtype=torch.float32)
    c2w_s = _to_homogeneous_pose(c2w_s).to(dtype=torch.float32)

    K_t_f = K_to_feature_pixels(K_t, image_h=image_h, image_w=image_w, feat_h=feat_h, feat_w=feat_w)
    K_s_f = K_to_feature_pixels(K_s, image_h=image_h, image_w=image_w, feat_h=feat_h, feat_w=feat_w)

    R_t = c2w_t[:3, :3]
    R_s = c2w_s[:3, :3]

    # q_s ~ K_s * R_s^T * R_t * K_t^{-1} * p_t
    H = K_s_f @ R_s.transpose(0, 1) @ R_t @ torch.linalg.inv(K_t_f)
    return H


# -----------------------------------------------------------------------------
# Pixel and line sampling
# -----------------------------------------------------------------------------

def _sample_textured_pixels(
    gt_img01: torch.Tensor,
    *,
    num_pixels: int,
    textured_fraction: float,
) -> torch.Tensor:
    """
    Sample target pixels with a bias toward textured regions.

    Why:
        Epipolar matching on completely flat pixels is ambiguous and noisy.
    """
    _, _, H, W = gt_img01.shape
    gray = _rgb01_to_gray(gt_img01)
    gx, gy = _sobel(gray)
    grad_mag = torch.sqrt(gx.square() + gy.square()).squeeze(0).squeeze(0)  # [H,W]

    N_textured = int(round(num_pixels * textured_fraction))
    N_uniform = max(0, num_pixels - N_textured)

    device = gt_img01.device
    flat = grad_mag.flatten()
    flat = flat + 1e-6
    flat = flat / flat.sum()

    points = []

    if N_textured > 0:
        idx = torch.multinomial(flat, N_textured, replacement=True)
        y = idx // W
        x = idx % W
        points.append(torch.stack([x.float(), y.float()], dim=-1))

    if N_uniform > 0:
        x = torch.randint(0, W, (N_uniform,), device=device).float()
        y = torch.randint(0, H, (N_uniform,), device=device).float()
        points.append(torch.stack([x, y], dim=-1))

    if not points:
        return gt_img01.new_zeros((0, 2))

    return torch.cat(points, dim=0)


def _unique_points(points: List[Tuple[float, float]], tol: float = 1e-4) -> List[Tuple[float, float]]:
    """
    Small helper to deduplicate line-box intersection points.
    """
    uniq: List[Tuple[float, float]] = []
    for x, y in points:
        keep = True
        for ux, uy in uniq:
            if abs(x - ux) < tol and abs(y - uy) < tol:
                keep = False
                break
        if keep:
            uniq.append((x, y))
    return uniq


def _segment_from_line_box_intersection(
    a: float,
    b: float,
    c: float,
    *,
    W: int,
    H: int,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Intersect a 2D line ax + by + c = 0 with the image box [0,W-1] x [0,H-1].

    Returns:
        Two endpoints of the valid image segment, or None if the line does not
        cross the image.
    """
    pts: List[Tuple[float, float]] = []

    x_left = 0.0
    x_right = float(W - 1)
    y_top = 0.0
    y_bottom = float(H - 1)

    eps = 1e-8

    # Intersect with x = 0 and x = W-1
    if abs(b) > eps:
        y = -(a * x_left + c) / b
        if y_top <= y <= y_bottom:
            pts.append((x_left, y))
        y = -(a * x_right + c) / b
        if y_top <= y <= y_bottom:
            pts.append((x_right, y))

    # Intersect with y = 0 and y = H-1
    if abs(a) > eps:
        x = -(b * y_top + c) / a
        if x_left <= x <= x_right:
            pts.append((x, y_top))
        x = -(b * y_bottom + c) / a
        if x_left <= x <= x_right:
            pts.append((x, y_bottom))

    pts = _unique_points(pts)
    if len(pts) < 2:
        return None

    # Choose the two farthest intersection points in case the line hits a corner.
    best = None
    best_d2 = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if d2 > best_d2:
                best_d2 = d2
                best = (pts[i], pts[j])

    return best


def _sample_points_on_segment(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    K: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Uniformly sample K points on a 2D segment.
    """
    t = torch.linspace(0.0, 1.0, K, device=device, dtype=dtype)
    p0t = torch.tensor(p0, device=device, dtype=dtype)
    p1t = torch.tensor(p1, device=device, dtype=dtype)
    return (1.0 - t)[:, None] * p0t[None, :] + t[:, None] * p1t[None, :]


def sample_epipolar_line_points(
    F_s_t: torch.Tensor,
    target_pixels_xy: torch.Tensor,
    *,
    H: int,
    W: int,
    line_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each target pixel p_t, compute the source epipolar line l_s = F p_t and
    sample points along the valid line segment inside the source image.

    Returns:
        line_points_xy: [N, K, 2]
        valid:          [N] bool
    """
    N = int(target_pixels_xy.shape[0])
    device = target_pixels_xy.device
    dtype = target_pixels_xy.dtype

    out = torch.zeros((N, line_samples, 2), device=device, dtype=dtype)
    valid = torch.zeros((N,), device=device, dtype=torch.bool)

    ones = torch.ones((N, 1), device=device, dtype=dtype)
    pts_h = torch.cat([target_pixels_xy, ones], dim=-1)  # [N,3]
    lines = (F_s_t @ pts_h.t()).t()  # [N,3]

    for i in range(N):
        a = float(lines[i, 0].item())
        b = float(lines[i, 1].item())
        c = float(lines[i, 2].item())
        seg = _segment_from_line_box_intersection(a, b, c, W=W, H=H)
        if seg is None:
            continue
        p0, p1 = seg
        out[i] = _sample_points_on_segment(p0, p1, line_samples, device=device, dtype=dtype)
        valid[i] = True

    return out, valid


# -----------------------------------------------------------------------------
# Source/target frame selection
# -----------------------------------------------------------------------------

def _rank_source_ids_for_target(
    source_ids: torch.Tensor,
    target_id: int,
    c2ws_bt: torch.Tensor,
) -> torch.Tensor:
    """
    Rank candidate source frames for one target.

    Heuristic:
        prefer small translation baseline and similar viewing direction.
    """
    c2w_t = _to_homogeneous_pose(c2ws_bt[target_id]).float()
    R_t = c2w_t[:3, :3]
    C_t = c2w_t[:3, 3]
    f_t = -R_t[:, 2]
    f_t = f_t / (torch.linalg.norm(f_t) + 1e-8)

    scores = []
    for sid in source_ids.tolist():
        c2w_s = _to_homogeneous_pose(c2ws_bt[sid]).float()
        R_s = c2w_s[:3, :3]
        C_s = c2w_s[:3, 3]
        f_s = -R_s[:, 2]
        f_s = f_s / (torch.linalg.norm(f_s) + 1e-8)

        baseline = torch.linalg.norm(C_t - C_s)
        view_cost = 0.5 * (1.0 - torch.clamp(torch.dot(f_s, f_t), -1.0, 1.0))
        score = baseline + view_cost
        scores.append(score)

    scores_t = torch.stack(scores)
    order = torch.argsort(scores_t)
    return source_ids[order]


def _choose_target_ids(
    target_ids: torch.Tensor,
    sigma_bt: torch.Tensor,
    *,
    max_targets: int,
) -> torch.Tensor:
    """
    Choose target frames for epipolar supervision.

    Strategy:
        choose the lowest-sigma target frames first because x0_hat is more
        reliable there.
    """
    target_sigmas = sigma_bt[target_ids]
    order = torch.argsort(target_sigmas)
    return target_ids[order[:max_targets]]


# -----------------------------------------------------------------------------
# Predicted x0 reconstruction and AE decode
# -----------------------------------------------------------------------------

def reconstruct_x0_pred_latents(
    *,
    loss_out: Any,
    batch: Dict[str, Any],
    prediction_type: PredictionType = "epsilon",
) -> torch.Tensor:
    """
    Reconstruct predicted clean latents x0_hat from the current diffusion output.

    The current training path uses epsilon prediction, and DiffusionLossOutput
    already provides:
        - pred           : [B*T, C, h, w]
        - noisy_latents  : [B, T, C, h, w]
        - sigma          : [B, T]

    So we can reconstruct x0_hat without changing diffusion_loss.py.

    Returns:
        x0_pred: [B, T, C, h, w]
    """
    batch = _as_batched(batch)
    imgs = batch["imgs"]
    B, T = imgs.shape[:2]

    pred_flat = loss_out.pred
    pred_bt = pred_flat.view(B, T, *pred_flat.shape[1:])

    if prediction_type == "epsilon":
        sigma_e = loss_out.sigma[..., None, None, None].to(
            device=loss_out.noisy_latents.device,
            dtype=loss_out.noisy_latents.dtype,
        )
        return loss_out.noisy_latents - sigma_e * pred_bt

    if prediction_type == "x0":
        return pred_bt

    raise ValueError(f"Unsupported prediction_type: {prediction_type!r}")


def decode_latents_to_img01(
    ae: nn.Module,
    latents: torch.Tensor,
    *,
    chunk_size: Optional[int],
    auto_move_to_device: bool = True,
) -> torch.Tensor:
    """
    Decode SEVA/SD latents to RGB in [0,1].

    Important:
        We do NOT wrap decode in torch.no_grad() because the AE parameters may be
        frozen while gradients still need to flow *through* the decoder back to
        the predicted latents and then into the SEVA backbone.
    """
    device = latents.device
    if auto_move_to_device:
        try:
            ae_device = next(ae.parameters()).device
            if ae_device != device:
                ae.to(device)
        except StopIteration:
            pass

    try:
        decoded = ae.decode(latents, chunk_size)
    except TypeError:
        decoded = ae.decode(latents)

    if not isinstance(decoded, torch.Tensor):
        raise TypeError(f"ae.decode(...) must return Tensor, got {type(decoded)}")

    return _ensure_img01(decoded)


# -----------------------------------------------------------------------------
# Pair losses
# -----------------------------------------------------------------------------

def _teacher_student_confidence(
    sim_gt: torch.Tensor,
    p_gt: torch.Tensor,
    *,
    line_samples: int,
    config: EpipolarLossConfig,
) -> torch.Tensor:
    """
    Convert the teacher distribution into a confidence score.

    We combine two intuitions:
      1. Low entropy => the teacher strongly prefers a particular source-line region.
      2. High best-similarity => the match is visually plausible at all.

    If the teacher is flat or low-similarity, the pixel is likely occluded,
    textureless, or not visible in the source image and should not be trusted.
    """
    entropy = -(p_gt * p_gt.clamp_min(1e-8).log()).sum(dim=-1)
    entropy_conf = 1.0 - entropy / math.log(float(line_samples))

    best_sim = sim_gt.max(dim=-1).values
    match_conf = torch.sigmoid((best_sim - config.match_logit_center) / config.match_logit_scale)

    conf = entropy_conf * match_conf
    conf = torch.where(conf >= config.confidence_min, conf, torch.zeros_like(conf))
    return conf


def epipolar_distribution_pair_loss(
    *,
    pred_img01: torch.Tensor,
    gt_img01: torch.Tensor,
    src_img01: torch.Tensor,
    K_t: torch.Tensor,
    K_s: torch.Tensor,
    c2w_t: torch.Tensor,
    c2w_s: torch.Tensor,
    config: EpipolarLossConfig,
) -> Optional[PairLossStats]:
    """
    Main epipolar branch.

    Steps:
      1. Resize source/GT/pred images to feature_res x feature_res.
      2. Build cheap descriptors.
      3. Sample target pixels, biased toward textured regions.
      4. Compute source epipolar lines.
      5. Teacher:
           GT target feature vs source-line features -> p_gt
         Student:
           Pred target feature vs source-line features -> p_pred
      6. Confidence-gated KL(p_gt || p_pred)
    """
    feat_res = int(config.feature_res)

    pred_small = _resize_img01(pred_img01, feat_res)
    gt_small = _resize_img01(gt_img01, feat_res)
    src_small = _resize_img01(src_img01, feat_res)

    if config.feature_mode != "rgb_sobel":
        raise ValueError(f"Unsupported feature_mode: {config.feature_mode!r}")

    feat_pred = build_rgb_sobel_descriptor(pred_small)
    with torch.no_grad():
        feat_gt = build_rgb_sobel_descriptor(gt_small).detach()
        feat_src = build_rgb_sobel_descriptor(src_small).detach()

    Hf, Wf = feat_res, feat_res
    image_h = int(gt_img01.shape[-2])
    image_w = int(gt_img01.shape[-1])

    stats = pose_pair_stats(c2w_s, c2w_t)
    baseline = stats["baseline"]
    rotation_deg = stats["rotation_deg"]

    F_s_t = fundamental_matrix_from_cameras(
        K_t=K_t,
        K_s=K_s,
        c2w_t=c2w_t,
        c2w_s=c2w_s,
        image_h=image_h,
        image_w=image_w,
        feat_h=Hf,
        feat_w=Wf,
    )

    target_pixels = _sample_textured_pixels(
        gt_small,
        num_pixels=int(config.pixels_per_pair),
        textured_fraction=float(config.textured_fraction),
    )
    if target_pixels.numel() == 0:
        return None

    line_points, valid = sample_epipolar_line_points(
        F_s_t,
        target_pixels,
        H=Hf,
        W=Wf,
        line_samples=int(config.line_samples),
    )

    valid_ratio = valid.float().mean()
    if float(valid_ratio.item()) < float(config.min_valid_ratio):
        return None

    valid_pixels = target_pixels[valid]
    valid_lines = line_points[valid]
    if valid_pixels.numel() == 0:
        return None

    feat_gt_pts = _sample_feature_points(feat_gt, valid_pixels)         # [N,C]
    feat_pred_pts = _sample_feature_points(feat_pred, valid_pixels)     # [N,C]
    feat_src_line = _sample_feature_line_points(feat_src, valid_lines)  # [N,K,C]

    # Cosine because descriptors are L2-normalized.
    sim_gt = (feat_gt_pts[:, None, :] * feat_src_line).sum(dim=-1)      # [N,K]
    sim_pred = (feat_pred_pts[:, None, :] * feat_src_line).sum(dim=-1)  # [N,K]

    p_gt = F.softmax(sim_gt / config.tau, dim=-1).detach()
    log_p_pred = F.log_softmax(sim_pred / config.tau, dim=-1)

    conf = _teacher_student_confidence(
        sim_gt=sim_gt.detach(),
        p_gt=p_gt,
        line_samples=int(config.line_samples),
        config=config,
    )
    num_used = int((conf > 0).sum().item())
    if num_used == 0:
        return None

    # KL(p_gt || p_pred) up to an additive constant:
    #   sum p_gt * (log p_gt - log p_pred)
    # We can ignore the teacher entropy term for optimization and use
    #   cross_entropy = -sum p_gt * log p_pred
    per_pixel = -(p_gt * log_p_pred).sum(dim=-1)
    loss = (conf * per_pixel).sum() / conf.sum().clamp_min(1e-6)

    return PairLossStats(
        loss=loss,
        mean_confidence=conf.mean(),
        valid_ratio=valid_ratio,
        num_pixels_used=num_used,
        mode="epipolar",
        baseline=baseline.detach(),
        rotation_deg=rotation_deg.detach(),
    )


def rotation_homography_pair_loss(
    *,
    pred_img01: torch.Tensor,
    gt_img01: torch.Tensor,
    src_img01: torch.Tensor,
    K_t: torch.Tensor,
    K_s: torch.Tensor,
    c2w_t: torch.Tensor,
    c2w_s: torch.Tensor,
    config: EpipolarLossConfig,
) -> Optional[PairLossStats]:
    """
    Rotation-homography fallback for near-zero baseline / pure-rotation cases.

    Why this exists:
        Epipolar geometry degenerates when translation is near zero. But a
        rotation-only mapping still gives a useful correspondence prior.

    Loss:
        sample target pixels
        map them by H_{s<-t}
        confidence from GT-target vs mapped-source similarity
        loss on predicted-target vs mapped-source similarity
    """
    feat_res = int(config.feature_res)

    pred_small = _resize_img01(pred_img01, feat_res)
    gt_small = _resize_img01(gt_img01, feat_res)
    src_small = _resize_img01(src_img01, feat_res)

    feat_pred = build_rgb_sobel_descriptor(pred_small)
    with torch.no_grad():
        feat_gt = build_rgb_sobel_descriptor(gt_small).detach()
        feat_src = build_rgb_sobel_descriptor(src_small).detach()

    Hf, Wf = feat_res, feat_res
    image_h = int(gt_img01.shape[-2])
    image_w = int(gt_img01.shape[-1])

    stats = pose_pair_stats(c2w_s, c2w_t)
    baseline = stats["baseline"]
    rotation_deg = stats["rotation_deg"]

    H_s_t = rotation_homography_from_cameras(
        K_t=K_t,
        K_s=K_s,
        c2w_t=c2w_t,
        c2w_s=c2w_s,
        image_h=image_h,
        image_w=image_w,
        feat_h=Hf,
        feat_w=Wf,
    )

    target_pixels = _sample_textured_pixels(
        gt_small,
        num_pixels=int(config.pixels_per_pair),
        textured_fraction=float(config.textured_fraction),
    )
    if target_pixels.numel() == 0:
        return None

    ones = torch.ones((target_pixels.shape[0], 1), device=target_pixels.device, dtype=target_pixels.dtype)
    pts_h = torch.cat([target_pixels, ones], dim=-1)                    # [N,3]
    mapped_h = (H_s_t @ pts_h.t()).t()                                  # [N,3]
    denom = mapped_h[:, 2:3].abs().clamp_min(1e-6)
    mapped_xy = mapped_h[:, :2] / denom

    valid = (
        (mapped_xy[:, 0] >= 0.0)
        & (mapped_xy[:, 0] <= float(Wf - 1))
        & (mapped_xy[:, 1] >= 0.0)
        & (mapped_xy[:, 1] <= float(Hf - 1))
    )
    valid_ratio = valid.float().mean()
    if float(valid_ratio.item()) < float(config.min_valid_ratio):
        return None

    tgt_valid = target_pixels[valid]
    src_valid = mapped_xy[valid]
    if tgt_valid.numel() == 0:
        return None

    feat_gt_pts = _sample_feature_points(feat_gt, tgt_valid)      # [N,C]
    feat_pred_pts = _sample_feature_points(feat_pred, tgt_valid)  # [N,C]
    feat_src_pts = _sample_feature_points(feat_src, src_valid)    # [N,C]

    sim_gt = (feat_gt_pts * feat_src_pts).sum(dim=-1)             # [N]
    sim_pred = (feat_pred_pts * feat_src_pts).sum(dim=-1)         # [N]

    conf = torch.sigmoid((sim_gt.detach() - config.match_logit_center) / config.match_logit_scale)
    conf = torch.where(conf >= config.confidence_min, conf, torch.zeros_like(conf))

    num_used = int((conf > 0).sum().item())
    if num_used == 0:
        return None

    # 1 - cosine similarity because descriptors are normalized.
    per_pixel = 1.0 - sim_pred
    loss = (conf * per_pixel).sum() / conf.sum().clamp_min(1e-6)

    return PairLossStats(
        loss=loss,
        mean_confidence=conf.mean(),
        valid_ratio=valid_ratio,
        num_pixels_used=num_used,
        mode="rotation_homography",
        baseline=baseline.detach(),
        rotation_deg=rotation_deg.detach(),
    )


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def compute_visibility_gated_epipolar_loss(
    *,
    bundle: Any,
    batch: Dict[str, Any],
    loss_out: Any,
    config: EpipolarLossConfig,
    global_step: Optional[int] = None,
) -> EpipolarLossOutput:
    """
    Compute the visibility-gated epipolar loss for the current training batch.

    Inputs
    ------
    bundle:
        The current SEVA bundle. We only need bundle.ae for latent decoding.
    batch:
        Training batch containing at least imgs, Ks, c2ws, input_mask.
    loss_out:
        Output of compute_seva_diffusion_loss(...). We use:
            pred
            noisy_latents
            sigma
        to reconstruct x0_hat.
    config:
        EpipolarLossConfig.
    global_step:
        Used for start-step / warmup / every scheduling.

    Returns
    -------
    EpipolarLossOutput

    Notes
    -----
    - This function is intentionally separate from diffusion_loss.py so you can
      switch it on/off without rewriting the core diffusion loss.
    - The function only decodes selected target frames, not the whole clip.
    - If no valid geometric pairs are found, the returned loss is exactly zero.
    """
    batch = _as_batched(batch)

    if bundle.ae is None:
        raise ValueError("Epipolar loss requires bundle.ae so it can decode predicted target latents.")

    warm = _warmup_factor(global_step, config)
    if warm <= 0.0:
        zero = loss_out.loss * 0.0
        return EpipolarLossOutput(
            loss=zero,
            raw_loss=zero,
            warmup_factor=0.0,
            mean_sigma_gate=zero.detach(),
            mean_confidence=zero.detach(),
            mean_valid_ratio=zero.detach(),
            mean_baseline=zero.detach(),
            mean_rotation_deg=zero.detach(),
            num_pairs=0,
            num_target_frames=0,
            num_pixels=0,
            num_epipolar_pairs=0,
            num_homography_pairs=0,
            num_skipped_pairs=0,
            pair_modes=[],
        )

    imgs = batch["imgs"].to(loss_out.noisy_latents.device)
    Ks = batch["Ks"].to(loss_out.noisy_latents.device)
    c2ws = batch["c2ws"].to(loss_out.noisy_latents.device)
    input_mask = batch["input_mask"].bool().to(loss_out.noisy_latents.device)

    B, T = imgs.shape[:2]
    x0_pred = reconstruct_x0_pred_latents(
        loss_out=loss_out,
        batch=batch,
        prediction_type=config.prediction_type,
    )

    pair_losses: List[torch.Tensor] = []
    sigma_gates: List[torch.Tensor] = []
    confidences: List[torch.Tensor] = []
    valid_ratios: List[torch.Tensor] = []
    baselines: List[torch.Tensor] = []
    rotations: List[torch.Tensor] = []
    pair_modes: List[str] = []

    num_pairs = 0
    num_targets = 0
    num_pixels = 0
    num_epipolar = 0
    num_h = 0
    num_skipped = 0

    for b in range(B):
        source_ids = torch.where(input_mask[b])[0]
        target_ids = torch.where(~input_mask[b])[0]

        if source_ids.numel() == 0 or target_ids.numel() == 0:
            continue

        chosen_targets = _choose_target_ids(
            target_ids=target_ids,
            sigma_bt=loss_out.sigma[b],
            max_targets=int(config.target_frames_per_clip),
        )

        for tgt_id_t in chosen_targets.tolist():
            num_targets += 1

            sigma_t = loss_out.sigma[b, tgt_id_t].float()
            sigma_gate = torch.sigmoid((config.max_sigma - sigma_t) / config.sigma_softness)

            # Skip near-zero sigma gate. The diffusion objective already covers
            # all timesteps; we only want geometry when x0_hat is somewhat
            # reliable.
            if float(sigma_gate.item()) < 1e-3:
                num_skipped += 1
                continue

            ranked_sources = _rank_source_ids_for_target(
                source_ids=source_ids,
                target_id=tgt_id_t,
                c2ws_bt=c2ws[b],
            )
            chosen_sources = ranked_sources[: int(config.sources_per_target)]

            # Decode only the selected target frame.
            pred_img01 = decode_latents_to_img01(
                bundle.ae,
                x0_pred[b, tgt_id_t : tgt_id_t + 1],
                chunk_size=config.ae_decode_chunk_size,
                auto_move_to_device=config.auto_move_ae_to_device,
            )

            gt_img01 = _ensure_img01(imgs[b, tgt_id_t : tgt_id_t + 1])

            for src_id_t in chosen_sources.tolist():
                src_img01 = _ensure_img01(imgs[b, src_id_t : src_id_t + 1])

                pose_stats = pose_pair_stats(c2ws[b, src_id_t], c2ws[b, tgt_id_t])
                baseline = pose_stats["baseline"]
                rotation_deg = pose_stats["rotation_deg"]

                pair: Optional[PairLossStats] = None

                if float(baseline.item()) >= float(config.min_epipolar_baseline):
                    pair = epipolar_distribution_pair_loss(
                        pred_img01=pred_img01,
                        gt_img01=gt_img01,
                        src_img01=src_img01,
                        K_t=Ks[b, tgt_id_t],
                        K_s=Ks[b, src_id_t],
                        c2w_t=c2ws[b, tgt_id_t],
                        c2w_s=c2ws[b, src_id_t],
                        config=config,
                    )
                elif (
                    config.use_rotation_h_fallback
                    and float(rotation_deg.item()) >= float(config.min_rotation_for_h_deg)
                ):
                    pair = rotation_homography_pair_loss(
                        pred_img01=pred_img01,
                        gt_img01=gt_img01,
                        src_img01=src_img01,
                        K_t=Ks[b, tgt_id_t],
                        K_s=Ks[b, src_id_t],
                        c2w_t=c2ws[b, tgt_id_t],
                        c2w_s=c2ws[b, src_id_t],
                        config=config,
                    )

                if pair is None:
                    num_skipped += 1
                    pair_modes.append("skip")
                    continue

                weighted_pair = pair.loss * sigma_gate
                pair_losses.append(weighted_pair)
                sigma_gates.append(sigma_gate.detach())
                confidences.append(pair.mean_confidence.detach())
                valid_ratios.append(pair.valid_ratio.detach())
                baselines.append(pair.baseline.detach())
                rotations.append(pair.rotation_deg.detach())
                pair_modes.append(pair.mode)

                num_pairs += 1
                num_pixels += int(pair.num_pixels_used)
                if pair.mode == "epipolar":
                    num_epipolar += 1
                elif pair.mode == "rotation_homography":
                    num_h += 1

    if not pair_losses:
        zero = loss_out.loss * 0.0
        return EpipolarLossOutput(
            loss=zero,
            raw_loss=zero,
            warmup_factor=warm,
            mean_sigma_gate=zero.detach(),
            mean_confidence=zero.detach(),
            mean_valid_ratio=zero.detach(),
            mean_baseline=zero.detach(),
            mean_rotation_deg=zero.detach(),
            num_pairs=0,
            num_target_frames=num_targets,
            num_pixels=0,
            num_epipolar_pairs=0,
            num_homography_pairs=0,
            num_skipped_pairs=num_skipped,
            pair_modes=pair_modes,
        )

    raw_loss = torch.stack(pair_losses).mean()
    weighted_loss = raw_loss * float(config.loss_weight) * float(warm)

    return EpipolarLossOutput(
        loss=weighted_loss,
        raw_loss=raw_loss.detach(),
        warmup_factor=float(warm),
        mean_sigma_gate=torch.stack(sigma_gates).mean(),
        mean_confidence=torch.stack(confidences).mean(),
        mean_valid_ratio=torch.stack(valid_ratios).mean(),
        mean_baseline=torch.stack(baselines).mean(),
        mean_rotation_deg=torch.stack(rotations).mean(),
        num_pairs=num_pairs,
        num_target_frames=num_targets,
        num_pixels=num_pixels,
        num_epipolar_pairs=num_epipolar,
        num_homography_pairs=num_h,
        num_skipped_pairs=num_skipped,
        pair_modes=pair_modes,
    )


__all__ = [
    "EpipolarLossConfig",
    "EpipolarLossOutput",
    "PairLossStats",
    "compute_visibility_gated_epipolar_loss",
    "decode_latents_to_img01",
    "epipolar_distribution_pair_loss",
    "fundamental_matrix_from_cameras",
    "pose_pair_stats",
    "reconstruct_x0_pred_latents",
    "rotation_homography_from_cameras",
    "rotation_homography_pair_loss",
    "should_apply_epipolar_loss",
]
