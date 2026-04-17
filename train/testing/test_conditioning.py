from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.training.conditioning import (
    build_conditioning_from_value_dict,
    build_value_dict_from_batch,
    flatten_conditioning_for_model,
)


class DummyConditioner(nn.Module):
    def __init__(self, dim: int = 1024) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(x.shape[0], self.dim, device=x.device, dtype=x.dtype)


class DummyAE(nn.Module):
    def encode(self, x: torch.Tensor, encoding_t: int = 1) -> torch.Tensor:
        del encoding_t
        return torch.randn(x.shape[0], 4, 72, 72, device=x.device, dtype=x.dtype)


def main() -> None:
    batch_size, num_frames, height, width = 2, 8, 576, 576
    batch = {
        "imgs": torch.randn(batch_size, num_frames, 3, height, width),
        "Ks": torch.eye(3).view(1, 1, 3, 3).repeat(batch_size, num_frames, 1, 1),
        "c2ws": torch.eye(4).view(1, 1, 4, 4).repeat(batch_size, num_frames, 1, 1),
        "input_mask": torch.tensor(
            [
                [True, False, False, True, False, False, False, False],
                [True, True, False, False, False, False, False, False],
            ],
            dtype=torch.bool,
        ),
    }

    conditioner = DummyConditioner(dim=1024)
    ae = DummyAE()

    value_dict = build_value_dict_from_batch(
        batch=batch,
        camera_scale=2.0,
        latent_downsample_factor=8,
    )

    print("value_dict keys:", sorted(value_dict.keys()))
    for key, value in value_dict.items():
        print(key, value.shape, value.dtype)

    assert value_dict["cond_frames"].shape == (batch_size, num_frames, 3, height, width)
    assert value_dict["cond_frames_mask"].shape == (batch_size, num_frames)
    assert value_dict["plucker_coordinate"].shape == (batch_size, num_frames, 6, 72, 72)
    assert value_dict["c2w"].shape == (batch_size, num_frames, 4, 4)
    assert value_dict["K"].shape == (batch_size, num_frames, 3, 3)

    c, uc = build_conditioning_from_value_dict(
        value_dict=value_dict,
        conditioner=conditioner,
        ae=ae,
        return_unconditional=True,
    )

    print("\nconditional keys:", sorted(c.keys()))
    for key, value in c.items():
        print("c", key, value.shape, value.dtype)

    print("\nunconditional keys:", sorted(uc.keys()))
    for key, value in uc.items():
        print("uc", key, value.shape, value.dtype)

    assert c["crossattn"].shape == (batch_size, num_frames, 1, 1024)
    assert c["concat"].shape == (batch_size, num_frames, 7, 72, 72)
    assert c["dense_vector"].shape == (batch_size, num_frames, 6, 72, 72)
    assert c["replace"].shape == (batch_size, num_frames, 5, 72, 72)

    assert uc["crossattn"].shape == (batch_size, num_frames, 1, 1024)
    assert uc["concat"].shape == (batch_size, num_frames, 7, 72, 72)
    assert uc["dense_vector"].shape == (batch_size, num_frames, 6, 72, 72)
    assert uc["replace"].shape == (batch_size, num_frames, 5, 72, 72)

    flat_c = flatten_conditioning_for_model(c)
    print("\nflattened:")
    for key, value in flat_c.items():
        print(key, value.shape)

    assert flat_c["crossattn"].shape == (batch_size * num_frames, 1, 1024)
    assert flat_c["concat"].shape == (batch_size * num_frames, 7, 72, 72)
    assert flat_c["dense_vector"].shape == (batch_size * num_frames, 6, 72, 72)
    assert flat_c["replace"].shape == (batch_size * num_frames, 5, 72, 72)

    print("\nconditioning checks passed")


if __name__ == "__main__":
    main()
