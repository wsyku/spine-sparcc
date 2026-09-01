"""Preprocessing used by both final models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

SEGMENT_TO_ID = {"cervical": 0, "thoracic": 1, "lumbar": 2, "C": 0, "T": 1, "L": 2}


def load_raw(path: str | Path) -> np.ndarray:
    xyz = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    return np.transpose(xyz, (2, 1, 0))


def load_probability(path: str | Path, key: str = "probabilities", channel: int = 1) -> np.ndarray:
    path = Path(path)
    if path.name.endswith(".nii.gz"):
        return np.transpose(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32), (2, 1, 0))
    with np.load(path) as archive:
        array = np.asarray(archive[key], dtype=np.float32)
    return array[channel] if array.ndim == 4 else array


def resize_hw(array: np.ndarray, size: Sequence[int], mode: str = "bilinear") -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[:, None]
    kwargs = {"size": tuple(map(int, size)), "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor, **kwargs)[:, 0]


def foreground_zscore(volume: torch.Tensor) -> torch.Tensor:
    foreground = volume > 0
    if not foreground.any():
        return volume
    values = volume[foreground]
    return torch.where(
        foreground,
        (volume - values.mean()) / values.std(unbiased=False).clamp_min(1e-6),
        volume,
    )


def pad_or_center_crop(
    tensor: torch.Tensor, depth: int = 16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    original_depth = int(tensor.shape[0])
    if original_depth > depth:
        start = (original_depth - depth) // 2
        return (
            tensor[start : start + depth],
            torch.ones(depth, dtype=torch.bool),
            torch.arange(start, start + depth),
            start,
        )
    valid = torch.zeros(depth, dtype=torch.bool)
    valid[:original_depth] = True
    indices = torch.full((depth,), -1, dtype=torch.long)
    indices[:original_depth] = torch.arange(original_depth)
    if original_depth < depth:
        tensor = torch.cat(
            [tensor, torch.zeros((depth - original_depth, *tensor.shape[1:]), dtype=tensor.dtype)],
            dim=0,
        )
    return tensor, valid, indices, 0


def prepare_case(
    raw_path: str | Path,
    probability_path: str | Path,
    size: Sequence[int] = (320, 320),
    depth: int = 16,
    probability_key: str = "probabilities",
    foreground_channel: int = 1,
) -> dict[str, torch.Tensor | int]:
    raw_np = load_raw(raw_path)
    probability_np = load_probability(
        probability_path,
        key=probability_key,
        channel=foreground_channel,
    )
    if raw_np.shape != probability_np.shape:
        raise ValueError(f"Raw/probability shape mismatch: {raw_np.shape} vs {probability_np.shape}")
    raw = foreground_zscore(resize_hw(raw_np, size, "bilinear"))
    probability = resize_hw(probability_np, size, "bilinear").clamp(0, 1)
    raw, valid, indices, crop_start = pad_or_center_crop(raw, depth)
    probability, _, _, _ = pad_or_center_crop(probability, depth)
    return {
        "raw_image": raw.unsqueeze(0),
        "probability": probability.unsqueeze(0),
        "slice_mask": valid,
        "original_slice_indices": indices,
        "crop_start": crop_start,
        "original_depth": int(raw_np.shape[0]),
    }
