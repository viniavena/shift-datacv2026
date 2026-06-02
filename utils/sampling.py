"""Frame-sampling utilities: grouping, UFS, RFS, and per-method wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import random
import re

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    frame_index: int
    video_id: str


class ImagePathDataset(Dataset):
    def __init__(self, image_paths) -> None:
        self.image_paths = list(image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        return Image.open(self.image_paths[idx]).convert("RGB")


# ---------------------------------------------------------------------------
# Video grouping
# ---------------------------------------------------------------------------

def group_frames_by_video(
    image_paths,
    patterns: List[str],
    fallback_to_parent: bool = True,
) -> Dict[str, List[FrameInfo]]:
    """Group image paths into per-video lists, sorted by frame index."""
    grouped: Dict[str, List[FrameInfo]] = {}
    for path in image_paths:
        video_id, frame_index = extract_video_and_index(path, patterns, fallback_to_parent)
        grouped.setdefault(video_id, []).append(
            FrameInfo(path=path, frame_index=frame_index, video_id=video_id)
        )
    for frames in grouped.values():
        frames.sort(key=lambda f: (f.frame_index, f.path.name))
    return grouped


def extract_video_and_index(
    path: Path,
    patterns: List[str],
    fallback_to_parent: bool,
) -> tuple[str, int]:
    stem = path.stem
    for pattern in patterns:
        match = re.match(pattern, stem)
        if not match:
            continue
        groups = match.groupdict()
        if "video" in groups and "frame" in groups:
            return groups["video"], int(groups["frame"])
        if match.lastindex and match.lastindex >= 2:
            return match.group(1), int(match.group(2))

    video_id = path.parent.name if fallback_to_parent else stem
    return video_id, _extract_last_int(stem)


def _extract_last_int(text: str) -> int:
    matches = re.findall(r"\d+", text)
    return int(matches[-1]) if matches else 0


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------

def sample_uniform(frames: List[FrameInfo], stride: int) -> List[Path]:
    """UFS: Uniform Frame Sampling — select every `stride`-th frame."""
    if stride <= 1:
        return [frame.path for frame in frames]
    return [frame.path for frame in frames[::stride]]


def sample_random(frames: List[FrameInfo], k: int, seed: int) -> List[Path]:
    """RFS: Random Frame Sampling — draw k frames without replacement."""
    if k >= len(frames):
        return [frame.path for frame in frames]
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(frames)), k))
    return [frames[idx].path for idx in indices]


# ---------------------------------------------------------------------------
# AFS wrappers
# ---------------------------------------------------------------------------

def sample_afs_ssvd(
    frames: List[FrameInfo],
    k: int,
    cache_dir: Optional[Path] = None,
) -> List[Path]:
    """AFS-SSVD: Adaptive Frame Sampling with SSIM-based FVI."""
    from methods.baselines.afs import sample_afs

    return sample_afs(frames=frames, k=k, fvi_method="ssvd", cache_dir=cache_dir)


def sample_afs_ofvd(
    frames: List[FrameInfo],
    k: int,
    cache_dir: Optional[Path] = None,
) -> List[Path]:
    """AFS-OFVD: Adaptive Frame Sampling with optical-flow FVI."""
    from methods.baselines.afs import sample_afs

    return sample_afs(frames=frames, k=k, fvi_method="ofvd", cache_dir=cache_dir)


def sample_afs_fsd(
    frames: List[FrameInfo],
    k: int,
    cache_dir: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> List[Path]:
    """AFS-FSD: Adaptive Frame Sampling with feature-similarity FVI."""
    from methods.baselines.afs import sample_afs

    return sample_afs(
        frames=frames, k=k, fvi_method="fsd", cache_dir=cache_dir,
        device=device, batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# CSOD wrapper
# ---------------------------------------------------------------------------

def sample_csod(
    frames: List[FrameInfo],
    k: int,
    cache_dir: Optional[Path] = None,
    device: str = "cuda",
    batch_size: int = 32,
    lambda_param: float = 1.0,
    n_classes: int = 6,
) -> List[Path]:
    """CSOD: Coreset Selection for Object Detection (Lee et al., CVPR 2024)."""
    from methods.baselines.csod import sample_csod as _sample_csod

    return _sample_csod(
        frames=frames, k=k, device=device, batch_size=batch_size,
        lambda_param=lambda_param, n_classes=n_classes, cache_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# SHIFT wrapper
# ---------------------------------------------------------------------------

def sample_shift(
    frames: List[FrameInfo],
    k: int,
    cache_dir: Optional[Path] = None,
    fvi_method: str = "ssvd",
    overselect_factor: float = 3.0,
    embedding_model: str = "resnet50",
    device: str = "cuda",
    batch_size: int = 32,
    pca_dim: int = 128,
    kernel_epsilon: float = 1e-4,
    label_aware: bool = False,
    label_entropy_weight: float = 0.5,
    ablation_mode: str = "full",
) -> tuple[List[Path], Dict[str, object]]:
    """SHIFT: Selecting High-Information Frames for Training (Avena et al., CVPR 2026)."""
    from methods.shift.shift import sample_shift as _sample_shift

    return _sample_shift(
        frames=frames,
        k=k,
        fvi_method=fvi_method,
        overselect_factor=overselect_factor,
        embedding_model=embedding_model,
        embedding_device=device,
        embedding_batch_size=batch_size,
        pca_dim=pca_dim,
        kernel_epsilon=kernel_epsilon,
        label_aware=label_aware,
        label_entropy_weight=label_entropy_weight,
        ablation_mode=ablation_mode,
        cache_dir=cache_dir,
    )
