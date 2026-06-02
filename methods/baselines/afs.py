"""Adaptive Frame Sampling (AFS) — Yoon & Choi, CVPR 2023.

Selects the top-k frames by Frame Variance Index (FVI) score.
Three FVI variants are supported:

  - AFS-SSVD : FVI = 1 - SSIM(f_t, f_{t-1})
  - AFS-OFVD : FVI = mean optical-flow magnitude between consecutive frames
  - AFS-FSD  : FVI = 1 - cosine_similarity(emb_t, emb_{t-1})  (ResNet-50)

Reference
---------
J. Yoon and M.-K. Choi. "Exploring Video Frame Redundancies for Efficient
Data Sampling and Annotation in Instance Segmentation." CVPR, 2023.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from utils.fvi import compute_fvi


def sample_afs(
    frames: Sequence[object],
    k: int,
    fvi_method: str,
    cache_dir: Optional[Path] = None,
    **fvi_kwargs,
) -> List[Path]:
    """Select the k frames with the highest FVI scores (top-k AFS).

    Parameters
    ----------
    frames : sequence of Path or FrameInfo
        All frames for one video, in temporal order.
    k : int
        Number of frames to select.
    fvi_method : str
        FVI variant: "ssvd", "ofvd", or "fsd".
    cache_dir : optional cache directory.

    Returns
    -------
    List of selected frame paths, sorted by temporal order.
    """
    frame_list = list(frames)
    if k >= len(frame_list):
        return [_frame_path(f) for f in frame_list]
    if k <= 0:
        return []

    fvi = compute_fvi(frame_list, method=fvi_method, cache_dir=cache_dir, **fvi_kwargs)
    top_indices = np.argsort(fvi)[-k:]
    top_indices = sorted(int(i) for i in top_indices)
    return [_frame_path(frame_list[i]) for i in top_indices]


def _frame_path(frame: object) -> Path:
    if isinstance(frame, Path):
        return frame
    path = getattr(frame, "path", None)
    if path is None:
        raise TypeError("Frame item must be Path or have `.path` attribute.")
    return Path(path)
