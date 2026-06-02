"""Frame Variance Index (FVI) — inter-frame change signals.

Used by:
  - SHIFT Stage 1 (SSVD variant by default)
  - AFS baselines (SSVD, OFVD, FSD variants)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple
import hashlib

import numpy as np

from utils.feature_extraction import extract_frame_embeddings


def compute_fvi(
    frames: Sequence[object],
    method: str,
    cache_dir: Optional[Path] = None,
    **kwargs,
) -> np.ndarray:
    """Dispatch to the requested FVI variant.

    Parameters
    ----------
    frames : sequence of Path or FrameInfo
    method : "ssvd" | "ofvd" | "pixel_diff" | "fsd"
    cache_dir : optional cache directory

    Returns
    -------
    (N,) float32 array — inter-frame change score per frame (0 at index 0).
    """
    method = method.lower()
    if method == "ofvd":
        return compute_fvi_ofvd(frames, cache_dir=cache_dir, **kwargs)
    if method == "ssvd":
        return compute_fvi_ssvd(frames, cache_dir=cache_dir, **kwargs)
    if method in {"pixel_diff", "pixeldiff"}:
        return compute_fvi_pixel_diff(frames, cache_dir=cache_dir, **kwargs)
    if method == "fsd":
        return compute_fvi_fsd(frames, cache_dir=cache_dir, **kwargs)
    raise ValueError(f"Unsupported FVI method: {method!r}. Use one of: ssvd, ofvd, pixel_diff, fsd.")


def compute_fvi_ssvd(
    frames: Sequence[object],
    resize: Tuple[int, int] = (320, 240),
    use_grayscale: bool = True,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """SSIM-based Video Difference: FVI[t] = 1 - SSIM(f_t, f_{t-1})."""
    frame_paths = _coerce_frame_paths(frames)
    cache_path = _cache_path(frame_paths, "ssvd", cache_dir=cache_dir)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)["fvi"]

    n = len(frame_paths)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    deltas = np.zeros((n,), dtype=np.float32)
    prev = _read_gray(frame_paths[0], resize=resize, gray=use_grayscale)
    for idx in range(1, n):
        curr = _read_gray(frame_paths[idx], resize=resize, gray=use_grayscale)
        deltas[idx] = 1.0 - _ssim(prev, curr)
        prev = curr

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fvi=deltas)
    return deltas


def compute_fvi_ofvd(
    frames: Sequence[object],
    resize: Tuple[int, int] = (320, 240),
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Optical-Flow Video Difference: FVI[t] = mean magnitude of Farneback flow."""
    frame_paths = _coerce_frame_paths(frames)
    cache_path = _cache_path(frame_paths, "ofvd", cache_dir=cache_dir)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)["fvi"]

    n = len(frame_paths)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    deltas = np.zeros((n,), dtype=np.float32)
    prev = _read_gray(frame_paths[0], resize=resize)
    for idx in range(1, n):
        curr = _read_gray(frame_paths[idx], resize=resize)
        try:
            import cv2  # type: ignore

            flow = cv2.calcOpticalFlowFarneback(
                prev, curr, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            deltas[idx] = float(np.mean(mag))
        except Exception:
            deltas[idx] = float(np.mean(np.abs(curr.astype(np.float32) - prev.astype(np.float32))))
        prev = curr

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fvi=deltas)
    return deltas


def compute_fvi_pixel_diff(
    frames: Sequence[object],
    resize: Tuple[int, int] = (320, 240),
    use_grayscale: bool = True,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Pixel-difference FVI: FVI[t] = mean |f_t - f_{t-1}| / 255."""
    frame_paths = _coerce_frame_paths(frames)
    cache_path = _cache_path(frame_paths, "pixel_diff", cache_dir=cache_dir)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)["fvi"]

    n = len(frame_paths)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    deltas = np.zeros((n,), dtype=np.float32)
    prev = _read_gray(frame_paths[0], resize=resize, gray=use_grayscale).astype(np.float32)
    for idx in range(1, n):
        curr = _read_gray(frame_paths[idx], resize=resize, gray=use_grayscale).astype(np.float32)
        deltas[idx] = float(np.mean(np.abs(curr - prev)) / 255.0)
        prev = curr

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fvi=deltas)
    return deltas


def compute_fvi_fsd(
    frames: Sequence[object],
    device: str = "cuda",
    batch_size: int = 32,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Feature-Similarity Difference: FVI[t] = 1 - cosine_sim(emb_t, emb_{t-1})."""
    frame_paths = _coerce_frame_paths(frames)
    cache_path = _cache_path(frame_paths, "fsd", cache_dir=cache_dir)
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)["fvi"]

    n = len(frame_paths)
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)

    emb = extract_frame_embeddings(
        frame_paths,
        model_name="resnet50",
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir / "emb_resnet50" if cache_dir is not None else None,
    )
    emb = emb.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1) + 1e-8
    emb = emb / norms[:, None]
    deltas = np.zeros((n,), dtype=np.float32)
    for idx in range(1, n):
        sim = float(np.dot(emb[idx - 1], emb[idx]))
        deltas[idx] = max(0.0, 1.0 - sim)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fvi=deltas)
    return deltas


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_frame_paths(frames: Sequence[object]) -> list[Path]:
    paths: list[Path] = []
    for item in frames:
        if isinstance(item, Path):
            paths.append(item)
            continue
        path = getattr(item, "path", None)
        if path is None:
            raise TypeError("Frame item must be Path or have `.path` attribute.")
        paths.append(Path(path))
    return paths


def _read_gray(path: Path, resize: Tuple[int, int], gray: bool = True) -> np.ndarray:
    try:
        import cv2  # type: ignore

        if gray:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            return cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        resized = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    except Exception:
        from PIL import Image

        mode = "L" if gray else "RGB"
        return np.asarray(Image.open(path).convert(mode).resize(resize))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity  # type: ignore

        return float(
            structural_similarity(a, b, data_range=255, channel_axis=-1 if a.ndim == 3 else None)
        )
    except Exception:
        diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0
        return float(max(0.0, 1.0 - diff))


def _cache_path(frame_paths: list[Path], method: str, cache_dir: Optional[Path]) -> Optional[Path]:
    if cache_dir is None or not frame_paths:
        return None
    digest = hashlib.sha1()
    for p in frame_paths:
        stat = p.stat()
        digest.update(str(p).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    key = digest.hexdigest()[:12]
    return cache_dir / f"fvi_{method}_{key}.npz"
