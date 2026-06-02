"""SHIFT — Selecting High-Information Frames for Training.

Two-stage, training-free frame selection for object detector training on driving video.

Stage 1  Variation-Based Temporal Allocation
    Interprets cumulative SSIM variation as an information signal and samples
    frames at uniform intervals in cumulative-variation space, allocating more
    candidates to dynamic segments and fewer to static ones.

Stage 2  Feature-Space Entropy Maximization
    Models candidate embeddings as a multivariate Gaussian and greedily selects
    the subset that maximises differential entropy:

        S* = argmax_{|S|=K}  log det(L_S + lambda * I)

    The log-determinant objective is submodular; the greedy algorithm achieves
    the standard (1 - 1/e) approximation guarantee.

Reference
---------
Avena et al. "Which Frames Matter? Frame Selection for Training Object
Detectors on Driving Videos." CVPR DataCV Workshop, 2026.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1 — Variation-Based Temporal Allocation
# ---------------------------------------------------------------------------

def variation_uniform_sample(
    fvi: np.ndarray,
    k_candidates: int,
) -> np.ndarray:
    """Sample *k_candidates* frame indices at uniform cumulative-variation intervals.

    Parameters
    ----------
    fvi : (N,) array
        Frame Variance Index scores (non-negative).  fvi[0] is typically 0.
    k_candidates : int
        Number of candidate indices to return.

    Returns
    -------
    indices : (k_candidates,) int array, sorted ascending.
    """
    n = len(fvi)
    if k_candidates >= n:
        return np.arange(n, dtype=np.intp)

    fvi_safe = np.maximum(fvi, 0.0).astype(np.float64)

    # Small floor so that truly-zero-change segments still get some probability
    floor = fvi_safe[fvi_safe > 0].min() * 0.01 if np.any(fvi_safe > 0) else 1.0
    fvi_floored = np.maximum(fvi_safe, floor)

    cuminfo = np.cumsum(fvi_floored)
    total = cuminfo[-1]
    if total <= 0:
        return np.round(np.linspace(0, n - 1, k_candidates)).astype(np.intp)

    # Target levels at midpoints of equal-variation bins
    targets = (np.arange(k_candidates) + 0.5) * (total / k_candidates)
    indices = np.searchsorted(cuminfo, targets, side="right")
    indices = np.clip(indices, 0, n - 1)

    # Deduplicate
    indices = np.unique(indices)
    if len(indices) < k_candidates:
        all_idx = set(range(n))
        used = set(indices.tolist())
        remaining = sorted(all_idx - used)
        needed = k_candidates - len(indices)
        extra = np.array(remaining[:needed], dtype=np.intp)
        indices = np.sort(np.concatenate([indices, extra]))

    return indices[:k_candidates]


def uniform_temporal_sample(n_frames: int, k_candidates: int) -> np.ndarray:
    """Uniform-time fallback — Stage 1 ablation variant."""
    if k_candidates <= 0:
        return np.zeros((0,), dtype=np.intp)
    if k_candidates >= n_frames:
        return np.arange(n_frames, dtype=np.intp)

    indices = np.round(np.linspace(0, n_frames - 1, k_candidates)).astype(np.intp)
    indices = np.clip(indices, 0, n_frames - 1)
    indices = np.unique(indices)
    if len(indices) < k_candidates:
        all_idx = np.arange(n_frames, dtype=np.intp)
        used = set(indices.tolist())
        remaining = [i for i in all_idx.tolist() if i not in used]
        needed = k_candidates - len(indices)
        extra = np.array(remaining[:needed], dtype=np.intp)
        indices = np.sort(np.concatenate([indices, extra]))
    return indices[:k_candidates]


# ---------------------------------------------------------------------------
# Stage 2 — Log-Determinant Greedy Selection (Maximum Entropy)
# ---------------------------------------------------------------------------

def logdet_greedy_select(
    embeddings: np.ndarray,
    budget: int,
    *,
    pca_dim: int = 128,
    epsilon: float = 1e-4,
    label_entropy: Optional[np.ndarray] = None,
    label_entropy_weight: float = 0.5,
) -> List[int]:
    """Greedy maximisation of log det(K_S + eps*I) — maximum-entropy subset.

    Parameters
    ----------
    embeddings : (n, d) array
        Candidate frame embeddings (PCA-reduced and L2-normalised internally).
    budget : int
        Number of frames to select (K in the paper).
    pca_dim : int
        Target dimensionality for PCA reduction.  Skipped if d <= pca_dim.
    epsilon : float
        Kernel regularisation for numerical stability.
    label_entropy : (n,) array or None
        Per-frame Shannon entropy of class labels (SHIFT-LA variant only).
    label_entropy_weight : float
        Weight beta for label-entropy diagonal boost (SHIFT-LA only).

    Returns
    -------
    selected : list[int]
        Indices into *embeddings*, in selection order.
    """
    n, d = embeddings.shape
    if budget >= n:
        return list(range(n))
    if budget <= 0:
        return []

    E = embeddings.astype(np.float64)

    # Optional PCA reduction
    if d > pca_dim and n > pca_dim:
        mean = E.mean(axis=0)
        E_centered = E - mean
        U, S, _ = np.linalg.svd(E_centered, full_matrices=False)
        E = U[:, :pca_dim] * S[:pca_dim]

    # L2-normalise → cosine kernel = dot product
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    E = E / norms

    # Kernel matrix
    K = E @ E.T
    np.clip(K, -1.0, 1.0, out=K)

    # Label-aware diagonal boost (SHIFT-LA)
    if label_entropy is not None and label_entropy_weight > 0:
        le = np.asarray(label_entropy, dtype=np.float64)
        if le.max() > 0:
            le_norm = le / le.max()
            K[np.diag_indices_from(K)] += label_entropy_weight * le_norm

    # Greedy Cholesky-based selection
    residual = np.diag(K).copy() + epsilon
    chol_components = np.zeros((budget, n), dtype=np.float64)
    selected: List[int] = []
    selected_set: set = set()

    for t in range(budget):
        scores = residual.copy()
        for s in selected_set:
            scores[s] = -np.inf

        best = int(np.argmax(scores))
        best_var = scores[best]
        if best_var <= 0:
            break

        selected.append(best)
        selected_set.add(best)

        c = K[:, best].copy()
        for s in range(t):
            c -= chol_components[s] * chol_components[s][best]
        c /= np.sqrt(best_var)
        chol_components[t] = c

        residual -= c ** 2
        np.maximum(residual, 0.0, out=residual)

    return selected


# ---------------------------------------------------------------------------
# Label-entropy helper (SHIFT-LA variant)
# ---------------------------------------------------------------------------

def compute_label_entropy(frame_paths: Sequence[Path]) -> np.ndarray:
    """Shannon entropy of the class distribution per frame (from YOLO labels).

    Frames with no annotations receive entropy 0.
    """
    from utils.feature_extraction import infer_label_path, read_yolo_bboxes

    n = len(frame_paths)
    entropies = np.zeros(n, dtype=np.float64)
    for i, img_path in enumerate(frame_paths):
        label_path = infer_label_path(Path(img_path))
        bboxes = read_yolo_bboxes(label_path)
        if not bboxes:
            continue
        class_ids = np.array([b[0] for b in bboxes])
        _, counts = np.unique(class_ids, return_counts=True)
        probs = counts / counts.sum()
        entropies[i] = -np.sum(probs * np.log(np.maximum(probs, 1e-12)))
    return entropies


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sample_shift(
    frames: list,
    k: int,
    *,
    fvi_method: str = "ssvd",
    overselect_factor: float = 3.0,
    embedding_model: str = "resnet50",
    embedding_device: str = "cuda",
    embedding_batch_size: int = 32,
    pca_dim: int = 128,
    kernel_epsilon: float = 1e-4,
    label_aware: bool = False,
    label_entropy_weight: float = 0.5,
    ablation_mode: str = "full",
    cache_dir: Optional[Path] = None,
) -> tuple[List[Path], Dict[str, object]]:
    """SHIFT: Selecting High-Information Frames for Training.

    Parameters
    ----------
    frames : list[FrameInfo]
        All frames for one video, sorted temporally.
    k : int
        Budget — number of frames to select (K in the paper).
    fvi_method : str
        FVI variant for Stage 1: "ssvd" (default), "ofvd", "pixel_diff", or "fsd".
    overselect_factor : float
        Stage 1 produces ceil(overselect_factor * k) candidates (alpha in the paper).
    embedding_model : str
        Backbone for Stage 2 embeddings. Default: "resnet50".
    pca_dim : int
        PCA target dimensionality for the kernel matrix.
    kernel_epsilon : float
        Regularisation added to the kernel diagonal (lambda in the paper).
    label_aware : bool
        If True, boost frames with diverse object types (SHIFT-LA variant).
    label_entropy_weight : float
        Weight for label-entropy boost (SHIFT-LA only).
    ablation_mode : str
        "full"                 — Stage 1 + Stage 2 (default, the method in the paper).
        "stage1_only"          — Stage 1 only (no diversity refinement).
        "stage2_only_uniform"  — Stage 1 replaced by uniform temporal sampling.
    cache_dir : Path or None
        Root directory for FVI and embedding caches.

    Returns
    -------
    (selected_paths, stats) : (list[Path], dict)
    """
    from utils.feature_extraction import extract_frame_embeddings
    from utils.fvi import compute_fvi

    n = len(frames)
    if k <= 0:
        return [], {"input_frames": n, "selected_frames": 0}
    if k >= n:
        return [f.path for f in frames], {
            "input_frames": n,
            "selected_frames": n,
            "shortcut": "k>=n",
        }

    mode = str(ablation_mode).strip().lower()
    valid_modes = {"full", "stage1_only", "stage2_only_uniform"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported ablation_mode '{ablation_mode}'. "
            "Use: full, stage1_only, stage2_only_uniform."
        )

    # ── Stage 1: Variation-Based Temporal Allocation ─────────────────────
    if mode in {"full", "stage1_only"}:
        fvi_cache = cache_dir / f"fvi_{fvi_method}" if cache_dir else None
        fvi_kwargs: Dict[str, object] = {}
        if fvi_method == "fsd":
            fvi_kwargs.update(device=embedding_device, batch_size=embedding_batch_size)

        fvi_scores = compute_fvi(frames, method=fvi_method, cache_dir=fvi_cache, **fvi_kwargs)
        k_candidates = min(n, k) if mode == "stage1_only" else min(n, max(k, int(np.ceil(overselect_factor * k))))
        candidate_indices = variation_uniform_sample(fvi_scores, k_candidates)
        stage1_method = f"variation_uniform_{fvi_method}"
    else:
        k_candidates = min(n, max(k, int(np.ceil(overselect_factor * k))))
        candidate_indices = uniform_temporal_sample(n_frames=n, k_candidates=k_candidates)
        stage1_method = "uniform_temporal"

    candidate_frames = [frames[i] for i in candidate_indices]
    candidate_paths = [frames[i].path for i in candidate_indices]

    if mode == "stage1_only":
        selected = sorted(candidate_frames, key=lambda f: (f.frame_index, f.path.name))
        return [f.path for f in selected], {
            "input_frames": n,
            "candidate_frames": len(candidate_frames),
            "selected_frames": len(selected),
            "ablation_mode": mode,
            "stage1_method": stage1_method,
            "stage2_method": "disabled",
        }

    if len(candidate_frames) <= k:
        selected = sorted(candidate_frames, key=lambda f: (f.frame_index, f.path.name))
        return [f.path for f in selected], {
            "input_frames": n,
            "candidate_frames": len(candidate_frames),
            "selected_frames": len(selected),
            "ablation_mode": mode,
            "stage1_method": stage1_method,
            "shortcut": "candidates<=k",
        }

    # ── Stage 2: Feature-Space Entropy Maximization ──────────────────────
    emb_cache_dir = cache_dir / "embeddings" if cache_dir else None
    embeddings = extract_frame_embeddings(
        frames=candidate_frames,
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
        cache_dir=emb_cache_dir,
    )

    label_ent = compute_label_entropy(candidate_paths) if label_aware else None

    stage2_indices = logdet_greedy_select(
        embeddings=embeddings,
        budget=k,
        pca_dim=pca_dim,
        epsilon=kernel_epsilon,
        label_entropy=label_ent,
        label_entropy_weight=label_entropy_weight if label_aware else 0.0,
    )

    selected_frames = [candidate_frames[i] for i in stage2_indices]
    selected_frames.sort(key=lambda f: (f.frame_index, f.path.name))

    stats: Dict[str, object] = {
        "input_frames": n,
        "candidate_frames": len(candidate_frames),
        "selected_frames": len(selected_frames),
        "ablation_mode": mode,
        "stage1_method": stage1_method,
        "stage1_overselect_factor": overselect_factor,
        "stage2_method": "logdet_greedy",
        "stage2_pca_dim": pca_dim,
        "stage2_epsilon": kernel_epsilon,
        "stage2_embedding_model": embedding_model,
        "label_aware": label_aware,
    }
    if label_aware:
        stats["label_entropy_weight"] = label_entropy_weight

    return [f.path for f in selected_frames], stats
