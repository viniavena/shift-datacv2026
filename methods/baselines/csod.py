"""Coreset Selection for Object Detection (CSOD) — Lee et al., CVPR 2024.

Two-stage greedy coreset method:
  Stage 1 : Extract class-wise averaged ResNet-50 crop features per image.
  Stage 2 : Greedy class-aware submodular selection maximising representativeness
             while penalising redundancy with already-selected frames.

Note: CSOD requires ground-truth bounding-box annotations during the selection
phase (training split labels are provided as input).

Reference
---------
H. Lee, S. Kim, J. Lee, J. Yoo, and N. Kwak.
"Coreset Selection for Object Detection." CVPR, 2024.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import hashlib
import re

import numpy as np

from utils.feature_extraction import (
    build_resnet50_feature_extractor,
    extract_crop_features,
    infer_label_path,
    read_yolo_bboxes,
)


def sample_csod(
    frames: Sequence[object],
    k: int,
    *,
    device: str = "cuda",
    batch_size: int = 32,
    lambda_param: float = 1.0,
    n_classes: int = 6,
    cache_dir: Optional[Path] = None,
) -> List[Path]:
    """Select k frames via CSOD coreset selection.

    Parameters
    ----------
    frames : sequence of Path or FrameInfo
        All candidate frames. Corresponding YOLO label files must exist.
    k : int
        Number of frames to select.
    device : str
        Device for ResNet-50 feature extraction.
    lambda_param : float
        Trade-off between representativeness and diversity (lambda in the paper).
    n_classes : int
        Number of object classes in the dataset.
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

    image_paths = [_frame_path(f) for f in frame_list]
    label_paths = [infer_label_path(p) for p in image_paths]
    image_features = extract_imagewise_classwise_features(
        image_paths=image_paths,
        label_paths=label_paths,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    selected_idx = select_csod(
        image_features=image_features,
        n_select=k,
        lambda_param=lambda_param,
        n_classes=n_classes,
    )
    selected_idx = sorted(set(selected_idx))
    return [image_paths[idx] for idx in selected_idx]


def extract_imagewise_classwise_features(
    image_paths: List[Path],
    label_paths: List[Path],
    device: str = "cuda",
    batch_size: int = 32,
    cache_dir: Optional[Path] = None,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Stage 1: class-wise averaged crop features per image."""
    if len(image_paths) != len(label_paths):
        raise ValueError("image_paths and label_paths must have the same length.")

    cache_path = _cache_path(image_paths, label_paths, cache_dir=cache_dir)
    cached = _load_cached_csod(cache_path, image_paths, label_paths)
    if cached is not None:
        return cached

    model, transform = build_resnet50_feature_extractor(device=device)
    image_features: Dict[int, Dict[int, np.ndarray]] = {}
    _ = batch_size  # batch_size kept for API compatibility

    for idx, (img_path, lbl_path) in enumerate(zip(image_paths, label_paths)):
        bboxes = read_yolo_bboxes(lbl_path)
        feats = extract_crop_features(
            image_path=img_path, bboxes=bboxes, model=model,
            transform=transform, device=device,
        )
        image_features[idx] = feats if feats else {}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=np.array([image_features], dtype=object),
            image_paths=np.array([str(p) for p in image_paths]),
            label_paths=np.array([str(p) for p in label_paths]),
            fingerprint=np.array([_inputs_fingerprint(image_paths, label_paths)]),
        )
    return image_features


def select_csod(
    image_features: Dict[int, Dict[int, np.ndarray]],
    n_select: int,
    lambda_param: float = 1.0,
    n_classes: int = 6,
) -> List[int]:
    """Stage 2: greedy class-aware submodular-like selection."""
    if n_select <= 0:
        return []
    all_indices = sorted(image_features.keys())
    if n_select >= len(all_indices):
        return all_indices

    selected: List[int] = []
    selected_set: set = set()

    class_to_pool: Dict[int, List[int]] = {c: [] for c in range(n_classes)}
    for idx, feat_map in image_features.items():
        for class_id in feat_map.keys():
            if 0 <= int(class_id) < n_classes:
                class_to_pool[int(class_id)].append(idx)

    class_cursor = 0
    while len(selected) < n_select:
        class_id = class_cursor % n_classes
        class_cursor += 1

        candidates = [i for i in class_to_pool[class_id] if i not in selected_set]
        if not candidates:
            if class_cursor > (n_classes * 3):
                fallback = [i for i in all_indices if i not in selected_set]
                if not fallback:
                    break
                selected.append(fallback[0])
                selected_set.add(fallback[0])
                class_cursor = 0
            continue

        best_idx = None
        best_score = -1e18
        for idx in candidates:
            score = _score_candidate(
                idx=idx,
                class_id=class_id,
                image_features=image_features,
                selected=selected,
                lambda_param=lambda_param,
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            continue
        selected.append(best_idx)
        selected_set.add(best_idx)
        class_cursor = 0

    return selected[:n_select]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_candidate(
    *,
    idx: int,
    class_id: int,
    image_features: Dict[int, Dict[int, np.ndarray]],
    selected: List[int],
    lambda_param: float,
) -> float:
    feat_i = image_features[idx].get(class_id)
    if feat_i is None:
        return -1e9
    peers = [j for j, fmap in image_features.items() if class_id in fmap and j != idx]
    if not peers:
        return -1e6

    rep_sims = [_cosine(feat_i, image_features[j][class_id]) for j in peers]
    rep_term = float(np.mean(rep_sims)) if rep_sims else 0.0

    sel_peers = [j for j in selected if class_id in image_features.get(j, {})]
    if not sel_peers:
        div_term = 0.0
    else:
        div_sims = [_cosine(feat_i, image_features[j][class_id]) for j in sel_peers]
        div_term = float(np.mean(div_sims))
    return lambda_param * rep_term - div_term


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    return float(np.dot(a, b) / denom)


def _frame_path(frame: object) -> Path:
    if isinstance(frame, Path):
        return frame
    path = getattr(frame, "path", None)
    if path is None:
        raise TypeError("Frame item must be Path or have `.path` attribute.")
    return Path(path)


def _cache_path(
    image_paths: List[Path], label_paths: List[Path], cache_dir: Optional[Path]
) -> Optional[Path]:
    if cache_dir is None:
        return None
    if not image_paths:
        return cache_dir / "csod_feats_empty.npz"
    split = _infer_split_tag(image_paths[0])
    scope = _csod_scope_slug(image_paths)
    return cache_dir / f"csod_feats_{split}_{scope}_n{len(image_paths)}.npz"


def _load_cached_csod(
    cache_path: Optional[Path],
    image_paths: List[Path],
    label_paths: List[Path],
) -> Optional[Dict[int, Dict[int, np.ndarray]]]:
    if cache_path is None or not cache_path.exists():
        return None
    expected_images = [str(p) for p in image_paths]
    expected_labels = [str(p) for p in label_paths]
    expected_fp = _inputs_fingerprint(image_paths, label_paths)
    try:
        with np.load(cache_path, allow_pickle=True) as payload:
            required = {"features", "image_paths", "label_paths", "fingerprint"}
            if not required.issubset(payload.files):
                return None
            if payload["image_paths"].tolist() != expected_images:
                return None
            if payload["label_paths"].tolist() != expected_labels:
                return None
            cached_fp_raw = payload["fingerprint"].tolist()
            cached_fp = str(cached_fp_raw[0]) if isinstance(cached_fp_raw, list) else str(cached_fp_raw)
            if cached_fp != expected_fp:
                return None
            return payload["features"][0]
    except Exception:
        return None


def _inputs_fingerprint(image_paths: List[Path], label_paths: List[Path]) -> str:
    digest = hashlib.sha1()
    for p in image_paths + label_paths:
        if p.exists():
            stat = p.stat()
            digest.update(str(p).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime)).encode("utf-8"))
        else:
            digest.update(str(p).encode("utf-8"))
    return digest.hexdigest()


def _infer_split_tag(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    for split in ("train", "val", "test"):
        if split in lowered:
            return split
    return "nosplit"


def _csod_scope_slug(image_paths: List[Path]) -> str:
    videos = {_extract_video_stem(p) for p in image_paths}
    base = f"video_{next(iter(videos))}" if len(videos) == 1 else "split_collection"
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", base).strip("._-")
    return safe or "scope"


def _extract_video_stem(path: Path) -> str:
    stem = path.stem
    match = re.match(r"^(?P<video>.+?)[-_]frame[_-]?(?P<frame>\d+)$", stem)
    if not match:
        match = re.match(r"^(?P<video>.+?)[-_](?P<frame>\d+)$", stem)
    if match:
        return match.group("video")
    return path.parent.name or stem
