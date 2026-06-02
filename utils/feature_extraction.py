from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np


def extract_frame_embeddings(
    frames: Sequence[object],
    model_name: str = "resnet50",
    device: str = "cuda",
    batch_size: int = 32,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """Extract frame-level embeddings using a pretrained ResNet backbone.

    Supported model_name values: "resnet18", "resnet50".
    """
    image_paths = _coerce_frame_paths(frames)
    if not image_paths:
        return np.zeros((0, 0), dtype=np.float32)

    cache_path = _resolve_cache_path(
        cache_dir=cache_dir, image_paths=image_paths, prefix=f"frameemb_{model_name}"
    )
    if cache_path is not None and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        cached_paths = payload["paths"].tolist()
        if cached_paths == [str(p) for p in image_paths]:
            return payload["embeddings"]

    name = model_name.lower()
    if name in {"resnet50", "resnet"}:
        embeddings = _extract_with_resnet50(image_paths, device=device, batch_size=batch_size)
    elif name in {"resnet18"}:
        embeddings = _extract_with_resnet18(image_paths, device=device, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            embeddings=embeddings.astype(np.float32),
            paths=np.array([str(p) for p in image_paths]),
        )
    return embeddings


def extract_crop_features(
    image_path: Path,
    bboxes: List[Tuple[int, float, float, float, float]],
    model,
    transform,
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """Extract class-wise averaged features from YOLO-format GT crops.

    Args:
        bboxes: list of (class_id, cx, cy, w, h), normalized [0,1].
    """
    if not bboxes:
        return {}

    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    class_to_tensors: Dict[int, List[torch.Tensor]] = {}
    for class_id, cx, cy, w, h in bboxes:
        x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, width=width, height=height)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img.crop((x1, y1, x2, y2))
        class_to_tensors.setdefault(int(class_id), []).append(transform(crop))

    model.eval()
    output: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for class_id, crop_tensors in class_to_tensors.items():
            batch = torch.stack(crop_tensors, dim=0).to(device)
            feats = model(batch)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.flatten(1).detach().cpu().numpy()
            output[class_id] = feats.mean(axis=0).astype(np.float32)
    return output


def read_yolo_bboxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    if not label_path.exists():
        return []
    bboxes: List[Tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        bboxes.append((class_id, cx, cy, w, h))
    return bboxes


def infer_label_path(image_path: Path) -> Path:
    path_str = str(image_path)
    if "/images/" in path_str:
        path_str = path_str.replace("/images/", "/labels/")
    if "\\images\\" in path_str:
        path_str = path_str.replace("\\images\\", "\\labels\\")
    return Path(path_str).with_suffix(".txt")


def build_resnet50_feature_extractor(device: str = "cuda"):
    return _build_resnet_feature_extractor(backbone="resnet50", device=device)


def build_resnet18_feature_extractor(device: str = "cuda"):
    return _build_resnet_feature_extractor(backbone="resnet18", device=device)


def _build_resnet_feature_extractor(backbone: str, device: str = "cuda"):
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    backbone = backbone.lower()
    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    elif backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet backbone: {backbone}")
    extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    return extractor, transform


def _extract_with_resnet50(image_paths: List[Path], device: str, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image

    model, transform = build_resnet50_feature_extractor(device=device)
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(batch_tensors, dim=0).to(device)
            feats = model(batch).flatten(1).cpu().numpy().astype(np.float32)
            embeddings.append(feats)
    return np.concatenate(embeddings, axis=0)


def _extract_with_resnet18(image_paths: List[Path], device: str, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image

    model, transform = build_resnet18_feature_extractor(device=device)
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch_tensors = [transform(Image.open(p).convert("RGB")) for p in batch_paths]
            batch = torch.stack(batch_tensors, dim=0).to(device)
            feats = model(batch).flatten(1).cpu().numpy().astype(np.float32)
            embeddings.append(feats)
    return np.concatenate(embeddings, axis=0)


def _coerce_frame_paths(frames: Sequence[object]) -> List[Path]:
    paths: List[Path] = []
    for item in frames:
        if isinstance(item, Path):
            paths.append(item)
            continue
        path = getattr(item, "path", None)
        if path is None:
            raise TypeError("Frame item must be Path or have `.path` attribute.")
        paths.append(Path(path))
    return paths


def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, *, width: int, height: int
) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, round((cx - w / 2.0) * width))))
    y1 = int(max(0, min(height - 1, round((cy - h / 2.0) * height))))
    x2 = int(max(0, min(width, round((cx + w / 2.0) * width))))
    y2 = int(max(0, min(height, round((cy + h / 2.0) * height))))
    return x1, y1, x2, y2


def _resolve_cache_path(
    cache_dir: Optional[Path], image_paths: List[Path], prefix: str
) -> Optional[Path]:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not image_paths:
        return cache_dir / f"{prefix}_empty.npz"
    split = _infer_split_tag(image_paths[0])
    video = _video_slug(image_paths)
    return cache_dir / f"{prefix}_{split}_{video}_n{len(image_paths)}.npz"


def _infer_split_tag(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    for split in ("train", "val", "test"):
        if split in lowered:
            return split
    return "nosplit"


def _video_slug(image_paths: List[Path]) -> str:
    first = image_paths[0]
    stem = first.stem
    match = re.match(r"^(?P<video>.+?)[-_]frame[_-]?(?P<frame>\d+)$", stem)
    if not match:
        match = re.match(r"^(?P<video>.+?)[-_](?P<frame>\d+)$", stem)
    if match:
        base = match.group("video")
    else:
        base = first.parent.name or stem
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", base).strip("._-")
    return safe or "video"
