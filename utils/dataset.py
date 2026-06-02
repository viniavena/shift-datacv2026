from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import os
import shutil

import yaml


@dataclass(frozen=True)
class YoloDatasetSpec:
    data_yaml: Path
    dataset_root: Path
    splits: Dict[str, Path]
    names: Optional[List[str]]
    nc: Optional[int]


def load_yolo_dataset_spec(data_yaml: Path, dataset_root: Path) -> YoloDatasetSpec:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", dataset_root))
    if not base.is_absolute():
        base = (dataset_root / base).resolve()

    splits: Dict[str, Path] = {}
    for split_key in ("train", "val", "test"):
        if split_key in data:
            split_path = Path(data[split_key])
            if not split_path.is_absolute():
                split_path = (base / split_path).resolve()
            splits[split_key] = split_path

    names = _normalize_names(data.get("names"))
    nc = data.get("nc")

    return YoloDatasetSpec(
        data_yaml=data_yaml,
        dataset_root=dataset_root,
        splits=splits,
        names=names,
        nc=nc,
    )


def _normalize_names(names_value) -> Optional[List[str]]:
    if names_value is None:
        return None
    if isinstance(names_value, list):
        return [str(name) for name in names_value]
    if isinstance(names_value, dict):
        return [str(names_value[k]) for k in sorted(names_value)]
    return None


def collect_split_images(
    split_path: Path,
    dataset_root: Path,
    valid_extensions: Tuple[str, ...],
) -> List[Path]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        return _load_image_list_file(split_path, dataset_root)
    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")
    return sorted(
        p for p in split_path.rglob("*") if p.suffix.lower() in valid_extensions
    )


def _load_image_list_file(list_path: Path, dataset_root: Path) -> List[Path]:
    paths: List[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (dataset_root / candidate).resolve()
        paths.append(candidate)
    return paths


def infer_labels_root(
    image_split_dir: Path,
    images_dir_name: str = "images",
    labels_dir_name: str = "labels",
) -> Path:
    parts = list(image_split_dir.parts)
    if images_dir_name in parts:
        idx = parts.index(images_dir_name)
        parts[idx] = labels_dir_name
        return Path(*parts)
    return image_split_dir.parent.parent / labels_dir_name / image_split_dir.name


def map_image_to_label(
    image_path: Path,
    image_split_root: Path,
    labels_split_root: Path,
) -> Path:
    rel = image_path.relative_to(image_split_root)
    return labels_split_root / rel.with_suffix(".txt")


def copy_subset_dataset(
    output_root: Path,
    split_to_images: Dict[str, List[Path]],
    split_to_image_root: Dict[str, Path],
    link_mode: str = "hardlink",
    labels_dir_name: str = "labels",
    images_dir_name: str = "images",
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for split, images in split_to_images.items():
        image_root = split_to_image_root[split]
        labels_root = infer_labels_root(
            image_root, images_dir_name=images_dir_name, labels_dir_name=labels_dir_name
        )
        output_images_root = output_root / images_dir_name / split
        output_labels_root = output_root / labels_dir_name / split

        copied_images = 0
        copied_labels = 0
        for img_path in images:
            rel = img_path.relative_to(image_root)
            dst_img = output_images_root / rel
            dst_lbl = output_labels_root / rel.with_suffix(".txt")
            src_lbl = map_image_to_label(img_path, image_root, labels_root)

            _link_or_copy(img_path, dst_img, link_mode)
            copied_images += 1

            if src_lbl.exists():
                _link_or_copy(src_lbl, dst_lbl, link_mode)
                copied_labels += 1
            else:
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.write_text("", encoding="utf-8")
                copied_labels += 1

        stats[split] = {"images": copied_images, "labels": copied_labels}
    return stats


def write_subset_data_yaml(
    output_yaml: Path,
    dataset_root: Path,
    names: Optional[List[str]],
    nc: Optional[int],
    splits: Iterable[str],
    images_dir_name: str = "images",
) -> None:
    data: Dict[str, object] = {"path": str(dataset_root)}
    for split in splits:
        data[split] = f"{images_dir_name}/{split}"
    if names is not None:
        data["names"] = names
    if nc is not None:
        data["nc"] = nc

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def write_manifest(
    manifest_path: Path,
    split_to_images: Dict[str, List[Path]],
    dataset_root: Path,
) -> None:
    manifest = {}
    for split, images in split_to_images.items():
        rel_paths = []
        for path in images:
            try:
                rel_paths.append(str(path.relative_to(dataset_root)))
            except ValueError:
                rel_paths.append(str(path))
        manifest[split] = rel_paths
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _link_or_copy(src: Path, dst: Path, link_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if link_mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass

    shutil.copy2(src, dst)
