"""Run SHIFT ablation study at 1.0 FPS (Table 3 in the paper).

Ablations evaluated
-------------------
  ab_fvi_ofvd      — Replace SSIM FVI (Stage 1) with optical-flow FVI.
  ab_fvi_pixdiff   — Replace SSIM FVI (Stage 1) with pixel-difference FVI.
  ab_stage1_only   — Stage 1 only (no entropy maximization).
  ab_uniform_s2    — Stage 1 replaced by uniform temporal sampling (Stage 2 only).

All configurations produce 1.0 FPS budgets and train a YOLO11X detector.

Usage
-----
    python scripts/run_ablations.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List

from utils.dataset import (
    collect_split_images,
    copy_subset_dataset,
    load_yolo_dataset_spec,
    write_subset_data_yaml,
)
from utils.sampling import group_frames_by_video, sample_shift
from utils.detector import run_detector_train, run_detector_val

# ── Configuration ──────────────────────────────────────────────────────────

import os

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/data/avadip_yolo"))
DATASET_YAML = DATASET_ROOT / "data.yaml"
TEST_DATASET_ROOT = Path(os.getenv("TEST_DATASET_ROOT", str(DATASET_ROOT)))
DATASET_BASE_FPS = 30.0
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
VIDEO_ID_PATTERNS = [
    r"^(?P<video>.+?)[-_]frame[_-]?(?P<frame>\d+)$",
    r"^(?P<video>.+?)[-_](?P<frame>\d+)$",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "experiments_output" / "shift_ablations_1fps"
RESULTS_JSON = OUTPUT_ROOT / "ablation_results.json"
CACHE_DIR = OUTPUT_ROOT / "cache"

BUDGET_FPS = 1.0

# Default SHIFT hyperparameters (same as the main paper experiments)
SHIFT_DEFAULT = {
    "fvi_method": "ssvd",
    "overselect_factor": 3.0,
    "embedding_model": "resnet50",
    "pca_dim": 128,
    "kernel_epsilon": 1e-4,
    "label_aware": False,
}

ABLATIONS = [
    # Vary Stage 1 variation signal
    {
        "id_suffix": "ab_fvi_ofvd",
        "ablation_mode": "full",
        "fvi_method": "ofvd",
    },
    {
        "id_suffix": "ab_fvi_pixdiff",
        "ablation_mode": "full",
        "fvi_method": "pixel_diff",
    },
    # Remove Stage 2
    {
        "id_suffix": "ab_stage1_only",
        "ablation_mode": "stage1_only",
        "fvi_method": "ssvd",
    },
    # Replace Stage 1 with uniform (only Stage 2 active)
    {
        "id_suffix": "ab_uniform_s2",
        "ablation_mode": "stage2_only_uniform",
        "fvi_method": "ssvd",
    },
]

YOLO_MODEL = "yolo11x.pt"
YOLO_EPOCHS = 50
YOLO_BATCH = 8
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")
YOLO_PATIENCE = 20
YOLO_IMGSZ = 640
YOLO_SEED = 13
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = 8


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dataset_spec = load_yolo_dataset_spec(DATASET_YAML, DATASET_ROOT)
    test_images_root = _resolve_test_images(TEST_DATASET_ROOT)

    all_results: List[Dict[str, Any]] = []

    for ablation in ABLATIONS:
        cfg = {**SHIFT_DEFAULT, **{k: v for k, v in ablation.items() if k != "id_suffix"}}
        exp_id = f"shift_{ablation['id_suffix']}_{BUDGET_FPS:.1f}fps"

        print(f"\n{'='*70}")
        print(f"  Ablation: {exp_id}")
        print(f"  Config:   {cfg}")
        print(f"{'='*70}")

        exp_root = OUTPUT_ROOT / exp_id
        dataset_out = exp_root / "dataset"
        yolo_project = exp_root / "yolo_runs"

        split_to_images: Dict[str, List[Path]] = {}
        split_to_root: Dict[str, Path] = {}
        sampling_stats: Dict[str, Any] = {}

        for split_name in ("train", "val"):
            if split_name not in dataset_spec.splits:
                continue
            split_path = dataset_spec.splits[split_name]
            split_to_root[split_name] = split_path

            images = collect_split_images(
                split_path, dataset_root=DATASET_ROOT, valid_extensions=VALID_EXTENSIONS
            )
            grouped = group_frames_by_video(images, VIDEO_ID_PATTERNS)
            selected_all: List[Path] = []
            split_stats: Dict[str, Any] = {"videos": {}}

            for video_id, frames in grouped.items():
                k = _compute_budget(len(frames), BUDGET_FPS, DATASET_BASE_FPS)
                selected, stats = sample_shift(
                    frames=frames,
                    k=k,
                    cache_dir=CACHE_DIR / "shift",
                    fvi_method=str(cfg["fvi_method"]),
                    overselect_factor=float(cfg["overselect_factor"]),
                    embedding_model=str(cfg["embedding_model"]),
                    device=EMBEDDING_DEVICE,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    pca_dim=int(cfg["pca_dim"]),
                    kernel_epsilon=float(cfg["kernel_epsilon"]),
                    label_aware=bool(cfg["label_aware"]),
                    ablation_mode=str(cfg["ablation_mode"]),
                )
                selected_all.extend(selected)
                split_stats["videos"][video_id] = {"total": len(frames), "selected": len(selected)}

            split_stats["total_frames"] = sum(v["total"] for v in split_stats["videos"].values())
            split_stats["selected_frames"] = len(selected_all)
            sampling_stats[split_name] = split_stats
            split_to_images[split_name] = selected_all
            print(
                f"  [{split_name}] {len(selected_all)} / "
                f"{split_stats['total_frames']} frames selected"
            )

        split_to_images["test"] = collect_split_images(
            test_images_root, dataset_root=TEST_DATASET_ROOT, valid_extensions=VALID_EXTENSIONS
        )
        split_to_root["test"] = test_images_root

        copy_subset_dataset(
            output_root=dataset_out,
            split_to_images=split_to_images,
            split_to_image_root=split_to_root,
            link_mode="hardlink",
        )
        data_yaml = dataset_out / "data.yaml"
        write_subset_data_yaml(
            output_yaml=data_yaml,
            dataset_root=dataset_out,
            names=dataset_spec.names,
            nc=dataset_spec.nc,
            splits=["train", "val", "test"],
        )

        print(f"  Training YOLO ({YOLO_MODEL}, {YOLO_EPOCHS} epochs)...")
        train_start = time.time()
        train_result = run_detector_train(
            backend="yolo",
            data_yaml=data_yaml,
            model=YOLO_MODEL,
            project_dir=yolo_project,
            name="train",
            epochs=YOLO_EPOCHS,
            imgsz=YOLO_IMGSZ,
            batch=YOLO_BATCH,
            device=YOLO_DEVICE,
            seed=YOLO_SEED,
            patience=YOLO_PATIENCE,
            extra_args={"cache": False, "exist_ok": True},
        )
        train_duration = time.time() - train_start

        weights_dir = Path(train_result["run_dir"]) / "weights"
        test_evals: Dict[str, Any] = {}
        for ckpt_name in ("best", "last"):
            ckpt_path = weights_dir / f"{ckpt_name}.pt"
            if not ckpt_path.exists():
                test_evals[ckpt_name] = {"skipped": True, "reason": "not_found"}
                continue
            eval_result = run_detector_val(
                backend="yolo",
                data_yaml=data_yaml,
                checkpoint=ckpt_path,
                project_dir=yolo_project,
                name=f"eval_{ckpt_name}",
                split="test",
                imgsz=YOLO_IMGSZ,
                batch=YOLO_BATCH,
                device=YOLO_DEVICE,
                extra_args={"plots": False},
            )
            test_evals[ckpt_name] = eval_result
            metrics = eval_result.get("metrics") or {}
            map5095 = metrics.get("metrics/mAP50-95(B)", metrics.get("box_map", "N/A"))
            print(f"  [{ckpt_name}] mAP50-95 = {map5095}")

        record = {
            "id": exp_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {**cfg, "budget_fps": BUDGET_FPS},
            "sampling_stats": sampling_stats,
            "train": {
                "duration_sec": train_duration,
                "run_dir": train_result.get("run_dir"),
                "metrics": train_result.get("metrics"),
            },
            "test_evaluations": test_evals,
        }
        all_results.append(record)
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(
            json.dumps(
                {
                    "experiment": "SHIFT ablation study (1.0 FPS)",
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "results": all_results,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"  Results saved to: {RESULTS_JSON}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  ABLATION SUMMARY (1 FPS)\n{'='*70}")
    for rec in all_results:
        ev = rec["test_evaluations"].get("best", {})
        m = ev.get("metrics", {})
        map5095 = m.get("metrics/mAP50-95(B)", m.get("box_map", "N/A"))
        if isinstance(map5095, float):
            map5095 = f"{map5095:.4f}"
        print(f"  {rec['id']:<50}  mAP50-95={map5095}")


def _compute_budget(total: int, budget_fps: float, base_fps: float) -> int:
    keep_ratio = min(1.0, budget_fps / base_fps)
    return max(1, math.ceil(total * keep_ratio))


def _resolve_test_images(test_root: Path) -> Path:
    test_yaml = test_root / "data.yaml"
    if test_yaml.exists():
        import yaml

        data = yaml.safe_load(test_yaml.read_text(encoding="utf-8"))
        base = Path(data.get("path", test_root))
        if not base.is_absolute():
            base = (test_root / base).resolve()
        for key in ("test", "val", "train"):
            if key in data:
                p = Path(data[key])
                if not p.is_absolute():
                    p = (base / p).resolve()
                if p.exists():
                    return p

    for candidate in (
        test_root / "images" / "test",
        test_root / "images" / "val",
        test_root / "images",
        test_root,
    ):
        if candidate.exists() and any(candidate.rglob("*.jpg")):
            return candidate

    raise FileNotFoundError(f"Could not locate test images under {test_root}.")


if __name__ == "__main__":
    main()
