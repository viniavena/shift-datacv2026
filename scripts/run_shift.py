"""Run SHIFT frame selection → YOLO training → test evaluation.

Usage
-----
    python scripts/run_shift.py

Environment variables
---------------------
    DATASET_ROOT          Path to the 30 FPS YOLO dataset (train/val splits).
    TEST_DATASET_ROOT     Path to the test YOLO dataset (default: same as DATASET_ROOT).
    DATASET_BASE_FPS      Base FPS of the source video (default: 30).
    YOLO_MODEL            YOLO checkpoint to fine-tune (default: yolo11x.pt).
    YOLO_EPOCHS           Training epochs (default: 50).
    YOLO_BATCH            Batch size (default: 8).
    YOLO_DEVICE           Device for training (default: cuda).
    YOLO_PATIENCE         Early-stopping patience (default: 20).
    YOLO_IMGSZ            Input resolution (default: 640).
    SHIFT_FVI_METHOD      FVI variant for Stage 1: ssvd | ofvd | pixel_diff | fsd
                          (default: ssvd).
    SHIFT_EMBEDDING_MODEL Embedding model for Stage 2 (default: resnet50).
    SHIFT_PCA_DIM         PCA dimensionality for the kernel (default: 128).
    SHIFT_OVERSELECT      Overselection factor alpha (default: 3.0).
    SHIFT_LABEL_AWARE     Set to 1 for the label-aware SHIFT-LA variant (default: 0).
    EMBEDDING_DEVICE      Device for embedding extraction (default: cuda).
    EMBEDDING_BATCH_SIZE  Batch size for embedding extraction (default: 8).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Configuration ──────────────────────────────────────────────────────────

DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/data/avadip_yolo"))
DATASET_YAML = DATASET_ROOT / "data.yaml"
TEST_DATASET_ROOT = Path(os.getenv("TEST_DATASET_ROOT", str(DATASET_ROOT)))
DATASET_BASE_FPS = float(os.getenv("DATASET_BASE_FPS", "30.0"))
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "experiments_output" / "shift_runs"
RESULTS_JSON = OUTPUT_ROOT / "shift_results.json"
CACHE_DIR = OUTPUT_ROOT / "cache"

VIDEO_ID_PATTERNS = [
    r"^(?P<video>.+?)[-_]frame[_-]?(?P<frame>\d+)$",
    r"^(?P<video>.+?)[-_](?P<frame>\d+)$",
]

# Annotation budgets (FPS) to evaluate
BUDGET_FPS_LIST = [float(x) for x in os.getenv("SHIFT_BUDGETS", "0.3,0.5,1,2,3,5,10").split(",")]

# YOLO training
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11x.pt")
YOLO_EPOCHS = int(os.getenv("YOLO_EPOCHS", "50"))
YOLO_BATCH = int(os.getenv("YOLO_BATCH", "8"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")
YOLO_PATIENCE = int(os.getenv("YOLO_PATIENCE", "20"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_SEED = 13

# SHIFT parameters
SHIFT_FVI_METHOD = os.getenv("SHIFT_FVI_METHOD", "ssvd")
SHIFT_EMBEDDING_MODEL = os.getenv("SHIFT_EMBEDDING_MODEL", "resnet50")
SHIFT_PCA_DIM = int(os.getenv("SHIFT_PCA_DIM", "128"))
SHIFT_OVERSELECT = float(os.getenv("SHIFT_OVERSELECT", "3.0"))
SHIFT_LABEL_AWARE = os.getenv("SHIFT_LABEL_AWARE", "0") == "1"
SHIFT_LABEL_ENTROPY_WEIGHT = float(os.getenv("SHIFT_LABEL_ENTROPY_WEIGHT", "0.5"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))


def main() -> None:
    from utils.dataset import (
        collect_split_images,
        copy_subset_dataset,
        load_yolo_dataset_spec,
        write_subset_data_yaml,
    )
    from utils.sampling import group_frames_by_video, sample_shift
    from utils.detector import run_detector_train, run_detector_val

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_spec = load_yolo_dataset_spec(DATASET_YAML, DATASET_ROOT)
    test_images_root = _resolve_test_images(TEST_DATASET_ROOT)

    all_results: List[Dict[str, Any]] = []

    for budget_fps in BUDGET_FPS_LIST:
        exp_id = _build_exp_id(budget_fps)
        print(f"\n{'='*70}")
        print(f"  SHIFT experiment: {exp_id}  (budget={budget_fps} FPS)")
        print(f"{'='*70}")

        exp_root = OUTPUT_ROOT / exp_id
        dataset_out = exp_root / "dataset"
        yolo_project = exp_root / "yolo_runs"

        # ── Frame selection (train and val splits) ─────────────────────
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
            print(f"  [{split_name}] Total frames: {len(images)}")

            grouped = group_frames_by_video(images, VIDEO_ID_PATTERNS)
            selected_all: List[Path] = []
            split_stats: Dict[str, Any] = {"videos": {}}

            for video_id, frames in grouped.items():
                k = _compute_budget(len(frames), budget_fps, DATASET_BASE_FPS)
                selected, stats = sample_shift(
                    frames=frames,
                    k=k,
                    cache_dir=CACHE_DIR / "shift",
                    fvi_method=SHIFT_FVI_METHOD,
                    overselect_factor=SHIFT_OVERSELECT,
                    embedding_model=SHIFT_EMBEDDING_MODEL,
                    device=EMBEDDING_DEVICE,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    pca_dim=SHIFT_PCA_DIM,
                    label_aware=SHIFT_LABEL_AWARE,
                    label_entropy_weight=SHIFT_LABEL_ENTROPY_WEIGHT,
                )
                selected_all.extend(selected)
                split_stats["videos"][video_id] = {
                    "total": len(frames),
                    "selected": len(selected),
                }

            split_stats["total_frames"] = sum(v["total"] for v in split_stats["videos"].values())
            split_stats["selected_frames"] = len(selected_all)
            sampling_stats[split_name] = split_stats
            split_to_images[split_name] = selected_all
            print(
                f"  [{split_name}] Selected: {len(selected_all)} / "
                f"{split_stats['total_frames']} frames"
            )

        # ── Copy test split as-is ──────────────────────────────────────
        split_to_images["test"] = collect_split_images(
            test_images_root, dataset_root=TEST_DATASET_ROOT, valid_extensions=VALID_EXTENSIONS
        )
        split_to_root["test"] = test_images_root
        print(f"  [test]  Frames: {len(split_to_images['test'])}")

        # ── Build subset dataset ───────────────────────────────────────
        copy_stats = copy_subset_dataset(
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
        print(f"  Dataset written to: {dataset_out}")

        # ── Train ──────────────────────────────────────────────────────
        print(f"\n  Training YOLO ({YOLO_MODEL}, {YOLO_EPOCHS} epochs)...")
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
        print(f"  Training done in {train_duration:.0f}s  |  run_dir: {train_result.get('run_dir')}")

        # ── Evaluate on test split ─────────────────────────────────────
        weights_dir = Path(train_result["run_dir"]) / "weights"
        test_evals: Dict[str, Any] = {}

        for ckpt_name in ("best", "last"):
            ckpt_path = weights_dir / f"{ckpt_name}.pt"
            if not ckpt_path.exists():
                print(f"  [WARN] {ckpt_name}.pt not found, skipping eval")
                test_evals[ckpt_name] = {"skipped": True, "reason": "not_found"}
                continue
            print(f"  Evaluating {ckpt_name}.pt on test split...")
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
                extra_args={"plots": True},
            )
            test_evals[ckpt_name] = eval_result
            metrics = eval_result.get("metrics") or {}
            print(f"  [{ckpt_name}] Metrics: {json.dumps(_round_metrics(metrics), indent=4)}")

        # ── Persist result ─────────────────────────────────────────────
        experiment_record = {
            "id": exp_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": {
                "budget_fps": budget_fps,
                "base_fps": DATASET_BASE_FPS,
                "fvi_method": SHIFT_FVI_METHOD,
                "embedding_model": SHIFT_EMBEDDING_MODEL,
                "pca_dim": SHIFT_PCA_DIM,
                "overselect": SHIFT_OVERSELECT,
                "label_aware": SHIFT_LABEL_AWARE,
                "yolo_model": YOLO_MODEL,
                "yolo_epochs": YOLO_EPOCHS,
                "yolo_batch": YOLO_BATCH,
                "yolo_imgsz": YOLO_IMGSZ,
            },
            "sampling_stats": sampling_stats,
            "copy_stats": copy_stats,
            "train": {
                "duration_sec": train_duration,
                "run_dir": train_result.get("run_dir"),
                "metrics": train_result.get("metrics"),
            },
            "test_evaluations": test_evals,
        }
        all_results.append(experiment_record)
        _save_results(all_results)
        print(f"  Results saved to: {RESULTS_JSON}")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for rec in all_results:
        fps = rec["config"]["budget_fps"]
        train_sel = rec["sampling_stats"].get("train", {}).get("selected_frames", "?")
        for ckpt in ("best", "last"):
            ev = rec["test_evaluations"].get(ckpt, {})
            m = ev.get("metrics", {})
            map50 = m.get("metrics/mAP50(B)", m.get("box_map50", "N/A"))
            map5095 = m.get("metrics/mAP50-95(B)", m.get("box_map", "N/A"))
            if isinstance(map50, float):
                map50 = f"{map50:.4f}"
            if isinstance(map5095, float):
                map5095 = f"{map5095:.4f}"
            print(
                f"  {fps:>5} FPS | {ckpt:>4} | train={train_sel:>5} frames"
                f" | mAP50={map50} | mAP50-95={map5095}"
            )


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_exp_id(budget_fps: float) -> str:
    fps_str = f"{budget_fps:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    suffix = f"_{SHIFT_FVI_METHOD}"
    if SHIFT_LABEL_AWARE:
        suffix += "_la"
    return f"shift{suffix}_{fps_str}fps"


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

    raise FileNotFoundError(
        f"Could not locate test images under {test_root}. "
        "Set TEST_DATASET_ROOT to a valid YOLO dataset path."
    )


def _round_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}


def _save_results(results: List[Dict[str, Any]]) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "method": "SHIFT (Selecting High-Information Frames for Training)",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "experiments": results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
