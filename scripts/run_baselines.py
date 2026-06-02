"""Run baseline frame-selection experiments → YOLO training → test evaluation.

Baselines implemented
---------------------
  baseline_full  : Train on the complete dataset (no selection).
  ufs            : Uniform Frame Sampling.
  rfs            : Random Frame Sampling (averaged over multiple seeds).
  afs_ssvd       : Adaptive Frame Sampling — SSIM FVI variant.
  afs_ofvd       : Adaptive Frame Sampling — optical-flow FVI variant.
  afs_fsd        : Adaptive Frame Sampling — feature-similarity FVI variant.
  csod           : Coreset Selection for Object Detection (requires annotations).

Usage
-----
    python scripts/run_baselines.py

Environment variables
---------------------
    DATASET_ROOT        Path to the YOLO dataset (train/val splits).
    TEST_DATASET_ROOT   Path to the test YOLO dataset (default: DATASET_ROOT).
    DATASET_BASE_FPS    Base FPS of the source video (default: 30).
    YOLO_MODEL          YOLO checkpoint (default: yolo11x.pt).
    YOLO_EPOCHS         Training epochs (default: 50).
    YOLO_BATCH          Batch size (default: 8).
    YOLO_DEVICE         Training device (default: cuda).
    YOLO_PATIENCE       Early-stopping patience (default: 20).
    YOLO_IMGSZ          Input resolution (default: 640).
    EXPERIMENT_METHODS  Comma-separated list of methods to run.
                        Default: baseline_full,ufs,rfs,afs_ssvd,afs_ofvd,afs_fsd,csod
    BUDGET_FPS          Comma-separated list of annotation budgets in FPS.
                        Default: 0.33,0.5,1,2,3,5,10
    N_CLASSES           Number of object classes for CSOD (default: 6).
    EMBEDDING_DEVICE    Device for feature extraction (default: cuda).
    EMBEDDING_BATCH     Batch size for feature extraction (default: 8).
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
OUTPUT_ROOT = REPO_ROOT / "experiments_output" / "baselines"
RESULTS_JSON = OUTPUT_ROOT / "baseline_results.json"
LINK_MODE = "hardlink"

VIDEO_ID_PATTERNS = [
    r"^(?P<video>.+?)[-_]frame[_-]?(?P<frame>\d+)$",
    r"^(?P<video>.+?)[-_](?P<frame>\d+)$",
]

def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if val is None:
        return list(default)
    return [x.strip() for x in val.split(",") if x.strip()]

def _env_floats(name: str, default: List[float]) -> List[float]:
    val = os.getenv(name)
    if val is None:
        return list(default)
    return [float(x.strip()) for x in val.split(",") if x.strip()]

BUDGET_FPS = _env_floats("BUDGET_FPS", [0.33, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
EXPERIMENT_METHODS = _env_list(
    "EXPERIMENT_METHODS",
    ["baseline_full", "ufs", "rfs", "afs_ssvd", "afs_ofvd", "afs_fsd", "csod"],
)

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11x.pt")
YOLO_EPOCHS = int(os.getenv("YOLO_EPOCHS", "50"))
YOLO_BATCH = int(os.getenv("YOLO_BATCH", "8"))
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "cuda")
YOLO_PATIENCE = int(os.getenv("YOLO_PATIENCE", "20"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_SEED = 13

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")
EMBEDDING_BATCH = int(os.getenv("EMBEDDING_BATCH", "8"))
CSOD_N_CLASSES = int(os.getenv("N_CLASSES", "6"))
RANDOM_SEEDS = [13, 42, 97, 123, 256]
CACHE_DIR = OUTPUT_ROOT / "cache"


def main() -> None:
    from utils.dataset import (
        collect_split_images,
        copy_subset_dataset,
        load_yolo_dataset_spec,
        write_manifest,
        write_subset_data_yaml,
    )
    from utils.results import append_result
    from utils.sampling import (
        group_frames_by_video,
        sample_uniform,
        sample_random,
        sample_afs_ssvd,
        sample_afs_ofvd,
        sample_afs_fsd,
        sample_csod,
    )
    from utils.detector import run_detector_train, run_detector_val

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_spec = load_yolo_dataset_spec(DATASET_YAML, DATASET_ROOT)
    test_images_root = _resolve_test_images(TEST_DATASET_ROOT)

    full_train_count = len(
        collect_split_images(
            dataset_spec.splits["train"],
            dataset_root=DATASET_ROOT,
            valid_extensions=VALID_EXTENSIONS,
        )
    )

    for experiment in build_experiments(dataset_spec):
        exp_id = experiment["id"]
        method = experiment["method"]
        exp_root = OUTPUT_ROOT / exp_id
        dataset_out = exp_root / "dataset"
        yolo_project = exp_root / "yolo_runs"

        print(f"\n{'='*70}")
        print(f"  Experiment: {exp_id}  (method={method})")
        print(f"{'='*70}")

        # ── Frame selection ────────────────────────────────────────────
        split_to_images: Dict[str, List[Path]] = {}
        split_to_root: Dict[str, Path] = {}

        for split_name in ("train", "val"):
            if split_name not in dataset_spec.splits:
                continue
            split_path = dataset_spec.splits[split_name]
            split_to_root[split_name] = split_path
            images = collect_split_images(
                split_path, dataset_root=DATASET_ROOT, valid_extensions=VALID_EXTENSIONS
            )
            selected = _apply_sampling(
                images=images,
                experiment=experiment,
                base_fps=DATASET_BASE_FPS,
                split_name=split_name,
            )
            split_to_images[split_name] = selected
            print(f"  [{split_name}] {len(selected)} / {len(images)} frames selected")

        # Test split: use all frames
        test_images = collect_split_images(
            test_images_root, dataset_root=TEST_DATASET_ROOT, valid_extensions=VALID_EXTENSIONS
        )
        split_to_images["test"] = test_images
        split_to_root["test"] = test_images_root

        # ── Build dataset ──────────────────────────────────────────────
        copy_stats = copy_subset_dataset(
            output_root=dataset_out,
            split_to_images=split_to_images,
            split_to_image_root=split_to_root,
            link_mode=LINK_MODE,
        )
        data_yaml = dataset_out / "data.yaml"
        write_subset_data_yaml(
            output_yaml=data_yaml,
            dataset_root=dataset_out,
            names=dataset_spec.names,
            nc=dataset_spec.nc,
            splits=split_to_images.keys(),
        )
        write_manifest(
            manifest_path=dataset_out / "manifest.json",
            split_to_images=split_to_images,
            dataset_root=DATASET_ROOT,
        )

        # ── Scale epochs proportionally to subset size ─────────────────
        train_count = len(split_to_images.get("train", []))
        epochs = _scale_epochs(
            base_epochs=YOLO_EPOCHS,
            full_count=full_train_count,
            subset_count=train_count,
        )

        # ── Train ──────────────────────────────────────────────────────
        print(f"  Training YOLO ({YOLO_MODEL}, {epochs} epochs, {train_count} frames)...")
        train_start = time.time()
        train_result = run_detector_train(
            backend="yolo",
            data_yaml=data_yaml,
            model=YOLO_MODEL,
            project_dir=yolo_project,
            name="train",
            epochs=epochs,
            imgsz=YOLO_IMGSZ,
            batch=YOLO_BATCH,
            device=YOLO_DEVICE,
            seed=YOLO_SEED,
            patience=YOLO_PATIENCE,
            extra_args={"cache": False, "exist_ok": True},
        )
        train_duration = time.time() - train_start
        print(f"  Training done in {train_duration:.0f}s")

        # ── Evaluate on test split ─────────────────────────────────────
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
            print(f"  [{ckpt_name}] mAP50-95={map5095}")

        record = {
            "id": exp_id,
            "method": method,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": experiment,
            "copy_stats": copy_stats,
            "train": {
                "train_count": train_count,
                "epochs": epochs,
                "duration_sec": train_duration,
                "run_dir": train_result.get("run_dir"),
                "metrics": train_result.get("metrics"),
            },
            "test_evaluations": test_evals,
        }
        append_result(RESULTS_JSON, record)
        print(f"  Results appended to: {RESULTS_JSON}")


# ── Experiment builder ─────────────────────────────────────────────────────

def build_experiments(dataset_spec) -> List[Dict[str, object]]:
    experiments: List[Dict[str, object]] = []

    for method in EXPERIMENT_METHODS:
        if method == "baseline_full":
            experiments.append({"id": "baseline_full", "method": "baseline_full"})
        elif method == "ufs":
            for fps in BUDGET_FPS:
                experiments.append({
                    "id": _exp_id("ufs", fps),
                    "method": "ufs",
                    "budget_fps": fps,
                })
        elif method == "rfs":
            for fps in BUDGET_FPS:
                for seed in RANDOM_SEEDS:
                    experiments.append({
                        "id": _exp_id("rfs", fps, suffix=f"seed{seed}"),
                        "method": "rfs",
                        "budget_fps": fps,
                        "seed": seed,
                    })
        elif method == "afs_ssvd":
            for fps in BUDGET_FPS:
                experiments.append({
                    "id": _exp_id("afs_ssvd", fps),
                    "method": "afs_ssvd",
                    "budget_fps": fps,
                })
        elif method == "afs_ofvd":
            for fps in BUDGET_FPS:
                experiments.append({
                    "id": _exp_id("afs_ofvd", fps),
                    "method": "afs_ofvd",
                    "budget_fps": fps,
                })
        elif method == "afs_fsd":
            for fps in BUDGET_FPS:
                experiments.append({
                    "id": _exp_id("afs_fsd", fps),
                    "method": "afs_fsd",
                    "budget_fps": fps,
                })
        elif method == "csod":
            for fps in BUDGET_FPS:
                experiments.append({
                    "id": _exp_id("csod", fps),
                    "method": "csod",
                    "budget_fps": fps,
                })
        else:
            raise ValueError(f"Unknown method: {method!r}")

    return experiments


# ── Sampling dispatcher ────────────────────────────────────────────────────

def _apply_sampling(
    images: List[Path],
    experiment: Dict[str, object],
    base_fps: float,
    split_name: str,
) -> List[Path]:
    from utils.sampling import (
        group_frames_by_video,
        sample_uniform,
        sample_random,
        sample_afs_ssvd,
        sample_afs_ofvd,
        sample_afs_fsd,
        sample_csod,
    )

    method = str(experiment["method"])

    if method == "baseline_full":
        return list(images)

    budget_fps = float(experiment["budget_fps"])
    grouped = group_frames_by_video(images, VIDEO_ID_PATTERNS)
    selected: List[Path] = []

    for frames in grouped.values():
        k = _compute_budget(len(frames), budget_fps, base_fps)

        if method == "ufs":
            stride = max(1, math.ceil(base_fps / budget_fps))
            selected.extend(sample_uniform(frames, stride=stride))
        elif method == "rfs":
            seed = int(experiment.get("seed", RANDOM_SEEDS[0]))
            selected.extend(sample_random(frames, k=k, seed=seed))
        elif method == "afs_ssvd":
            selected.extend(sample_afs_ssvd(frames, k=k, cache_dir=CACHE_DIR / "afs_ssvd"))
        elif method == "afs_ofvd":
            selected.extend(sample_afs_ofvd(frames, k=k, cache_dir=CACHE_DIR / "afs_ofvd"))
        elif method == "afs_fsd":
            selected.extend(
                sample_afs_fsd(
                    frames, k=k,
                    cache_dir=CACHE_DIR / "afs_fsd",
                    device=EMBEDDING_DEVICE,
                    batch_size=EMBEDDING_BATCH,
                )
            )
        elif method == "csod":
            selected.extend(
                sample_csod(
                    frames, k=k,
                    cache_dir=CACHE_DIR / "csod",
                    device=EMBEDDING_DEVICE,
                    batch_size=EMBEDDING_BATCH,
                    n_classes=CSOD_N_CLASSES,
                )
            )
        else:
            raise ValueError(f"Unknown sampling method: {method!r}")

    return selected


# ── Helpers ────────────────────────────────────────────────────────────────

def _exp_id(method: str, fps: float, suffix: str = "") -> str:
    fps_str = f"{fps:.2f}".rstrip("0").rstrip(".").replace(".", "_")
    base = f"{method}_{fps_str}fps"
    return f"{base}_{suffix}" if suffix else base


def _compute_budget(total: int, budget_fps: float, base_fps: float) -> int:
    keep_ratio = min(1.0, budget_fps / base_fps)
    return max(1, math.ceil(total * keep_ratio))


def _scale_epochs(base_epochs: int, full_count: int, subset_count: int) -> int:
    if full_count <= 0 or subset_count <= 0:
        return base_epochs
    scale = full_count / subset_count
    scaled = int(round(base_epochs * scale))
    return max(10, min(200, scaled))


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


if __name__ == "__main__":
    main()
