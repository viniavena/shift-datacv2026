"""YOLO detector training and evaluation wrappers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import contextlib
import csv
import json
import time


def run_detector_train(
    *,
    backend: str = "yolo",
    data_yaml: Path,
    model: str,
    project_dir: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
    patience: Optional[int] = None,
    extra_args: Optional[Dict[str, object]] = None,
    saved_epoch_checkpoints: Optional[list] = None,
) -> Dict[str, object]:
    """Train a detector. Currently supports backend='yolo' only."""
    if backend.lower() not in {"yolo", "ultralytics"}:
        raise ValueError(f"Unsupported backend: {backend!r}. Use 'yolo'.")
    return run_yolo_train(
        data_yaml=data_yaml,
        model=model,
        project_dir=project_dir,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        patience=patience,
        extra_args=extra_args,
    )


def run_detector_val(
    *,
    backend: str = "yolo",
    data_yaml: Path,
    checkpoint: Path,
    project_dir: Path,
    name: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    extra_args: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Evaluate a detector checkpoint. Currently supports backend='yolo' only."""
    if backend.lower() not in {"yolo", "ultralytics"}:
        raise ValueError(f"Unsupported backend: {backend!r}. Use 'yolo'.")
    return run_yolo_val(
        data_yaml=data_yaml,
        checkpoint=checkpoint,
        project_dir=project_dir,
        name=name,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        extra_args=extra_args,
    )


def run_yolo_train(
    *,
    data_yaml: Path,
    model: str,
    project_dir: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
    patience: Optional[int] = None,
    extra_args: Optional[Dict[str, str | int | float | bool]] = None,
) -> Dict[str, object]:
    project_dir.mkdir(parents=True, exist_ok=True)
    run_dir = project_dir / name
    log_path = project_dir / f"{name}_train.log"

    train_args: Dict[str, object] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "seed": seed,
        "project": str(project_dir),
        "name": name,
        "task": "detect",
        "mode": "train",
    }
    if patience is not None:
        train_args["patience"] = patience
    if extra_args:
        train_args.update(extra_args)

    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            from ultralytics import YOLO  # type: ignore

            yolo = YOLO(model)
            yolo.train(**train_args)
    duration_sec = time.time() - start_time

    results_csv = run_dir / "results.csv"
    results_json = run_dir / "results.json"
    metrics = parse_results_csv(results_csv)
    summary = parse_results_json(results_json)

    return {
        "command": "ultralytics.YOLO.train",
        "train_args": train_args,
        "run_dir": str(run_dir),
        "duration_sec": duration_sec,
        "weights": {
            "best": str(run_dir / "weights" / "best.pt"),
            "last": str(run_dir / "weights" / "last.pt"),
        },
        "results_csv": str(results_csv) if results_csv.exists() else None,
        "results_json": str(results_json) if results_json.exists() else None,
        "metrics": metrics,
        "summary": summary,
    }


def run_yolo_val(
    *,
    data_yaml: Path,
    checkpoint: Path,
    project_dir: Path,
    name: str,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    extra_args: Optional[Dict[str, str | int | float | bool]] = None,
) -> Dict[str, object]:
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        return {
            "skipped": True,
            "reason": "checkpoint_not_found",
            "checkpoint": str(checkpoint),
            "split": split,
        }

    project_dir.mkdir(parents=True, exist_ok=True)
    log_path = project_dir / f"{name}_val_{split}.log"

    val_args: Dict[str, object] = {
        "data": str(data_yaml),
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project_dir),
        "name": name,
        "task": "detect",
        "mode": "val",
        "split": split,
    }
    if extra_args:
        val_args.update(extra_args)

    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            from ultralytics import YOLO  # type: ignore

            yolo = YOLO(str(checkpoint))
            val_result = yolo.val(**val_args)
    duration_sec = time.time() - start_time

    return {
        "command": "ultralytics.YOLO.val",
        "checkpoint": str(checkpoint),
        "split": split,
        "val_args": val_args,
        "duration_sec": duration_sec,
        "log_path": str(log_path),
        "metrics": _extract_val_metrics(val_result),
        "run_dir": _extract_save_dir(val_result),
    }


def parse_results_csv(csv_path: Path) -> Optional[Dict[str, float]]:
    if not csv_path.exists():
        return None
    last_row: Optional[Dict[str, str]] = None
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            last_row = row
    if not last_row:
        return None
    metrics: Dict[str, float] = {}
    for key, value in last_row.items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def parse_results_json(json_path: Path) -> Optional[Dict[str, object]]:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_save_dir(val_result: Any) -> Optional[str]:
    if val_result is None:
        return None
    save_dir = getattr(val_result, "save_dir", None)
    if save_dir is None:
        return None
    try:
        return str(save_dir)
    except Exception:
        return None


def _extract_val_metrics(val_result: Any) -> Optional[Dict[str, object]]:
    if val_result is None:
        return None

    raw_dict = getattr(val_result, "results_dict", None)
    if isinstance(raw_dict, dict):
        return _to_jsonable_metrics(raw_dict)

    if isinstance(val_result, dict):
        return _to_jsonable_metrics(val_result)

    metrics: Dict[str, object] = {}
    for attr in ("fitness", "speed"):
        if hasattr(val_result, attr):
            metrics[attr] = _coerce_metric_value(getattr(val_result, attr))

    box_obj = getattr(val_result, "box", None)
    if box_obj is not None:
        for attr in ("map", "map50", "map75", "mp", "mr"):
            if hasattr(box_obj, attr):
                metrics[f"box_{attr}"] = _coerce_metric_value(getattr(box_obj, attr))

    return metrics or None


def _to_jsonable_metrics(values: Dict[str, Any]) -> Dict[str, object]:
    return {str(k): _coerce_metric_value(v) for k, v in values.items()}


def _coerce_metric_value(value: Any) -> object:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_metric_value(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return str(value)
