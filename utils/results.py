from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import json
import time


SCHEMA_VERSION = "1.0"


def load_results(results_path: Path) -> Dict[str, object]:
    if results_path.exists():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "experiments": [],
    }


def append_result(results_path: Path, experiment: Dict[str, object]) -> None:
    payload = load_results(results_path)
    payload["generated_at"] = _utc_timestamp()
    experiments: List[Dict[str, object]] = payload.get("experiments", [])
    experiments.append(experiment)
    payload["experiments"] = experiments
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
