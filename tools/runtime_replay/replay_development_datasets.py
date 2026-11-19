from __future__ import annotations

import json
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.feature_engineering.feature_extraction import (
    FEATURE_COLUMNS,
    extract_features,
)


INVENTORY_PATH = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "acceptance"
    / "dataset_inventory_2.11.1.json"
)
LINEAGE_PATH = ROOT / "datasets" / "training" / "lineage_manifest.json"
DATASET_DIR = ROOT / "datasets" / "generated"
OUTPUT_PATH = ROOT / "reports" / "runtime_replay" / "development_replay_2.22.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def replay() -> dict[str, Any]:
    inventory = load_json(INVENTORY_PATH)
    lineage = load_json(LINEAGE_PATH)
    if inventory.get("overall_status") != "PASS":
        raise RuntimeError("2.11.1 inventory must PASS before replay.")
    if lineage.get("unresolved_dataset_ids"):
        raise RuntimeError("Replay requires fully resolved source lineage.")

    filenames = sorted(
        {
            filename
            for category in inventory.get("categories", {}).values()
            for filename in category.get("files", [])
        }
    )
    lineage_by_id = {
        entry["dataset_id"]: entry
        for entry in lineage.get("datasets", [])
        if entry.get("status") == "RESOLVED"
    }
    dataset_results: list[dict[str, Any]] = []
    total_rows = 0
    total_elapsed = 0.0
    total_timing_stalls = 0
    availability_counts = {
        name: 0
        for name in (
            "accel_available",
            "gyro_available",
            "orientation_available",
            "environment_available",
            "audio_available",
            "ppg_contact",
            "ppg_available",
        )
    }
    tracemalloc.start()
    try:
        for filename in filenames:
            dataset_id = Path(filename).stem
            lineage_entry = lineage_by_id.get(dataset_id)
            if lineage_entry is None:
                raise RuntimeError(f"No source lineage for {dataset_id}.")
            source_path = DATASET_DIR / filename
            if not source_path.is_file():
                raise FileNotFoundError(f"Accepted dataset is missing: {source_path}")

            source = pd.read_csv(source_path)
            started = time.perf_counter()
            features = extract_features(source)
            elapsed_seconds = time.perf_counter() - started
            if len(features) != len(source):
                raise RuntimeError(f"Replay row mismatch for {dataset_id}.")
            if tuple(features.columns) != FEATURE_COLUMNS:
                raise RuntimeError(f"Feature schema mismatch for {dataset_id}.")
            if not np.isfinite(features.to_numpy(dtype=float)).all():
                raise RuntimeError(f"Non-finite replay feature for {dataset_id}.")

            mask_counts: dict[str, int] = {}
            for mask_name in availability_counts:
                mask = features[mask_name]
                if not mask.isin((0.0, 1.0)).all():
                    raise RuntimeError(
                        f"Non-binary availability mask {mask_name} in {dataset_id}."
                    )
                count = int(mask.sum())
                availability_counts[mask_name] += count
                mask_counts[mask_name] = count
            stalls = int(features["timing_stall"].sum())
            total_timing_stalls += stalls
            total_rows += len(source)
            total_elapsed += elapsed_seconds
            dataset_results.append(
                {
                    "dataset_id": dataset_id,
                    "source_session_id": lineage_entry["source_session_id"],
                    "rows": len(source),
                    "feature_count": len(FEATURE_COLUMNS),
                    "elapsed_seconds": elapsed_seconds,
                    "rows_per_second": (
                        len(source) / elapsed_seconds
                        if elapsed_seconds > 0
                        else None
                    ),
                    "timing_stall_rows": stalls,
                    "available_sample_counts": mask_counts,
                    "status": "PASS",
                }
            )
    finally:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        "milestone": "2.22",
        "name": "Offline Development Replay",
        "status": "DEVELOPMENT ONLY",
        "classification": "FEATURE_PIPELINE_REPLAY_NO_MODEL_INFERENCE",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "replay_mode": "accelerated_offline_csv",
        "dataset_count": len(dataset_results),
        "row_count": total_rows,
        "feature_count": len(FEATURE_COLUMNS),
        "total_feature_extraction_seconds": total_elapsed,
        "aggregate_rows_per_second": (
            total_rows / total_elapsed if total_elapsed > 0 else None
        ),
        "python_peak_traced_memory_bytes": peak_memory_bytes,
        "host_device": "current development host; not wearable hardware",
        "model_loaded": False,
        "fall_predictions": "NOT_RUN",
        "hazard_labels_used_for_inference": False,
        "held_out_evaluation": "BLOCKED_BY_2.12",
        "timing_stall_rows": total_timing_stalls,
        "available_sample_counts": availability_counts,
        "datasets": dataset_results,
    }


def main() -> int:
    report = replay()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("OFFLINE DEVELOPMENT REPLAY : PASS")
    print(f"Datasets                : {report['dataset_count']}")
    print(f"Rows                    : {report['row_count']}")
    print(f"Feature extraction rate : {report['aggregate_rows_per_second']:.1f} rows/s")
    print("Model inference         : NOT RUN")
    print("Hardware performance    : NOT MEASURED")
    print(f"Report                  : {OUTPUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
