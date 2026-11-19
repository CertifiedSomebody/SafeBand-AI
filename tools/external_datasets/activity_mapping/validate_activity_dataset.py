from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.external_datasets.activity_mapping.build_activity_dataset import (  # noqa: E402
    MODEL_FEATURES,
    OUTPUT_FIELDS,
)


def validate_dataset(path: Path, chunk_size: int = 100_000) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"External activity dataset missing: {path}")
    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    missing = set(OUTPUT_FIELDS) - set(header)
    if missing:
        raise ValueError(f"Processed dataset schema is missing: {sorted(missing)}")
    if set(MODEL_FEATURES) & {
        "dataset_id", "subject_id", "session_id", "source_path",
        "activity_original", "source_activity_detail", "activity_safeband",
        "mapping_status", "hazard_label",
    }:
        raise AssertionError("Metadata leaked into model feature selection")
    if any(feature not in header for feature in MODEL_FEATURES):
        raise ValueError("A declared model feature is absent from the processed dataset")

    rows = 0
    datasets: set[str] = set()
    labels: set[str] = set()
    mapping_statuses: set[str] = set()
    group_subject: dict[tuple[str, str], set[str]] = {}
    nonfinite_features = 0
    nonincreasing_windows = 0
    invalid_availability = 0
    invalid_gyro = 0
    window_counts: Counter[str] = Counter()
    chunks = pd.read_csv(path, compression="gzip", chunksize=chunk_size)
    for frame in chunks:
        rows += len(frame)
        datasets.update(frame["dataset_id"].astype(str).unique())
        labels.update(frame["activity_safeband"].astype(str).unique())
        mapping_statuses.update(frame["mapping_status"].astype(str).unique())
        values = frame.loc[:, MODEL_FEATURES].to_numpy(dtype=float)
        nonfinite_features += int((~np.isfinite(values)).sum())
        starts = pd.to_numeric(frame["timestamp"], errors="coerce").to_numpy(float)
        ends = pd.to_numeric(frame["window_end_timestamp"], errors="coerce").to_numpy(float)
        nonincreasing_windows += int((~np.isfinite(starts) | ~np.isfinite(ends) | (ends < starts)).sum())
        accel_mask = pd.to_numeric(frame["accel_available"], errors="coerce").to_numpy(float)
        gyro_mask = pd.to_numeric(frame["gyro_available"], errors="coerce").to_numpy(float)
        invalid_availability += int((~np.isin(accel_mask, [0, 1]) | ~np.isin(gyro_mask, [0, 1])).sum())
        gyro_values = frame[["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]].to_numpy(float)
        invalid_gyro += int((np.abs(gyro_values).max(axis=1) > 1e-12).sum())
        window_counts.update(frame["window_seconds"].astype(str).value_counts().to_dict())
        for (dataset, subject), sessions in frame.groupby(
            ["dataset_id", "subject_id"], sort=False
        )["session_id"]:
            key = (str(dataset), str(subject))
            seen = set(sessions.astype(str).unique())
            group_subject.setdefault(key, set()).update(seen)

    if rows == 0:
        raise ValueError("Processed activity dataset contains no rows")
    failures = {
        "nonfinite_model_features": nonfinite_features,
        "invalid_window_timestamps": nonincreasing_windows,
        "invalid_availability_flags": invalid_availability,
        "nonzero_unavailable_gyro_values": invalid_gyro,
    }
    if any(failures.values()):
        raise ValueError(f"Processed activity dataset failed invariants: {failures}")
    if datasets != {"aicos_har", "sisfall"}:
        raise ValueError(f"Unexpected processed sources: {sorted(datasets)}")
    if not {"INCLUDED", "SECONDARY_ONLY", "UNMAPPED"}.issubset(mapping_statuses):
        raise ValueError(f"Expected mapping decisions missing: {sorted(mapping_statuses)}")
    if "FALL_OR_HAZARD" not in labels:
        raise ValueError("Staged falls were silently dropped from the separate hazard task")
    return {
        "status": "PASS",
        "rows": rows,
        "datasets": sorted(datasets),
        "safeband_activity_labels": sorted(labels),
        "mapping_statuses": sorted(mapping_statuses),
        "window_row_counts": dict(sorted(window_counts.items())),
        "model_feature_count": len(MODEL_FEATURES),
        "metadata_excluded_from_model_features": True,
        "group_metadata_preserved": bool(group_subject),
        "source_data_modified": False,
    }


def validate_lineage(repo_root: Path) -> dict[str, object]:
    inventory = repo_root / "reports" / "data_audit" / "independent_dataset_inventory.json"
    if not inventory.is_file():
        raise FileNotFoundError(f"Existing lineage audit report missing: {inventory}")
    report = json.loads(inventory.read_text(encoding="utf-8-sig"))
    lineage = report["known_safeband_lineage"]
    if lineage["accepted_generated_inventory_independent_source_sessions"] != 1:
        raise AssertionError("Current SafeBand source-session count changed")
    if lineage["external_datasets_added_to_2_12_session_count"] != 0:
        raise AssertionError("External data was counted toward the SafeBand 2.12 session gate")
    if "BLOCKED" not in lineage["current_2_12_status"]:
        raise AssertionError("2.12 lineage gate is no longer recorded as blocked")
    return {
        "source_session_id": lineage["source_session_id"],
        "independent_source_sessions": 1,
        "status": lineage["current_2_12_status"],
        "external_datasets_added": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate external activity data and immutable lineage invariants.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "external_processed" / "activity" / "activity_windows.csv.gz",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    result = {
        "evaluation_scope": "DEVELOPMENT / COMPONENT-LEVEL EVALUATION",
        "dataset": validate_dataset(args.input),
        "lineage": validate_lineage(args.repo_root),
    }
    out = args.repo_root / "reports" / "external_evaluation" / "activity_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
