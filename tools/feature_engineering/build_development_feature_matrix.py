from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from feature_extraction import FEATURE_COLUMNS, extract_features


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCEPTANCE_PATH = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "acceptance"
    / "dataset_expansion_acceptance_2.11.9.json"
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
OUTPUT_PATH = (
    ROOT
    / "datasets"
    / "features"
    / "development_feature_matrix_development_only.csv"
)
REPORT_PATH = (
    ROOT
    / "reports"
    / "dataset_training"
    / "development_feature_matrix_2.13.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def build_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    acceptance = load_json(ACCEPTANCE_PATH)
    inventory = load_json(INVENTORY_PATH)
    lineage = load_json(LINEAGE_PATH)
    if acceptance.get("overall_status") != "PASS":
        raise RuntimeError("2.11.9 acceptance must PASS before feature extraction.")
    if inventory.get("overall_status") != "PASS":
        raise RuntimeError("2.11.1 inventory must PASS before feature extraction.")
    if (
        lineage.get("resolved_dataset_count") != lineage.get("accepted_dataset_count")
        or lineage.get("unresolved_dataset_ids")
    ):
        raise RuntimeError("Feature extraction requires fully resolved lineage.")

    lineage_by_id = {
        entry["dataset_id"]: entry
        for entry in lineage.get("datasets", [])
        if entry.get("status") == "RESOLVED"
    }
    categories = {
        filename: category
        for category, details in inventory.get("categories", {}).items()
        for filename in details.get("files", [])
    }
    frames: list[pd.DataFrame] = []
    dataset_summaries: list[dict[str, Any]] = []
    for filename in sorted(categories):
        dataset_id = Path(filename).stem
        dataset_path = DATASET_DIR / filename
        lineage_entry = lineage_by_id.get(dataset_id)
        if lineage_entry is None:
            raise RuntimeError(f"No resolved source session for {dataset_id}.")
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Accepted dataset is missing: {dataset_path}")

        source = pd.read_csv(dataset_path)
        features = extract_features(source)
        output = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "dataset_category": categories[filename],
                "dataset_status": lineage_entry.get(
                    "synthetic_status",
                    "UNKNOWN",
                ),
                "source_session_id": lineage_entry["source_session_id"],
                "sample_index": pd.to_numeric(
                    source["sample_index"],
                    errors="raise",
                ),
                "elapsed_ms": pd.to_numeric(
                    source["elapsed_ms"],
                    errors="raise",
                ),
                "activity": source["activity"].astype(str)
                if "activity" in source.columns
                else "UNKNOWN",
                "hazard_label": source["hazard_label"]
                if "hazard_label" in source.columns
                else 0,
                "hazard_type": source["hazard_type"]
                if "hazard_type" in source.columns
                else "NONE",
                "hazard_event_id": source["hazard_event_id"]
                if "hazard_event_id" in source.columns
                else -1,
                "is_hazard": source["is_hazard"]
                if "is_hazard" in source.columns
                else 0,
            },
            index=source.index,
        )
        frames.append(pd.concat([output, features], axis=1))
        dataset_summaries.append(
            {
                "dataset_id": dataset_id,
                "category": categories[filename],
                "source_session_id": lineage_entry["source_session_id"],
                "rows": len(source),
                "feature_count": len(FEATURE_COLUMNS),
            }
        )

    matrix = pd.concat(frames, ignore_index=True)
    report = {
        "milestone": "2.13",
        "name": "Feature Engineering and Preprocessing",
        "status": "DEVELOPMENT ONLY",
        "classification": "DEVELOPMENT_FEATURE_EXTRACTION_NOT_MODEL_EVALUATION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_acceptance_status": acceptance.get("overall_status"),
        "accepted_dataset_count": len(dataset_summaries),
        "feature_row_count": len(matrix),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns_excluded_from_features": [
            "dataset_id",
            "dataset_category",
            "dataset_status",
            "source_session_id",
            "sample_index",
            "elapsed_ms",
            "activity",
            "hazard_label",
            "hazard_type",
            "hazard_event_id",
            "is_hazard",
        ],
        "rolling_window_ms": 1000,
        "rolling_features_are_causal": True,
        "sensor_masks_included": True,
        "invalid_measurements_masked_to_zero": True,
        "sentinel": -999,
        "labels_used_to_compute_features": False,
        "normalization": "NOT_FIT_NO_TRAIN_PARTITION",
        "model_training": "NOT_RUN",
        "independent_evaluation": "BLOCKED_BY_2.12",
        "split_created": False,
        "source_session_count": lineage.get(
            "independent_source_session_count"
        ),
        "datasets": dataset_summaries,
    }
    return matrix, report


def main() -> int:
    matrix, report = build_matrix()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUTPUT_PATH, index=False, float_format="%.8g")
    report["output_csv"] = OUTPUT_PATH.relative_to(ROOT).as_posix()
    report["report_path"] = REPORT_PATH.relative_to(ROOT).as_posix()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("DEVELOPMENT FEATURE EXTRACTION : COMPLETE")
    print(f"Accepted datasets : {report['accepted_dataset_count']}")
    print(f"Feature rows      : {report['feature_row_count']}")
    print(f"Features per row  : {report['feature_count']}")
    print("Scaling/model/evaluation: NOT RUN")
    print(f"Matrix            : {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Report            : {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
