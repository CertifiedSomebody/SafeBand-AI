from __future__ import annotations

import json
import sys
from pathlib import Path

from build_training_dataset import CONFIG_PATH, load_json, resolve_path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    config = load_json(CONFIG_PATH)
    report_path = resolve_path(config["acceptance_output"])
    lineage_path = resolve_path(config["lineage_manifest"])
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Run the lineage audit first; report is missing: {report_path}"
        )
    report = load_json(report_path)
    if lineage_path.is_file():
        lineage = load_json(lineage_path)
        report["independent_source_session_count"] = lineage.get(
            "independent_source_session_count"
        )
        report["unresolved_dataset_count"] = len(
            lineage.get("unresolved_dataset_ids", [])
        )
    report["training_splits_created"] = False
    report["split_validation_status"] = "BLOCKED_NOT_CREATED"
    report["status"] = "BLOCKED"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("TRAINING DATASET VALIDATION : BLOCKED")
    print("Reason: no independent group-level partitions exist.")
    print("No leakage-free train/validation/test package is claimed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
