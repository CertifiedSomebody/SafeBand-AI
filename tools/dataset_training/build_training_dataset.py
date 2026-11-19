from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("training_dataset_config.json")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_source_reference(
    value: Any,
    real_source_dir: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.replace("\\", "/").strip()
    if normalized.upper() == "UNKNOWN":
        return None

    candidates = [resolve_path(normalized)]
    relative = Path(normalized)
    candidates.extend(
        [
            ROOT / relative,
            real_source_dir / relative.name,
            real_source_dir / relative,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def row_signature_matches(
    source_path: Path,
    generated_path: Path,
) -> bool:
    import pandas as pd

    source = pd.read_csv(source_path)
    generated = pd.read_csv(generated_path)
    if len(source) != len(generated):
        return False

    stable_columns = [
        column
        for column in (
            "sample_index",
            "elapsed_ms",
            "activity",
        )
        if column in source.columns and column in generated.columns
    ]
    return (
        len(stable_columns) == 3
        and all(source[column].equals(generated[column]) for column in stable_columns)
    )


def preserved_sensor_values_match(
    source_path: Path,
    generated_path: Path,
) -> bool:
    import pandas as pd

    source = pd.read_csv(source_path)
    generated = pd.read_csv(generated_path)
    if len(source) != len(generated):
        return False
    columns = [
        column
        for column in source.columns
        if column in generated.columns
        and column not in {"elapsed_ms", "hazard_label", "hazard_type", "hazard_event_id", "is_hazard"}
    ]
    return (
        len(columns) >= 20
        and "sample_index" in columns
        and all(source[column].equals(generated[column]) for column in columns)
    )


def validation_source(
    entry: dict[str, Any],
    dataset_path: Path,
    real_source_dir: Path,
) -> Path | None:
    report_value = entry.get("validation_report")
    if not isinstance(report_value, str) or not report_value.strip():
        return None
    report_path = resolve_path(report_value)
    if not report_path.is_file():
        return None
    report = load_json(report_path)
    for key in ("source", "source_path", "source_dataset"):
        source_path = resolve_source_reference(
            report.get(key),
            real_source_dir,
        )
        if source_path is not None:
            return source_path
    return None


def find_source_session(
    entry: dict[str, Any],
    dataset_path: Path,
    real_source_dir: Path,
) -> tuple[Path | None, str]:
    for key in (
        "source_path",
        "source_dataset",
        "parent_dataset",
    ):
        source_path = resolve_source_reference(
            entry.get(key),
            real_source_dir,
        )
        if source_path is not None:
            return source_path, f"manifest.{key}"

    native_value = entry.get("provenance_file")
    if isinstance(native_value, str) and native_value.strip():
        native_path = resolve_path(native_value)
        if native_path.is_file():
            native = load_json(native_path)
            for key in (
                "source",
                "source_path",
                "source_dataset",
                "parent_dataset",
            ):
                source_path = resolve_source_reference(
                    native.get(key),
                    real_source_dir,
                )
                if source_path is not None:
                    return source_path, f"native_provenance.{key}"

    source_path = validation_source(
        entry,
        dataset_path,
        real_source_dir,
    )
    if source_path is not None:
        return source_path, "validation_report"

    matches = [
        candidate.resolve()
        for candidate in sorted(real_source_dir.glob("*.csv"))
        if row_signature_matches(candidate, dataset_path)
    ]
    if len(matches) == 0:
        preserved_value_matches = [
            candidate.resolve()
            for candidate in sorted(real_source_dir.glob("*.csv"))
            if preserved_sensor_values_match(candidate, dataset_path)
        ]
        if len(preserved_value_matches) == 1:
            return (
                preserved_value_matches[0],
                "preserved_sensor_values_excluding_regenerated_timing",
            )
        if len(preserved_value_matches) > 1:
            raise ValueError(
                f"Ambiguous preserved sensor source for {dataset_path.name}: "
                + ", ".join(path.name for path in preserved_value_matches)
            )
    if len(matches) == 1:
        return matches[0], "preserved_sample_index_elapsed_activity"
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous source session for {dataset_path.name}: "
            + ", ".join(path.name for path in matches)
        )
    return None, "unresolved"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lineage() -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_json(CONFIG_PATH)
    acceptance_path = resolve_path(config["acceptance_report"])
    inventory_path = resolve_path(config["inventory_report"])
    session_manifest_path = resolve_path(config["session_manifest"])
    real_source_dir = resolve_path(config["real_source_directory"])
    generated_dir = resolve_path(config["generated_directory"])

    acceptance = load_json(acceptance_path)
    inventory = load_json(inventory_path)
    session_manifest = load_json(session_manifest_path)
    if acceptance.get("overall_status") != "PASS":
        raise RuntimeError(
            "2.11.9 final acceptance must PASS before lineage audit."
        )
    if inventory.get("overall_status") != "PASS":
        raise RuntimeError(
            "2.11.1 dataset inventory must PASS before lineage audit."
        )

    manifest_entries = {
        entry.get("dataset_id"): entry
        for entry in session_manifest.get("datasets", [])
        if isinstance(entry, dict)
    }
    inventory_files = sorted(
        {
            filename
            for category in inventory.get("categories", {}).values()
            for filename in category.get("files", [])
        }
    )
    if not inventory_files:
        raise ValueError("Acceptance inventory contains no dataset files.")

    datasets: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for filename in inventory_files:
        dataset_path = generated_dir / filename
        if not dataset_path.is_file():
            unresolved.append(filename)
            datasets.append(
                {
                    "dataset_id": Path(filename).stem,
                    "dataset": filename,
                    "source_session_id": None,
                    "source_path": None,
                    "lineage_evidence": "accepted dataset file missing",
                    "status": "UNRESOLVED",
                }
            )
            continue

        dataset_id = dataset_path.stem
        entry = manifest_entries.get(dataset_id)
        if entry is None:
            unresolved.append(filename)
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": filename,
                    "source_session_id": None,
                    "source_path": None,
                    "lineage_evidence": "dataset missing from provenance manifest",
                    "status": "UNRESOLVED",
                }
            )
            continue

        try:
            source_path, evidence = find_source_session(
                entry,
                dataset_path,
                real_source_dir,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            unresolved.append(filename)
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": filename,
                    "source_session_id": None,
                    "source_path": None,
                    "lineage_evidence": str(exc),
                    "status": "UNRESOLVED",
                }
            )
            continue

        if source_path is None:
            unresolved.append(filename)
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": filename,
                    "source_session_id": None,
                    "source_path": None,
                    "lineage_evidence": evidence,
                    "status": "UNRESOLVED",
                }
            )
            continue

        source_hash = file_sha256(source_path)
        source_session_id = f"SESSION_SHA256_{source_hash[:16].upper()}"
        datasets.append(
            {
                "dataset_id": dataset_id,
                "dataset": filename,
                "source_session_id": source_session_id,
                "source_path": source_path.relative_to(ROOT).as_posix(),
                "source_sha256": source_hash,
                "lineage_evidence": evidence,
                "generation_stage": entry.get("generation_stage"),
                "synthetic_status": entry.get("synthetic_status"),
                "activity": entry.get("activity"),
                "profile": entry.get("profile"),
                "row_count": entry.get("row_count"),
                "column_count": entry.get("column_count"),
                "status": "RESOLVED",
            }
        )

    sessions: dict[str, list[str]] = {}
    for dataset in datasets:
        session_id = dataset.get("source_session_id")
        if session_id:
            sessions.setdefault(session_id, []).append(
                dataset["dataset_id"]
            )
    manifest = {
        "manifest_schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "classification": "DEVELOPMENT_LINEAGE_AUDIT",
        "source_acceptance_report": acceptance_path.relative_to(ROOT).as_posix(),
        "source_inventory_report": inventory_path.relative_to(ROOT).as_posix(),
        "source_provenance_manifest": session_manifest_path.relative_to(ROOT).as_posix(),
        "accepted_dataset_count": len(inventory_files),
        "resolved_dataset_count": sum(
            dataset["status"] == "RESOLVED" for dataset in datasets
        ),
        "unresolved_dataset_ids": [
            Path(filename).stem for filename in unresolved
        ],
        "independent_source_session_count": len(sessions),
        "source_sessions": {
            session_id: sorted(dataset_ids)
            for session_id, dataset_ids in sorted(sessions.items())
        },
        "datasets": datasets,
        "split_created": False,
        "warning": (
            "Derived datasets retain their original source-session identity; "
            "augmentation does not create an independent session."
        ),
    }
    audit = {
        "milestone": "2.12",
        "name": "Training-Ready Dataset Packaging",
        "status": "BLOCKED",
        "classification": "DELIBERATE_DATA_INTEGRITY_GATE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "final_dataset_acceptance": acceptance.get("overall_status")
            == "PASS",
            "accepted_inventory_read": len(inventory_files) > 0,
            "all_dataset_lineage_resolved": not unresolved,
            "minimum_independent_source_sessions": len(sessions)
            >= config["minimum_independent_source_sessions"],
            "group_level_split": False,
            "no_cross_split_lineage": "NOT_APPLICABLE_NO_SPLIT",
            "training_splits_created": False,
        },
        "independent_source_session_count": len(sessions),
        "required_independent_source_session_count": config[
            "minimum_independent_source_sessions"
        ],
        "accepted_dataset_count": len(inventory_files),
        "resolved_dataset_count": manifest["resolved_dataset_count"],
        "unresolved_dataset_count": len(unresolved),
        "unresolved_datasets": unresolved,
        "source_sessions": list(sessions),
        "train_validation_test_created": False,
        "reason": (
            "Insufficient independent source sessions. The accepted corpus "
            "contains derivatives sharing a recording lineage. No split "
            "was created and the no-leakage rule was not weakened."
        ),
        "required_next_action": (
            "Collect independently recorded source sessions, then rerun the "
            "lineage audit and group-level split."
        ),
    }
    return manifest, audit


def main() -> int:
    config = load_json(CONFIG_PATH)
    lineage_path = resolve_path(config["lineage_manifest"])
    acceptance_output = resolve_path(config["acceptance_output"])
    try:
        manifest, audit = build_lineage()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        audit = {
            "milestone": "2.12",
            "name": "Training-Ready Dataset Packaging",
            "status": "BLOCKED",
            "classification": "INPUT_OR_LINEAGE_AUDIT_BLOCKED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "lineage_audit_complete": False,
                "training_splits_created": False,
            },
            "train_validation_test_created": False,
            "reason": str(exc),
            "required_next_action": (
                "Resolve the reported prerequisite, then rerun the lineage audit."
            ),
        }
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
        acceptance_output.parent.mkdir(parents=True, exist_ok=True)
        acceptance_output.write_text(
            json.dumps(audit, indent=2),
            encoding="utf-8",
        )
        print(f"2.12 BLOCKED: {exc}")
        return 2

    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_output.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    acceptance_output.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print("TRAINING DATASET BUILD : BLOCKED")
    print(f"Accepted datasets             : {audit['accepted_dataset_count']}")
    print(f"Independent source sessions   : {audit['independent_source_session_count']}")
    print("Train/validation/test created : NO")
    print(f"Lineage manifest              : {lineage_path.relative_to(ROOT)}")
    print(f"Acceptance report             : {acceptance_output.relative_to(ROOT)}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
