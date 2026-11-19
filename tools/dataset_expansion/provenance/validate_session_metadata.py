from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

SCHEMA_PATH = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "provenance"
    / "session_metadata_schema.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def load_schema() -> dict:
    return load_json(
        SCHEMA_PATH
    )


def validate_manifest(
    manifest_path: Path,
) -> dict:
    manifest = load_json(
        manifest_path
    )

    schema = load_schema()

    required_fields = schema.get(
        "required_fields",
        [],
    )

    allowed_status = set(
        schema.get(
            "synthetic_status_values",
            [],
        )
    )

    checks = {}

    checks["session_schema_version"] = (
        manifest.get(
            "session_schema_version"
        )
        == schema.get(
            "schema_version",
            "1.0",
        )
    )

    datasets = manifest.get(
        "datasets"
    )

    checks["datasets_array"] = isinstance(
        datasets,
        list,
    )

    if not isinstance(
        datasets,
        list,
    ):
        datasets = []

    checks["dataset_count"] = (
        manifest.get(
            "generated_dataset_count"
        )
        == len(datasets)
    )

    all_required = True
    valid_status = True
    valid_counts = True
    unique_ids = True
    valid_provenance = True

    ids = []

    for entry in datasets:
        if not isinstance(
            entry,
            dict,
        ):
            all_required = False
            continue

        for field in required_fields:
            if field not in entry:
                all_required = False

        dataset_id = entry.get(
            "dataset_id"
        )

        if dataset_id in ids:
            unique_ids = False

        ids.append(dataset_id)

        status = entry.get(
            "synthetic_status"
        )

        if status not in allowed_status:
            valid_status = False

        row_count = entry.get(
            "row_count"
        )

        column_count = entry.get(
            "column_count"
        )

        if (
            not isinstance(
                row_count,
                int,
            )
            or row_count <= 0
            or not isinstance(
                column_count,
                int,
            )
            or column_count <= 0
        ):
            valid_counts = False

        provenance_file = entry.get(
            "provenance_file"
        )

        if provenance_file is not None:
            provenance_path = (
                ROOT / provenance_file
            )

            if not provenance_path.exists():
                valid_provenance = False

    checks["required_fields"] = (
        all_required
    )

    checks["synthetic_status"] = (
        valid_status
    )

    checks["dataset_counts"] = (
        valid_counts
    )

    checks["unique_dataset_ids"] = (
        unique_ids
    )

    checks["provenance_files"] = (
        valid_provenance
    )

    overall = all(
        checks.values()
    )

    return {
        "schema_version": schema.get(
            "schema_version",
            "1.0",
        ),
        "manifest": str(
            manifest_path.relative_to(
                ROOT
            )
        ),
        "dataset_count": len(
            datasets
        ),
        "checks": checks,
        "overall_status": (
            "PASS"
            if overall
            else "FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = (
        ROOT / args.manifest
    )

    output_path = (
        ROOT / args.output
    )

    result = validate_manifest(
        manifest_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print()
    print(
        "SAFEband MILESTONE 2.10.8 "
        "— SESSION METADATA VALIDATION"
    )
    print()

    for name, passed in (
        result["checks"].items()
    ):
        print(
            f"{name.replace('_', ' ').title():30}"
            f": "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print(
        f"Datasets                       : "
        f"{result['dataset_count']}"
    )

    print(
        f"Overall status                 : "
        f"{result['overall_status']}"
    )

    print(
        f"Report                         : "
        f"{args.output}"
    )

    if (
        result["overall_status"]
        != "PASS"
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
