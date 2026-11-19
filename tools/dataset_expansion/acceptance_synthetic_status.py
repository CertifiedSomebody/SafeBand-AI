from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

GENERATED_DIR = (
    ROOT
    / "datasets"
    / "generated"
)

PROVENANCE_DIR = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "provenance"
)

ACCEPTANCE_DIR = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "acceptance"
)

SESSION_MANIFEST = (
    PROVENANCE_DIR
    / "dataset_expansion_session_2.10.8.json"
)

SCHEMA_PATH = (
    ROOT
    / "tools"
    / "dataset_expansion"
    / "provenance"
    / "session_metadata_schema.json"
)

OUTPUT_REPORT = (
    ACCEPTANCE_DIR
    / "synthetic_status_2.11.4.json"
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def find_native_provenance(
    dataset_path: Path,
) -> Path | None:

    candidates = [
        Path(
            str(dataset_path)
            + ".provenance.json"
        ),
        dataset_path.parent
        / f"{dataset_path.stem}.provenance.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def expected_status_for_dataset(
    dataset_name: str,
) -> str | None:

    if dataset_name == (
        "running_synthetic_dataset.csv"
    ):
        return (
            "SYNTHETIC_DEVELOPMENT_FIXTURE"
        )

    if dataset_name in {
        "stationary.csv",
        "stationary_2.csv",
        "walking.csv",
    }:
        return "REAL"

    if (
        dataset_name.startswith(
            "walking_aug_"
        )
        or dataset_name.startswith(
            "motion_"
        )
        or dataset_name.startswith(
            "walking_profile_"
        )
        or dataset_name.startswith(
            "ppg_"
        )
        or dataset_name.startswith(
            "sensor_noise_"
        )
        or dataset_name.startswith(
            "timing_"
        )
        or dataset_name.startswith(
            "hardneg_"
        )
        or dataset_name.startswith(
            "hazard_"
        )
    ):
        return "SYNTHETIC_UNVALIDATED"

    return None


def validate_dataset(
    dataset_path: Path,
    manifest_entry: dict,
    allowed_statuses: set[str],
) -> tuple[bool, list[str]]:

    failures: list[str] = []

    dataset_name = dataset_path.name

    manifest_status = manifest_entry.get(
        "synthetic_status"
    )

    if manifest_status not in allowed_statuses:
        failures.append(
            "Invalid or missing manifest "
            f"synthetic_status: "
            f"{manifest_status!r}"
        )

    expected_status = (
        expected_status_for_dataset(
            dataset_name
        )
    )

    if expected_status is None:
        failures.append(
            "No expected synthetic status rule "
            f"for dataset: {dataset_name}"
        )
    elif manifest_status != expected_status:
        failures.append(
            "Unexpected manifest synthetic_status: "
            f"{manifest_status!r} "
            f"(expected {expected_status!r})"
        )

    native_path = find_native_provenance(
        dataset_path
    )

    if native_path is None:
        failures.append(
            "Native provenance file missing"
        )
    else:
        try:
            native = load_json(
                native_path
            )
        except Exception as exc:
            failures.append(
                "Native provenance JSON invalid: "
                f"{exc}"
            )
            native = {}

        if native:
            native_status = native.get(
                "status"
            )

            if native_status is not None:
                if native_status not in (
                    allowed_statuses
                ):
                    failures.append(
                        "Invalid native synthetic status: "
                        f"{native_status!r}"
                    )
                elif (
                    manifest_status is not None
                    and native_status
                    != manifest_status
                ):
                    failures.append(
                        "Manifest/native synthetic status "
                        f"mismatch: "
                        f"{manifest_status!r} "
                        f"!= {native_status!r}"
                    )

    return (
        len(failures) == 0,
        failures,
    )


def main() -> None:

    if not GENERATED_DIR.exists():
        raise RuntimeError(
            f"Generated dataset directory not found: "
            f"{GENERATED_DIR}"
        )

    manifest = load_json(
        SESSION_MANIFEST
    )

    schema = load_json(
        SCHEMA_PATH
    )

    allowed_statuses = set(
        schema.get(
            "synthetic_status_values",
            [],
        )
    )

    entries = manifest.get(
        "datasets",
        [],
    )

    manifest_by_id = {
        entry.get("dataset_id"): entry
        for entry in entries
        if isinstance(entry, dict)
    }

    dataset_paths = sorted(
        GENERATED_DIR.glob(
            "*.csv"
        )
    )

    results = []

    pass_count = 0
    fail_count = 0

    for dataset_path in dataset_paths:

        dataset_id = (
            dataset_path.stem
        )

        manifest_entry = (
            manifest_by_id.get(
                dataset_id
            )
        )

        if manifest_entry is None:
            passed = False
            failures = [
                "Dataset missing from session manifest"
            ]
        else:
            passed, failures = (
                validate_dataset(
                    dataset_path,
                    manifest_entry,
                    allowed_statuses,
                )
            )

        if passed:
            pass_count += 1
        else:
            fail_count += 1

        results.append(
            {
                "dataset": dataset_path.name,
                "dataset_id": dataset_id,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
                "manifest_status": (
                    manifest_entry.get(
                        "synthetic_status"
                    )
                    if manifest_entry
                    else None
                ),
                "expected_status": (
                    expected_status_for_dataset(
                        dataset_path.name
                    )
                ),
                "failures": failures,
            }
        )

    overall = (
        fail_count == 0
        and len(dataset_paths)
        == len(entries)
    )

    report = {
        "milestone": "2.11.4",
        "name": (
            "Synthetic-Status Integrity"
        ),
        "allowed_statuses": sorted(
            allowed_statuses
        ),
        "datasets_checked": len(
            dataset_paths
        ),
        "manifest_entries": len(
            entries
        ),
        "datasets_pass": pass_count,
        "datasets_fail": fail_count,
        "overall_status": (
            "PASS"
            if overall
            else "FAIL"
        ),
        "datasets": results,
    }

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "SAFEband MILESTONE 2.11.4 "
        "— SYNTHETIC-STATUS INTEGRITY"
    )
    print()

    print(
        "Allowed statuses           : "
        + ", ".join(
            sorted(
                allowed_statuses
            )
        )
    )

    print(
        f"Datasets checked           : "
        f"{len(dataset_paths)}"
    )

    print(
        f"Manifest entries           : "
        f"{len(entries)}"
    )

    print(
        f"Datasets PASS              : "
        f"{pass_count}"
    )

    print(
        f"Datasets FAIL              : "
        f"{fail_count}"
    )

    if fail_count:
        print()
        print("Failures:")

        for result in results:

            if result["status"] != "FAIL":
                continue

            print()
            print(
                f"  {result['dataset']}"
            )

            for failure in result[
                "failures"
            ]:
                print(
                    f"    - {failure}"
                )

    print()
    print(
        f"Overall status             : "
        f"{report['overall_status']}"
    )

    print(
        f"Report                     : "
        f"{OUTPUT_REPORT.relative_to(ROOT)}"
    )

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
