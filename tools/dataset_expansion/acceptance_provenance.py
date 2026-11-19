from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

GENERATED_DIR = ROOT / "datasets" / "generated"

REPORT_DIR = ROOT / "reports" / "dataset_expansion"

PROVENANCE_DIR = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "provenance"
)

SESSION_MANIFEST = (
    PROVENANCE_DIR
    / "dataset_expansion_session_2.10.8.json"
)

OUTPUT_REPORT = (
    REPORT_DIR
    / "acceptance"
    / "provenance_completeness_2.11.3.json"
)

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


def normalize_path(path: str) -> Path:
    return ROOT / path.replace("\\", "/")


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


def native_structural_evidence(
    native: dict,
) -> bool:
    evidence_fields = [
        "schema_preserved",
        "source_rows",
        "output_rows",
        "source_columns",
        "output_columns",
        "rows",
        "columns",
        "sample_index_preserved",
        "sensor_values_modified",
        "validated_against_real_running",
        "generation_model",
        "limitations",
    ]

    return any(
        field in native
        for field in evidence_fields
    )


def validate_dataset(
    dataset_path: Path,
    manifest_entry: dict,
    allowed_status: set[str],
) -> tuple[bool, list[str]]:

    failures: list[str] = []

    dataset_id = dataset_path.stem

    if manifest_entry.get(
        "dataset_id"
    ) != dataset_id:
        failures.append(
            "Manifest dataset_id mismatch"
        )

    manifest_stage = manifest_entry.get(
        "generation_stage"
    )

    manifest_profile = manifest_entry.get(
        "profile"
    )

    manifest_seed = manifest_entry.get(
        "seed"
    )

    manifest_status = manifest_entry.get(
        "synthetic_status"
    )

    manifest_report = manifest_entry.get(
        "validation_report"
    )

    if (
        manifest_stage is None
        or manifest_stage == "UNKNOWN"
        or manifest_profile is None
        or manifest_profile == "BASE_DATASET"
        or manifest_seed is None
    ):
        failures.append(
            "Incomplete generation metadata"
        )

    if manifest_status not in allowed_status:
        failures.append(
            "Invalid synthetic_status: "
            f"{manifest_status!r}"
        )

    if manifest_report is not None:
        report_path = normalize_path(
            manifest_report
        )

        if not report_path.exists():
            failures.append(
                "Referenced validation report "
                f"does not exist: "
                f"{manifest_report}"
            )

    native_path = find_native_provenance(
        dataset_path
    )

    native = {}

    if native_path is not None:
        try:
            native = load_json(
                native_path
            )
        except Exception as exc:
            failures.append(
                "Native provenance JSON invalid: "
                f"{exc}"
            )

    if native:

        native_stage = native.get(
            "stage"
        )

        if (
            native_stage is not None
            and manifest_stage not in (
                None,
                "UNKNOWN",
            )
            and str(native_stage)
            != str(manifest_stage)
        ):
            failures.append(
                "Native provenance generation_stage "
                f"mismatch: {native_stage!r} "
                f"!= {manifest_stage!r}"
            )

        native_status = native.get(
            "status"
        )

        if (
            native_status is not None
            and manifest_status is not None
            and str(native_status)
            != str(manifest_status)
        ):
            failures.append(
                "Native provenance synthetic_status "
                f"mismatch: {native_status!r} "
                f"!= {manifest_status!r}"
            )

        native_profile = native.get(
            "profile"
        )

        if isinstance(
            native_profile,
            str,
        ):
            if (
                manifest_profile not in (
                    None,
                    "UNKNOWN",
                    "BASE_DATASET",
                )
                and native_profile
                != manifest_profile
            ):
                failures.append(
                    "Native provenance profile mismatch: "
                    f"{native_profile!r} "
                    f"!= {manifest_profile!r}"
                )

        elif isinstance(
            native_profile,
            dict,
        ):
            manifest_parameters = (
                manifest_entry.get(
                    "augmentation_parameters"
                )
            )

            # The normalized 2.10.8 manifest stores
            # the semantic profile name separately.
            # Native generators may store the actual
            # parameter dictionary under "profile".
            #
            # An empty normalized parameter dictionary
            # means there is no canonical parameter set
            # to compare against.
            if manifest_parameters:
                if (
                    native_profile
                    != manifest_parameters
                ):
                    failures.append(
                        "Native provenance profile "
                        "parameters mismatch"
                    )

        native_seed = native.get(
            "seed"
        )

        if native_seed is None:
            native_seed = native.get(
                "random_seed"
            )

        if (
            native_seed is not None
            and manifest_seed is not None
        ):
            try:
                if int(native_seed) != int(
                    manifest_seed
                ):
                    failures.append(
                        "Native provenance seed "
                        f"mismatch: {native_seed!r} "
                        f"!= {manifest_seed!r}"
                    )
            except (
                TypeError,
                ValueError,
            ):
                if str(native_seed) != str(
                    manifest_seed
                ):
                    failures.append(
                        "Native provenance seed "
                        f"mismatch: {native_seed!r} "
                        f"!= {manifest_seed!r}"
                    )

        native_rows = native.get(
            "output_rows"
        )

        if native_rows is None:
            native_rows = native.get(
                "rows"
            )

        if native_rows is None:
            native_rows = native.get(
                "source_rows"
            )

        if native_rows is not None:
            try:
                if int(native_rows) != int(
                    manifest_entry.get(
                        "row_count"
                    )
                ):
                    failures.append(
                        "Native provenance row count "
                        f"mismatch: {native_rows!r} "
                        f"!= "
                        f"{manifest_entry.get('row_count')!r}"
                    )
            except (
                TypeError,
                ValueError,
            ):
                failures.append(
                    "Native provenance row count "
                    "is not numeric"
                )

        native_columns = native.get(
            "output_columns"
        )

        if native_columns is None:
            native_columns = native.get(
                "columns"
            )

        if native_columns is None:
            native_columns = native.get(
                "source_columns"
            )

        if native_columns is not None:
            try:
                if int(native_columns) != int(
                    manifest_entry.get(
                        "column_count"
                    )
                ):
                    failures.append(
                        "Native provenance column count "
                        f"mismatch: {native_columns!r} "
                        f"!= "
                        f"{manifest_entry.get('column_count')!r}"
                    )
            except (
                TypeError,
                ValueError,
            ):
                failures.append(
                    "Native provenance column count "
                    "is not numeric"
                )

        if (
            manifest_report is None
            and not native_structural_evidence(
                native
            )
        ):
            failures.append(
                "No validation report or native "
                "structural validation evidence"
            )

    elif manifest_report is None:
        failures.append(
            "No native provenance or validation "
            "report available"
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

    allowed_status = set(
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
        path
        for path in GENERATED_DIR.glob(
            "*.csv"
        )
        if not path.name.startswith(
            "~$"
        )
    )

    results = []

    native_count = 0
    pass_count = 0
    fail_count = 0

    for dataset_path in dataset_paths:

        dataset_id = dataset_path.stem

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
                    allowed_status,
                )
            )

        if (
            find_native_provenance(
                dataset_path
            )
            is not None
        ):
            native_count += 1

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
                "failures": failures,
            }
        )

    overall = (
        fail_count == 0
        and len(dataset_paths)
        == len(entries)
    )

    report = {
        "milestone": "2.11.3",
        "name": (
            "Provenance Completeness"
        ),
        "session_manifest": str(
            SESSION_MANIFEST.relative_to(
                ROOT
            )
        ),
        "datasets_checked": len(
            dataset_paths
        ),
        "manifest_entries": len(
            entries
        ),
        "native_provenance_files": (
            native_count
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
        "SAFEband MILESTONE 2.11.3 "
        "— PROVENANCE COMPLETENESS"
    )
    print()

    print(
        f"Datasets checked           : "
        f"{len(dataset_paths)}"
    )

    print(
        f"Manifest entries           : "
        f"{len(entries)}"
    )

    print(
        f"Native provenance files    : "
        f"{native_count}"
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
