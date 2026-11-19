from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "datasets" / "generated"

REPORT_DIR = (
    ROOT
    / "reports"
    / "dataset_expansion"
    / "acceptance"
)

REPORT_PATH = (
    REPORT_DIR
    / "sentinel_validity_2.11.6.json"
)

SENTINEL = -999.0

VALIDITY_COLUMNS = [
    "bme_valid",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
]

MEASUREMENT_VALIDITY_MAP = {
    "raw_hr": "raw_hr_valid",
    "raw_spo2": "raw_spo2_valid",
    "final_hr": "final_hr_valid",
    "final_spo2": "final_spo2_valid",
}


def is_close_to_sentinel(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isclose(
        values.to_numpy(dtype=float),
        SENTINEL,
        atol=1e-9,
        rtol=0.0,
    )


def validate_validity_flags(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []

    for column in VALIDITY_COLUMNS:
        if column not in df.columns:
            errors.append(
                f"missing validity column: {column}"
            )
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            errors.append(
                f"{column}: contains non-numeric/NaN values"
            )
            continue

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            errors.append(
                f"{column}: contains non-finite values"
            )

        invalid_values = ~values.isin([0, 1])

        if invalid_values.any():
            errors.append(
                f"{column}: contains values other than 0/1"
            )

    return len(errors) == 0, errors


def validate_measurement_sentinels(
    df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    errors = []

    for measurement, validity in MEASUREMENT_VALIDITY_MAP.items():
        if measurement not in df.columns:
            errors.append(
                f"missing measurement column: {measurement}"
            )
            continue

        if validity not in df.columns:
            errors.append(
                f"missing validity column: {validity}"
            )
            continue

        values = pd.to_numeric(
            df[measurement],
            errors="coerce",
        )

        flags = pd.to_numeric(
            df[validity],
            errors="coerce",
        )

        valid_rows = flags == 1

        if valid_rows.any():
            valid_values = values[valid_rows]

            if valid_values.isna().any():
                errors.append(
                    f"{measurement}: valid rows contain NaN"
                )

            if is_close_to_sentinel(valid_values).any():
                errors.append(
                    f"{measurement}: valid rows contain "
                    f"sentinel {SENTINEL}"
                )

        sentinel_rows = is_close_to_sentinel(values)

        if sentinel_rows.any():
            sentinel_flags = flags[sentinel_rows]

            if not (sentinel_flags == 0).all():
                errors.append(
                    f"{measurement}: sentinel values "
                    "must have validity flag 0"
                )

    return len(errors) == 0, errors


def validate_source_sentinel_preservation(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> tuple[bool, list[str]]:
    errors = []

    for column in source.columns:
        if column not in augmented.columns:
            errors.append(
                f"source column missing from output: {column}"
            )
            continue

        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        )

        augmented_values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        source_sentinel = is_close_to_sentinel(
            source_values
        )

        if not source_sentinel.any():
            continue

        output_at_source_sentinel = (
            augmented_values[source_sentinel]
        )

        if not is_close_to_sentinel(
            output_at_source_sentinel
        ).all():
            errors.append(
                f"{column}: source sentinel values "
                "were not preserved"
            )

    return len(errors) == 0, errors


def find_source_dataset(
    dataset_path: Path,
) -> Path | None:
    provenance_candidates = [
        dataset_path.with_suffix(
            dataset_path.suffix + ".provenance.json"
        ),
        dataset_path.with_suffix(
            ".provenance.json"
        ),
    ]

    for provenance_path in provenance_candidates:
        if not provenance_path.exists():
            continue

        try:
            with provenance_path.open(
                "r",
                encoding="utf-8-sig",
            ) as handle:
                provenance = json.load(handle)
        except Exception:
            continue

        source = provenance.get("source")

        if not source:
            continue

        source_path = ROOT / Path(source)

        if source_path.exists():
            return source_path

    return None


def validate_dataset(
    dataset_path: Path,
) -> dict:
    result = {
        "dataset": dataset_path.name,
        "validity_flags": "PASS",
        "measurement_sentinels": "PASS",
        "source_sentinel_preservation": "NOT_APPLICABLE",
        "status": "PASS",
        "errors": [],
    }

    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        result["status"] = "FAIL"
        result["errors"].append(
            f"read failure: {exc}"
        )
        return result

    flags_ok, flag_errors = validate_validity_flags(df)

    if not flags_ok:
        result["validity_flags"] = "FAIL"
        result["errors"].extend(flag_errors)

    measurements_ok, measurement_errors = (
        validate_measurement_sentinels(df)
    )

    if not measurements_ok:
        result["measurement_sentinels"] = "FAIL"
        result["errors"].extend(measurement_errors)

    source_path = find_source_dataset(dataset_path)

    if source_path is not None:
        try:
            source_df = pd.read_csv(source_path)

            source_ok, source_errors = (
                validate_source_sentinel_preservation(
                    source_df,
                    df,
                )
            )

            result["source_sentinel_preservation"] = (
                "PASS" if source_ok else "FAIL"
            )

            if not source_ok:
                result["errors"].extend(source_errors)

        except Exception as exc:
            result["source_sentinel_preservation"] = "FAIL"
            result["errors"].append(
                f"source validation failure: {exc}"
            )

    if result["errors"]:
        result["status"] = "FAIL"

    return result


def main() -> int:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = sorted(
        DATASET_DIR.glob("*.csv")
    )

    results = [
        validate_dataset(path)
        for path in datasets
    ]

    datasets_pass = sum(
        result["status"] == "PASS"
        for result in results
    )

    overall_pass = (
        len(datasets) > 0
        and datasets_pass == len(datasets)
    )

    report = {
        "milestone": "2.11.6",
        "name": "Sentinel/Validity Integrity",
        "dataset_count": len(datasets),
        "datasets_pass": datasets_pass,
        "datasets_fail": (
            len(datasets) - datasets_pass
        ),
        "sentinel": SENTINEL,
        "validity_columns": VALIDITY_COLUMNS,
        "measurement_validity_map": (
            MEASUREMENT_VALIDITY_MAP
        ),
        "overall_status": (
            "PASS" if overall_pass else "FAIL"
        ),
        "datasets": results,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        "SAFEband MILESTONE 2.11.6 — "
        "SENTINEL/VALIDITY INTEGRITY"
    )
    print()
    print(
        f"Dataset count            : "
        f"{'PASS' if len(datasets) > 0 else 'FAIL'}"
    )
    print(
        f"Validity flags           : "
        f"{'PASS' if all(r['validity_flags'] == 'PASS' for r in results) else 'FAIL'}"
    )
    print(
        f"Measurement sentinels    : "
        f"{'PASS' if all(r['measurement_sentinels'] == 'PASS' for r in results) else 'FAIL'}"
    )
    print(
        f"Source preservation      : "
        f"{'PASS' if all(r['source_sentinel_preservation'] in ('PASS', 'NOT_APPLICABLE') for r in results) else 'FAIL'}"
    )
    print(
        f"Datasets PASS            : "
        f"{datasets_pass}"
    )
    print(
        f"Datasets FAIL            : "
        f"{len(datasets) - datasets_pass}"
    )
    print()
    print(
        f"Overall status           : "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )
    print(
        f"Report                   : "
        f"{REPORT_PATH.relative_to(ROOT)}"
    )

    if not overall_pass:
        print()
        print("FAILURES:")
        for result in results:
            if result["status"] != "PASS":
                print(
                    f"  {result['dataset']}"
                )
                for error in result["errors"]:
                    print(
                        f"    - {error}"
                    )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
