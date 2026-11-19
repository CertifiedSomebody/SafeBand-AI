from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL = -999.0


NON_SIGNAL_COLUMNS = {
    "elapsed_ms",
    "sample_index",
}


BINARY_COLUMNS = {
    "bme_valid",
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "charger_connected",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate sensor-noise diversity output."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument("--augmented", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def numeric_columns(df: pd.DataFrame):
    columns = []

    for column in df.columns:
        if column in NON_SIGNAL_COLUMNS:
            continue

        if column in BINARY_COLUMNS:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.notna().any():
            columns.append(column)

    return columns


def validate_binary_flags(df: pd.DataFrame) -> bool:
    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():
            return False

        if not values.isin([0, 1]).all():
            return False

    return True


def validate_finite_values(df: pd.DataFrame) -> bool:
    for column in numeric_columns(df):
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        usable = values != SENTINEL

        if not np.isfinite(values[usable]).all():
            return False

    return True


def validate_sentinels(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    for column in augmented.columns:

        if column not in source.columns:
            continue

        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        )

        augmented_values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        source_sentinel = (
            source_values == SENTINEL
        )

        if not source_sentinel.any():
            continue

        augmented_at_source_sentinel = (
            augmented_values[source_sentinel]
        )

        if not (
            augmented_at_source_sentinel
            == SENTINEL
        ).all():
            return False

    return True


def measure_noise_effect(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
):
    changed_columns = {}
    total_changed = 0

    for column in numeric_columns(source):

        source_values = pd.to_numeric(
            source[column],
            errors="coerce",
        )

        augmented_values = pd.to_numeric(
            augmented[column],
            errors="coerce",
        )

        comparable = (
            source_values.notna()
            & augmented_values.notna()
            & (source_values != SENTINEL)
            & (augmented_values != SENTINEL)
            & np.isfinite(source_values)
            & np.isfinite(augmented_values)
        )

        if not comparable.any():
            continue

        difference = (
            augmented_values[comparable]
            - source_values[comparable]
        )

        changed = int(
            np.count_nonzero(
                np.abs(
                    difference.to_numpy()
                ) > 1e-12
            )
        )

        changed_columns[column] = {
            "comparable_rows": int(
                comparable.sum()
            ),
            "changed_rows": changed,
            "mean_abs_delta": float(
                np.mean(
                    np.abs(
                        difference.to_numpy()
                    )
                )
            ),
            "max_abs_delta": float(
                np.max(
                    np.abs(
                        difference.to_numpy()
                    )
                )
            ),
        }

        total_changed += changed

    return (
        total_changed,
        changed_columns,
    )


def main():

    args = parse_args()

    source_path = Path(args.source)
    augmented_path = Path(args.augmented)
    output_path = Path(args.output)

    source = pd.read_csv(source_path)
    augmented = pd.read_csv(augmented_path)

    checks = {}

    checks["schema"] = (
        list(source.columns)
        == list(augmented.columns)
    )

    checks["rows"] = (
        len(source)
        == len(augmented)
    )

    checks["nonfinite"] = (
        validate_finite_values(
            augmented
        )
    )

    checks["validity_flags"] = (
        validate_binary_flags(
            augmented
        )
    )

    checks["sentinels"] = (
        validate_sentinels(
            source,
            augmented,
        )
    )

    total_changed, changed_columns = (
        measure_noise_effect(
            source,
            augmented,
        )
    )

    checks["noise_effect"] = (
        total_changed > 0
    )

    overall = all(
        checks.values()
    )

    report = {
        "stage": "2.10.4",
        "checks": checks,
        "total_changed_values": total_changed,
        "changed_columns": changed_columns,
        "overall_status": (
            "PASS"
            if overall
            else "FAIL"
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print()
    print(
        "SAFEband MILESTONE 2.10.4 — "
        "SENSOR NOISE VALIDATION"
    )
    print()

    print(
        f"Schema          : "
        f"{'PASS' if checks['schema'] else 'FAIL'}"
    )

    print(
        f"Rows            : "
        f"{'PASS' if checks['rows'] else 'FAIL'}"
    )

    print(
        f"Non-finite      : "
        f"{'PASS' if checks['nonfinite'] else 'FAIL'}"
    )

    print(
        f"Validity flags  : "
        f"{'PASS' if checks['validity_flags'] else 'FAIL'}"
    )

    print(
        f"Sentinels       : "
        f"{'PASS' if checks['sentinels'] else 'FAIL'}"
    )

    print(
        f"Noise effect    : "
        f"{'PASS' if checks['noise_effect'] else 'FAIL'}"
    )

    print(
        f"Changed values  : "
        f"{total_changed}"
    )

    print(
        f"Overall status  : "
        f"{'PASS' if overall else 'FAIL'}"
    )

    print(
        f"Report          : "
        f"{output_path}"
    )

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
