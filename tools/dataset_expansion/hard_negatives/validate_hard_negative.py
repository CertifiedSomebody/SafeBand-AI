from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_COLUMNS = {
    "sample_index",
}

VALIDITY_COLUMNS = {
    "bme_valid",
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "charger_connected",
}

LABEL_COLUMNS = {
    "activity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate label-preserving hard-negative generation."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument("--augmented", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def finite_numeric(
    series: pd.Series,
) -> np.ndarray:

    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)


def compare_equal(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
    columns: list[str],
) -> bool:

    for column in columns:

        if column not in source.columns:
            continue

        if column not in augmented.columns:
            return False

        left = source[column]
        right = augmented[column]

        if pd.api.types.is_numeric_dtype(left):

            left_values = finite_numeric(left)
            right_values = finite_numeric(right)

            if not np.allclose(
                left_values,
                right_values,
                equal_nan=True,
            ):
                return False

        else:

            if not left.astype(str).equals(
                right.astype(str)
            ):
                return False

    return True


def sentinel_positions(
    series: pd.Series,
) -> np.ndarray:

    values = finite_numeric(
        series
    )

    return (
        np.isfinite(values)
        & (values == -999.0)
    )


def main() -> None:

    args = parse_args()

    source = pd.read_csv(
        args.source
    )

    augmented = pd.read_csv(
        args.augmented
    )

    results = {}

    results["schema"] = (
        list(source.columns)
        == list(augmented.columns)
    )

    results["rows"] = (
        len(source)
        == len(augmented)
    )

    results["sample_index"] = (
        compare_equal(
            source,
            augmented,
            ["sample_index"],
        )
    )

    source_activity = (
        source["activity"]
        .astype(str)
        .tolist()
    )

    augmented_activity = (
        augmented["activity"]
        .astype(str)
        .tolist()
    )

    results["activity_preserved"] = (
        source_activity
        == augmented_activity
    )

    validity_present = [
        column
        for column in VALIDITY_COLUMNS
        if column in source.columns
    ]

    results["validity_flags"] = (
        compare_equal(
            source,
            augmented,
            validity_present,
        )
    )

    label_present = [
        column
        for column in LABEL_COLUMNS
        if column in source.columns
    ]

    results["labels_preserved"] = (
        compare_equal(
            source,
            augmented,
            label_present,
        )
    )

    nonfinite = False

    for column in augmented.columns:

        values = finite_numeric(
            augmented[column]
        )

        if (
            pd.api.types.is_numeric_dtype(
                augmented[column]
            )
            and not np.isfinite(values).all()
        ):
            nonfinite = True
            break

    results["nonfinite"] = not nonfinite

    sentinel_ok = True

    for column in source.columns:

        if column not in augmented.columns:
            sentinel_ok = False
            break

        source_mask = sentinel_positions(
            source[column]
        )

        augmented_mask = sentinel_positions(
            augmented[column]
        )

        if np.any(
            source_mask
            & ~augmented_mask
        ):
            sentinel_ok = False
            break

    results["sentinels_preserved"] = (
        sentinel_ok
    )

    if "elapsed_ms" in augmented.columns:

        elapsed = finite_numeric(
            augmented["elapsed_ms"]
        )

        results["elapsed_monotonic"] = bool(
            np.all(
                np.diff(elapsed)
                >= 0
            )
        )

    else:
        results["elapsed_monotonic"] = False

    ignored_columns = (
        INDEX_COLUMNS
        | VALIDITY_COLUMNS
        | LABEL_COLUMNS
    )

    changed_columns = []
    changed_values = 0

    for column in source.columns:

        if column in ignored_columns:
            continue

        left = source[column]
        right = augmented[column]

        if not (
            pd.api.types.is_numeric_dtype(left)
            and pd.api.types.is_numeric_dtype(right)
        ):
            continue

        left_values = finite_numeric(
            left
        )
        right_values = finite_numeric(
            right
        )

        changed_mask = (
            left_values
            != right_values
        )

        count = int(
            np.sum(changed_mask)
        )

        if count > 0:
            changed_columns.append(
                column
            )
            changed_values += count

    results["hard_negative_effect"] = (
        changed_values > 0
    )

    results["changed_values"] = (
        changed_values
    )

    results["changed_columns"] = (
        changed_columns
    )

    required_checks = [
        "schema",
        "rows",
        "sample_index",
        "activity_preserved",
        "validity_flags",
        "labels_preserved",
        "nonfinite",
        "sentinels_preserved",
        "elapsed_monotonic",
        "hard_negative_effect",
    ]

    overall = all(
        bool(results[name])
        for name in required_checks
    )

    results["overall"] = overall

    output_path = Path(
        args.output
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
            results,
            handle,
            indent=2,
        )

    print()
    print(
        "SAFEband MILESTONE 2.10.6 "
        "— HARD-NEGATIVE VALIDATION"
    )
    print()
    print(
        f"Schema              : "
        f"{'PASS' if results['schema'] else 'FAIL'}"
    )
    print(
        f"Rows                : "
        f"{'PASS' if results['rows'] else 'FAIL'}"
    )
    print(
        f"Sample index        : "
        f"{'PASS' if results['sample_index'] else 'FAIL'}"
    )
    print(
        f"Activity preserved  : "
        f"{'PASS' if results['activity_preserved'] else 'FAIL'}"
    )
    print(
        f"Validity flags      : "
        f"{'PASS' if results['validity_flags'] else 'FAIL'}"
    )
    print(
        f"Labels preserved    : "
        f"{'PASS' if results['labels_preserved'] else 'FAIL'}"
    )
    print(
        f"Non-finite          : "
        f"{'PASS' if results['nonfinite'] else 'FAIL'}"
    )
    print(
        f"Sentinels           : "
        f"{'PASS' if results['sentinels_preserved'] else 'FAIL'}"
    )
    print(
        f"Elapsed monotonic   : "
        f"{'PASS' if results['elapsed_monotonic'] else 'FAIL'}"
    )
    print(
        f"Hard-negative effect: "
        f"{'PASS' if results['hard_negative_effect'] else 'FAIL'}"
    )
    print(
        f"Changed values      : "
        f"{results['changed_values']}"
    )
    print(
        f"Overall status      : "
        f"{'PASS' if overall else 'FAIL'}"
    )
    print(
        f"Report              : {output_path}"
    )


if __name__ == "__main__":
    main()
