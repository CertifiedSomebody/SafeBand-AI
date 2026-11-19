import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STALL_THRESHOLD_MS = 1000.0


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validate_schema(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    return list(source.columns) == list(
        augmented.columns
    )


def validate_rows(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    return len(source) == len(augmented)


def validate_sample_index(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    return np.array_equal(
        source["sample_index"].to_numpy(),
        augmented["sample_index"].to_numpy(),
    )


def validate_elapsed_monotonic(
    augmented: pd.DataFrame,
) -> bool:

    elapsed = pd.to_numeric(
        augmented["elapsed_ms"],
        errors="coerce",
    ).to_numpy()

    if not np.isfinite(elapsed).all():
        return False

    if len(elapsed) <= 1:
        return True

    return bool(
        np.all(np.diff(elapsed) > 0.0)
    )


def calculate_intervals(
    dataframe: pd.DataFrame,
) -> np.ndarray:

    elapsed = pd.to_numeric(
        dataframe["elapsed_ms"],
        errors="coerce",
    ).to_numpy(dtype=float)

    if len(elapsed) <= 1:
        return np.array([], dtype=float)

    return np.diff(elapsed)


def validate_timing_effect(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> tuple[bool, dict]:

    source_intervals = calculate_intervals(
        source
    )

    augmented_intervals = calculate_intervals(
        augmented
    )

    if len(source_intervals) != len(
        augmented_intervals
    ):
        return False, {}

    if len(augmented_intervals) == 0:
        return False, {}

    changed = np.abs(
        source_intervals
        - augmented_intervals
    )

    changed_count = int(
        np.count_nonzero(changed > 1e-6)
    )

    stalls = (
        augmented_intervals
        > STALL_THRESHOLD_MS
    )

    stats = {
        "source_mean_interval_ms": float(
            np.mean(source_intervals)
        ),
        "augmented_mean_interval_ms": float(
            np.mean(augmented_intervals)
        ),
        "source_median_interval_ms": float(
            np.median(source_intervals)
        ),
        "augmented_median_interval_ms": float(
            np.median(augmented_intervals)
        ),
        "augmented_p95_interval_ms": float(
            np.percentile(
                augmented_intervals,
                95,
            )
        ),
        "augmented_max_interval_ms": float(
            np.max(augmented_intervals)
        ),
        "changed_intervals": changed_count,
        "total_intervals": int(
            len(augmented_intervals)
        ),
        "stall_count": int(
            stalls.sum()
        ),
        "stall_fraction": float(
            stalls.mean()
        ),
    }

    effect_pass = changed_count > 0

    return effect_pass, stats


def validate_sensor_unchanged(
    source: pd.DataFrame,
    augmented: pd.DataFrame,
) -> bool:

    excluded = {
        "elapsed_ms",
    }

    for column in source.columns:

        if column in excluded:
            continue

        source_values = source[column]
        augmented_values = augmented[column]

        if source_values.dtype.kind in "biufc":
            if not np.array_equal(
                source_values.to_numpy(),
                augmented_values.to_numpy(),
                equal_nan=True,
            ):
                return False

        else:
            if not source_values.equals(
                augmented_values
            ):
                return False

    return True


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate SafeBand timing/acquisition "
            "diversity."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--augmented",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    augmented_path = Path(args.augmented)
    report_path = Path(args.output)

    source = load_csv(source_path)
    augmented = load_csv(augmented_path)

    schema_pass = validate_schema(
        source,
        augmented,
    )

    rows_pass = validate_rows(
        source,
        augmented,
    )

    index_pass = validate_sample_index(
        source,
        augmented,
    )

    monotonic_pass = validate_elapsed_monotonic(
        augmented
    )

    timing_pass, timing_stats = (
        validate_timing_effect(
            source,
            augmented,
        )
    )

    sensor_unchanged_pass = (
        validate_sensor_unchanged(
            source,
            augmented,
        )
    )

    checks = {
        "schema": schema_pass,
        "rows": rows_pass,
        "sample_index": index_pass,
        "elapsed_monotonic": monotonic_pass,
        "timing_effect": timing_pass,
        "sensor_values_unchanged": (
            sensor_unchanged_pass
        ),
    }

    overall = all(checks.values())

    report = {
        "stage": "2.10.5",
        "status": "PASS"
        if overall
        else "FAIL",
        "source": str(source_path),
        "augmented": str(augmented_path),
        "checks": checks,
        "timing": timing_stats,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
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
        "SAFEband MILESTONE 2.10.5 — "
        "TIMING/ACQUISITION VALIDATION"
    )
    print()

    print(
        f"Schema              : "
        f"{'PASS' if schema_pass else 'FAIL'}"
    )

    print(
        f"Rows                : "
        f"{'PASS' if rows_pass else 'FAIL'}"
    )

    print(
        f"Sample index        : "
        f"{'PASS' if index_pass else 'FAIL'}"
    )

    print(
        f"Elapsed monotonic   : "
        f"{'PASS' if monotonic_pass else 'FAIL'}"
    )

    print(
        f"Timing effect       : "
        f"{'PASS' if timing_pass else 'FAIL'}"
    )

    print(
        f"Sensor values       : "
        f"{'PASS' if sensor_unchanged_pass else 'FAIL'}"
    )

    if timing_stats:
        print(
            f"Changed intervals   : "
            f"{timing_stats['changed_intervals']}"
        )

        print(
            f"Stalls              : "
            f"{timing_stats['stall_count']}"
        )

        print(
            f"Median interval     : "
            f"{timing_stats['augmented_median_interval_ms']:.2f} ms"
        )

        print(
            f"P95 interval        : "
            f"{timing_stats['augmented_p95_interval_ms']:.2f} ms"
        )

        print(
            f"Maximum interval    : "
            f"{timing_stats['augmented_max_interval_ms']:.2f} ms"
        )

    print(
        f"Overall status      : "
        f"{'PASS' if overall else 'FAIL'}"
    )

    print(
        f"Report              : {report_path}"
    )

    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
