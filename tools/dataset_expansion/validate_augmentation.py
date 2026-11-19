from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fail(errors, message):
    errors.append(message)


def finite_numeric_rows(df):
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return 0
    return int((~np.isfinite(numeric.to_numpy(dtype=float))).any(axis=1).sum())


def timing_stats(df):
    intervals = np.diff(df["elapsed_ms"].astype(float).to_numpy())
    if len(intervals) == 0:
        return {"count": 0, "mean_ms": None, "median_ms": None, "stall_count": 0, "stall_fraction": 0.0}
    stalls = intervals > 1000
    return {
        "count": int(len(intervals)),
        "mean_ms": float(intervals.mean()),
        "median_ms": float(np.median(intervals)),
        "stall_count": int(stalls.sum()),
        "stall_fraction": float(stalls.mean()),
        "min_ms": float(intervals.min()),
        "max_ms": float(intervals.max()),
    }


def sentinel_counts(df):
    result = {}
    for col in df.columns:
        count = int((df[col] == -999).sum())
        if count:
            result[col] = count
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--augmented", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    src = pd.read_csv(args.source)
    aug = pd.read_csv(args.augmented)

    errors = []
    warnings = []

    source_columns = list(src.columns)
    augmented_columns = list(aug.columns)

    if source_columns != augmented_columns:
        fail(errors, "Schema/order mismatch.")

    if len(src) != len(aug):
        fail(errors, f"Row count changed: source={len(src)}, augmented={len(aug)}.")

    if "activity" in src.columns and "activity" in aug.columns:
        if not src["activity"].astype(str).equals(aug["activity"].astype(str)):
            fail(errors, "Activity labels changed.")

    duplicate_count = int(aug.duplicated().sum())
    if duplicate_count:
        fail(errors, f"Augmented dataset contains {duplicate_count} duplicate rows.")

    nonfinite_rows = finite_numeric_rows(aug)
    if nonfinite_rows:
        fail(errors, f"Augmented dataset contains {nonfinite_rows} rows with non-finite numeric values.")

    src_sentinels = sentinel_counts(src)
    aug_sentinels = sentinel_counts(aug)
    if src_sentinels != aug_sentinels:
        fail(errors, f"Sentinel counts changed: source={src_sentinels}, augmented={aug_sentinels}.")

    source_timing = timing_stats(src)
    augmented_timing = timing_stats(aug)

    if source_timing["stall_count"] != augmented_timing["stall_count"]:
        fail(errors, "Timing stall count changed.")

    if source_timing["stall_fraction"] != augmented_timing["stall_fraction"]:
        fail(errors, "Timing stall fraction changed.")

    if "sample_index" in aug.columns:
        expected = np.arange(len(aug))
        if not np.array_equal(aug["sample_index"].to_numpy(), expected):
            fail(errors, "sample_index is not contiguous after augmentation.")

    # Basic physical sanity bounds only. These are guardrails, not model labels.
    if "humidity_pct" in aug.columns:
        if ((aug["humidity_pct"] < 0) | (aug["humidity_pct"] > 100)).any():
            fail(errors, "Humidity outside [0, 100].")

    for col, low, high in [
        ("raw_hr", 25, 240),
        ("final_hr", 25, 240),
        ("raw_spo2", 50, 100),
        ("final_spo2", 50, 100),
    ]:
        if col in aug.columns:
            values = aug[col].to_numpy(dtype=float)
            mask = (values != -999) & np.isfinite(values)
            if mask.any() and ((values[mask] < low) | (values[mask] > high)).any():
                fail(errors, f"{col} outside guardrail range [{low}, {high}].")

    # Distribution sanity: generic augmentation should not wildly alter the
    # median of core motion magnitudes.
    for name, columns in {
        "acceleration": ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"],
        "gyro": ["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"],
    }.items():
        if all(c in src.columns for c in columns):
            src_mag = np.sqrt((src[columns].astype(float) ** 2).sum(axis=1))
            aug_mag = np.sqrt((aug[columns].astype(float) ** 2).sum(axis=1))
            src_median = float(np.median(src_mag))
            aug_median = float(np.median(aug_mag))
            ratio = aug_median / src_median if src_median else 1.0
            if not 0.75 <= ratio <= 1.25:
                warnings.append(f"{name} median magnitude ratio is {ratio:.4f}.")

    report = {
        "stage": "2.10.1",
        "source": str(args.source),
        "augmented": str(args.augmented),
        "source_rows": int(len(src)),
        "augmented_rows": int(len(aug)),
        "source_columns": int(len(src.columns)),
        "augmented_columns": int(len(aug.columns)),
        "schema_match": source_columns == augmented_columns,
        "activity_match": (
            src["activity"].astype(str).equals(aug["activity"].astype(str))
            if "activity" in src.columns and "activity" in aug.columns else False
        ),
        "duplicate_rows": duplicate_count,
        "nonfinite_numeric_rows": nonfinite_rows,
        "source_sentinels": src_sentinels,
        "augmented_sentinels": aug_sentinels,
        "source_timing": source_timing,
        "augmented_timing": augmented_timing,
        "warnings": warnings,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SAFEband MILESTONE 2.10.1 — AUGMENTATION VALIDATION")
    print(f"Schema          : {'PASS' if report['schema_match'] else 'FAIL'}")
    print(f"Rows            : {len(src)} -> {len(aug)}")
    print(f"Activity        : {'PASS' if report['activity_match'] else 'FAIL'}")
    print(f"Duplicates      : {duplicate_count}")
    print(f"Non-finite      : {nonfinite_rows}")
    print(f"Sentinels       : {'PASS' if src_sentinels == aug_sentinels else 'FAIL'}")
    print(f"Timing stalls   : {source_timing['stall_count']} -> {augmented_timing['stall_count']}")
    print(f"Stall fraction  : {source_timing['stall_fraction']:.4f} -> {augmented_timing['stall_fraction']:.4f}")
    print(f"Status          : {report['status']}")
    print(f"Report          : {args.output}")

    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
