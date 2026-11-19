import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "activity",
    "elapsed_ms",
    "sample_index",
    "heading_deg",
    "roll_deg",
    "pitch_deg",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "temperature_C",
    "humidity_pct",
    "pressure_hPa",
    "gas_kOhm",
    "bme_valid",
    "mic_rms",
    "mic_dc",
    "mic_average_rms",
    "mic_max_rms",
    "mic_blocks",
    "max_ir",
    "max_red",
    "finger_detected",
    "raw_hr",
    "raw_hr_valid",
    "raw_spo2",
    "raw_spo2_valid",
    "final_hr",
    "final_hr_valid",
    "final_spo2",
    "final_spo2_valid",
    "battery_voltage_V",
    "battery_percent",
    "charger_connected",
]

NUMERIC_COLUMNS = [
    "elapsed_ms",
    "sample_index",
    "heading_deg",
    "roll_deg",
    "pitch_deg",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "temperature_C",
    "humidity_pct",
    "pressure_hPa",
    "gas_kOhm",
    "mic_rms",
    "mic_dc",
    "mic_average_rms",
    "mic_max_rms",
    "mic_blocks",
    "max_ir",
    "max_red",
    "raw_hr",
    "raw_spo2",
    "final_hr",
    "final_spo2",
    "battery_voltage_V",
    "battery_percent",
]

VALIDITY_COLUMNS = [
    "bme_valid",
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "charger_connected",
]


def percentile_dict(series):
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return {}

    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]

    result = {}
    for p in q:
        result[f"p{p}"] = float(np.percentile(values, p))

    result["mean"] = float(values.mean())
    result["std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    return result


def run_lengths(values):
    values = np.asarray(values, dtype=bool)

    if len(values) == 0:
        return {}

    lengths = []
    current = 1

    for i in range(1, len(values)):
        if values[i] == values[i - 1]:
            current += 1
        else:
            lengths.append((bool(values[i - 1]), current))
            current = 1

    lengths.append((bool(values[-1]), current))

    valid_lengths = [n for state, n in lengths if state]
    invalid_lengths = [n for state, n in lengths if not state]

    return {
        "valid": {
            "count": len(valid_lengths),
            "max": int(max(valid_lengths)) if valid_lengths else 0,
            "mean": float(np.mean(valid_lengths)) if valid_lengths else 0.0,
        },
        "invalid": {
            "count": len(invalid_lengths),
            "max": int(max(invalid_lengths)) if invalid_lengths else 0,
            "mean": float(np.mean(invalid_lengths)) if invalid_lengths else 0.0,
        },
    }


def timing_characterization(df):
    if len(df) < 2:
        return {}

    dt = pd.to_numeric(df["elapsed_ms"], errors="coerce").diff().dropna()

    if dt.empty:
        return {}

    return {
        "count": int(len(dt)),
        "mean_ms": float(dt.mean()),
        "std_ms": float(dt.std(ddof=1)),
        "min_ms": float(dt.min()),
        "max_ms": float(dt.max()),
        "percentiles_ms": percentile_dict(dt),
        "intervals_le_500ms": int((dt <= 500).sum()),
        "intervals_gt_500ms": int((dt > 500).sum()),
        "intervals_gt_1000ms": int((dt > 1000).sum()),
        "intervals_gt_5000ms": int((dt > 5000).sum()),
        "fraction_gt_500ms": float((dt > 500).mean()),
        "fraction_gt_1000ms": float((dt > 1000).mean()),
        "fraction_gt_5000ms": float((dt > 5000).mean()),
        "effective_rate_hz_from_mean": (
            float(1000.0 / dt.mean()) if dt.mean() > 0 else None
        ),
        "effective_rate_hz_from_median": (
            float(1000.0 / dt.median()) if dt.median() > 0 else None
        ),
    }


def correlation_matrix(df):
    columns = [
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "temperature_C",
        "humidity_pct",
        "pressure_hPa",
        "gas_kOhm",
        "mic_rms",
        "mic_average_rms",
        "mic_max_rms",
        "max_ir",
        "max_red",
        "final_hr",
        "final_spo2",
        "battery_voltage_V",
        "battery_percent",
    ]

    available = [c for c in columns if c in df.columns]

    if len(available) < 2:
        return {}

    corr = df[available].corr(numeric_only=True)

    return {
        row: {
            col: (
                None
                if pd.isna(corr.loc[row, col])
                else float(corr.loc[row, col])
            )
            for col in corr.columns
        }
        for row in corr.index
    }


def characterize_file(path):
    df = pd.read_csv(path)

    missing_expected = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra_columns = [
        column for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    result = {
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "schema": {
            "matches_expected": not missing_expected and not extra_columns,
            "missing_expected": missing_expected,
            "extra_columns": extra_columns,
            "column_order_matches": list(df.columns) == EXPECTED_COLUMNS,
        },
        "activity": {
            "unique": df["activity"].dropna().astype(str).unique().tolist(),
            "counts": {
                str(k): int(v)
                for k, v in df["activity"].value_counts(dropna=False).items()
            },
        },
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
            if v > 0
        },
        "numeric": {},
        "timing": timing_characterization(df),
        "validity": {},
        "sentinels": {},
        "correlations": correlation_matrix(df),
    }

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            result["numeric"][column] = percentile_dict(df[column])

    for column in VALIDITY_COLUMNS:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        valid = numeric.eq(1)
        invalid = numeric.eq(0)

        result["validity"][column] = {
            "valid_count": int(valid.sum()),
            "invalid_count": int(invalid.sum()),
            "other_count": int((~(valid | invalid)).sum()),
            "valid_fraction": float(valid.mean()),
            "invalid_fraction": float(invalid.mean()),
            "runs": run_lengths(valid),
        }

    sentinel_columns = [
        "raw_hr",
        "raw_spo2",
        "final_hr",
        "final_spo2",
    ]

    for column in sentinel_columns:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")

            result["sentinels"][column] = {
                "minus_999_count": int((numeric == -999).sum()),
                "minus_999_fraction": float((numeric == -999).mean()),
            }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Characterize real SafeBand sensor CSV recordings."
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="SafeBand CSV files to characterize.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/real_data_characterization/"
            "stage_a_characterization.json"
        ),
    )

    args = parser.parse_args()

    output = {
        "tool": "SafeBand real-data characterization",
        "version": "1.0",
        "expected_columns": EXPECTED_COLUMNS,
        "files": [],
    }

    for file_path in args.files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"SafeBand data file not found: {file_path}"
            )

        print(f"Characterizing: {file_path}")
        output["files"].append(characterize_file(file_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print()
    print(f"Saved characterization: {args.output}")
    print(f"Files processed: {len(output['files'])}")


if __name__ == "__main__":
    main()
