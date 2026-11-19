import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "datasets/real sensor data"
)

OUTPUT = Path(
    "reports/real_data_characterization/"
    "stage_a_temporal_sensor_analysis.json"
)

FILES = [
    INPUT / "stationary.csv",
    INPUT / "stationary_2.csv",
    INPUT / "walking.csv",
]

SIGNALS = [
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
    "mic_average_rms",
    "mic_max_rms",
    "max_ir",
    "max_red",
    "final_hr",
    "final_spo2",
    "battery_voltage_V",
    "battery_percent",
]

CORRELATION_SIGNALS = [
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "mic_rms",
    "mic_average_rms",
    "mic_max_rms",
    "max_ir",
    "max_red",
    "final_hr",
    "final_spo2",
    "temperature_C",
    "humidity_pct",
    "pressure_hPa",
    "gas_kOhm",
    "battery_voltage_V",
    "battery_percent",
]


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def lag1_autocorrelation(series):
    values = clean_numeric(series).replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 3:
        return None

    x = values.iloc[:-1].to_numpy()
    y = values.iloc[1:].to_numpy()

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def adjacent_change(series):
    values = clean_numeric(series).replace([np.inf, -np.inf], np.nan)

    diff = values.diff().abs().dropna()

    if diff.empty:
        return {}

    return {
        "mean_abs_change": float(diff.mean()),
        "median_abs_change": float(diff.median()),
        "p95_abs_change": float(diff.quantile(0.95)),
        "max_abs_change": float(diff.max()),
    }


def magnitude_statistics(df):
    result = {}

    if all(
        c in df.columns
        for c in [
            "accel_x_m_s2",
            "accel_y_m_s2",
            "accel_z_m_s2",
        ]
    ):
        accel = np.sqrt(
            df["accel_x_m_s2"] ** 2
            + df["accel_y_m_s2"] ** 2
            + df["accel_z_m_s2"] ** 2
        )

        result["acceleration_magnitude"] = {
            "mean": float(accel.mean()),
            "std": float(accel.std()),
            "median": float(accel.median()),
            "p95": float(accel.quantile(0.95)),
            "p99": float(accel.quantile(0.99)),
            "min": float(accel.min()),
            "max": float(accel.max()),
        }

    if all(
        c in df.columns
        for c in [
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ]
    ):
        gyro = np.sqrt(
            df["gyro_x_rad_s"] ** 2
            + df["gyro_y_rad_s"] ** 2
            + df["gyro_z_rad_s"] ** 2
        )

        result["gyro_magnitude"] = {
            "mean": float(gyro.mean()),
            "std": float(gyro.std()),
            "median": float(gyro.median()),
            "p95": float(gyro.quantile(0.95)),
            "p99": float(gyro.quantile(0.99)),
            "min": float(gyro.min()),
            "max": float(gyro.max()),
        }

    return result


def correlation_pairs(df):
    available = [
        c for c in CORRELATION_SIGNALS
        if c in df.columns
    ]

    corr = df[available].corr(numeric_only=True)

    pairs = []

    for i, first in enumerate(available):
        for second in available[i + 1:]:
            value = corr.loc[first, second]

            if pd.isna(value):
                continue

            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "correlation": float(value),
                    "absolute_correlation": float(abs(value)),
                }
            )

    pairs.sort(
        key=lambda x: x["absolute_correlation"],
        reverse=True,
    )

    return pairs


def analyze_file(path):
    df = pd.read_csv(path)

    timing = pd.to_numeric(
        df["elapsed_ms"],
        errors="coerce",
    ).diff()

    timing_gap = timing > 1000

    result = {
        "file": str(path),
        "rows": int(len(df)),
        "lag1_autocorrelation": {},
        "adjacent_change": {},
        "motion_magnitude": magnitude_statistics(df),
        "timing": {
            "gaps_gt_1000ms": int(timing_gap.sum()),
            "gaps_gt_5000ms": int((timing > 5000).sum()),
        },
        "strongest_correlations": [],
    }

    for signal in SIGNALS:
        if signal not in df.columns:
            continue

        result["lag1_autocorrelation"][signal] = (
            lag1_autocorrelation(df[signal])
        )

        result["adjacent_change"][signal] = (
            adjacent_change(df[signal])
        )

    result["strongest_correlations"] = correlation_pairs(df)[:30]

    return result


def main():
    output = {
        "analysis": "SafeBand temporal and sensor relationship analysis",
        "version": "1.0",
        "files": [],
    }

    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(path)

        print(f"Analyzing: {path.name}")
        output["files"].append(analyze_file(path))

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
