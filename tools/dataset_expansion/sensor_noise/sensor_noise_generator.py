from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL = -999.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sensor-specific noise diversity."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def is_numeric_sensor_column(column: str) -> bool:
    name = column.lower()

    excluded = {
        "elapsed_ms",
        "sample_index",
        "bme_valid",
        "finger_detected",
        "raw_hr_valid",
        "raw_spo2_valid",
        "final_hr_valid",
        "final_spo2_valid",
        "charger_connected",
    }

    if name in excluded:
        return False

    if any(token in name for token in [
        "valid",
        "detected",
        "status",
        "activity",
        "label",
        "state",
        "timestamp",
        "time",
        "index",
        "sequence",
        "mode",
    ]):
        return False

    return True


def classify_sensor(column: str) -> str:
    name = column.lower()

    if any(token in name for token in [
        "accel",
        "acc_",
        "acceleration",
        "linear_acc",
    ]):
        return "accelerometer"

    if any(token in name for token in [
        "gyro",
        "gyroscope",
        "angular",
    ]):
        return "gyroscope"

    if any(token in name for token in [
        "temp",
        "temperature",
        "humid",
        "humidity",
        "pressure",
        "press",
        "gas",
        "iaq",
        "air_quality",
    ]):
        return "bme680"

    if any(token in name for token in [
        "raw_hr",
        "final_hr",
        "heart_rate",
        "hr",
        "spo2",
        "oxygen",
    ]):
        return "ppg"

    if any(token in name for token in [
        "battery",
        "batt",
        "voltage",
        "charger",
    ]):
        return "battery"

    return "other"


def build_correlated_noise(
    n: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)

    white = rng.normal(0.0, scale, n)

    correlated = np.zeros(n)

    alpha = 0.92

    for i in range(1, n):
        correlated[i] = (
            alpha * correlated[i - 1]
            + (1.0 - alpha) * white[i]
        )

    return correlated


def robust_scale(values: np.ndarray) -> float:
    usable = values[
        np.isfinite(values)
        & (values != SENTINEL)
    ]

    if len(usable) == 0:
        return 1.0

    median = float(np.median(usable))

    mad = float(
        np.median(
            np.abs(usable - median)
        )
    )

    scale = 1.4826 * mad

    if scale <= 1e-9:
        scale = float(np.std(usable))

    return max(scale, 1e-6)


def infer_validity_column(df: pd.DataFrame, column: str):
    candidates = [
        f"{column}_valid",
        f"{column}_validity",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def apply_noise(
    values: np.ndarray,
    sensor_type: str,
    profile: dict,
    rng: np.random.Generator,
    validity: np.ndarray | None,
):
    output = values.astype(float).copy()

    usable = (
        np.isfinite(output)
        & (output != SENTINEL)
    )

    if validity is not None:
        usable &= validity > 0.5

    if not usable.any():
        return output, {
            "modified": 0,
            "spikes": 0,
        }

    base_scale = robust_scale(output)

    gaussian_fraction = float(
        profile["gaussian_fraction"]
    )

    correlated_fraction = float(
        profile["correlated_fraction"]
    )

    spike_probability = float(
        profile["spike_probability"]
    )

    spike_fraction = float(
        profile["spike_fraction"]
    )

    if sensor_type == "accelerometer":
        gaussian_multiplier = 1.00
        correlated_multiplier = 1.00
        spike_multiplier = 1.00

    elif sensor_type == "gyroscope":
        gaussian_multiplier = 1.15
        correlated_multiplier = 1.10
        spike_multiplier = 1.20

    elif sensor_type == "bme680":
        gaussian_multiplier = 0.75
        correlated_multiplier = 1.35
        spike_multiplier = 0.60

    elif sensor_type == "ppg":
        gaussian_multiplier = 0.55
        correlated_multiplier = 0.80
        spike_multiplier = 0.35

    elif sensor_type == "battery":
        gaussian_multiplier = 0.35
        correlated_multiplier = 0.40
        spike_multiplier = 0.20

    else:
        gaussian_multiplier = 0.75
        correlated_multiplier = 0.75
        spike_multiplier = 0.50

    gaussian_scale = (
        base_scale
        * gaussian_fraction
        * gaussian_multiplier
    )

    correlated_scale = (
        base_scale
        * correlated_fraction
        * correlated_multiplier
    )

    gaussian = rng.normal(
        0.0,
        gaussian_scale,
        len(output),
    )

    correlated = build_correlated_noise(
        len(output),
        correlated_scale,
        rng,
    )

    output[usable] += (
        gaussian[usable]
        + correlated[usable]
    )

    spike_mask = (
        usable
        & (
            rng.random(len(output))
            < (
                spike_probability
                * spike_multiplier
            )
        )
    )

    spike_count = int(spike_mask.sum())

    if spike_count:
        spike_sign = rng.choice(
            [-1.0, 1.0],
            size=spike_count,
        )

        spike_magnitude = (
            base_scale
            * spike_fraction
            * spike_sign
            * rng.uniform(
                0.5,
                1.5,
                spike_count,
            )
        )

        output[spike_mask] += spike_magnitude

    output[~usable] = SENTINEL

    return output, {
        "modified": int(usable.sum()),
        "spikes": spike_count,
        "base_scale": base_scale,
    }


def main():
    args = parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    config_path = Path(args.config)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source not found: {source_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    df = pd.read_csv(source_path)

    if df.empty:
        raise ValueError("Source dataset is empty.")

    config = load_json(config_path)

    profiles = config.get("profiles", {})

    if args.profile not in profiles:
        raise ValueError(
            f"Unknown profile '{args.profile}'. "
            f"Available: {', '.join(profiles)}"
        )

    profile = profiles[args.profile]

    seed = (
        args.seed
        if args.seed is not None
        else int(config.get("seed", 20261118))
    )

    rng = np.random.default_rng(seed)

    output = df.copy()

    sensor_summary = {}

    numeric_columns = []

    for column in df.columns:
        if not is_numeric_sensor_column(column):
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if not numeric.notna().any():
            continue

        numeric_columns.append(column)

    for column in numeric_columns:
        sensor_type = classify_sensor(column)

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        validity_column = infer_validity_column(
            df,
            column,
        )

        validity = None

        if validity_column is not None:
            validity = pd.to_numeric(
                df[validity_column],
                errors="coerce",
            ).fillna(0).to_numpy(dtype=float)

        modified, stats = apply_noise(
            values=values,
            sensor_type=sensor_type,
            profile=profile,
            rng=rng,
            validity=validity,
        )

        output[column] = modified

        sensor_summary[column] = {
            "sensor_type": sensor_type,
            "validity_column": validity_column,
            **stats,
        }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    provenance = {
        "stage": "2.10.4",
        "component": "sensor_noise_diversity",
        "profile_name": args.profile,
        "profile": profile,
        "seed": seed,
        "source_rows": int(len(df)),
        "output_rows": int(len(output)),
        "source_columns": int(len(df.columns)),
        "output_columns": int(len(output.columns)),
        "schema_preserved": (
            list(df.columns)
            == list(output.columns)
        ),
        "sensor_columns_processed": numeric_columns,
        "sensor_summary": sensor_summary,
        "models": {
            "accelerometer": [
                "gaussian",
                "correlated_drift",
                "transient_spikes",
            ],
            "gyroscope": [
                "gaussian",
                "correlated_drift",
                "transient_spikes",
            ],
            "bme680": [
                "gaussian",
                "slow_correlated_drift",
                "transient_spikes",
            ],
            "ppg": [
                "measurement_domain_noise",
            ],
            "battery": [
                "measurement_noise",
                "quantization_compatible_noise",
            ],
        },
    }

    provenance_path = output_path.with_suffix(
        output_path.suffix
        + ".provenance.json"
    )

    with provenance_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            provenance,
            handle,
            indent=2,
        )

    print(
        f"SENSOR NOISE GENERATION PASS: "
        f"{args.profile}"
    )

    print(
        f"Rows: {len(output)} | "
        f"Columns: {len(output.columns)}"
    )

    print(
        f"Sensor columns processed: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
