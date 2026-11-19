from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL = -999.0

VALIDITY_COLUMNS = {
    "valid",
    "validity",
    "valid_flag",
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "bme_valid",
    "bno_valid",
    "ppg_valid",
    "gps_valid",
}

META_COLUMNS = {
    "sample_index",
    "elapsed_ms",
    "activity",
    "label",
    "hazard_label",
    "hazard_type",
    "hazard_event_id",
    "is_hazard",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def safe_numeric(series: pd.Series) -> np.ndarray:
    return np.array(
        pd.to_numeric(
            series,
            errors="coerce",
        ).to_numpy(dtype=float),
        dtype=float,
        copy=True,
    )


def find_columns(columns, aliases):
    result = []

    lowered = {
        str(column).lower(): column
        for column in columns
    }

    for alias in aliases:
        alias = alias.lower()

        for lowered_name, original_name in lowered.items():
            if lowered_name == alias:
                result.append(original_name)

    return list(dict.fromkeys(result))


def discover_sensor_columns(df: pd.DataFrame) -> dict:
    columns = list(df.columns)

    required = {
        "accel_x": "accel_x_m_s2",
        "accel_y": "accel_y_m_s2",
        "accel_z": "accel_z_m_s2",
        "gyro_x": "gyro_x_rad_s",
        "gyro_y": "gyro_y_rad_s",
        "gyro_z": "gyro_z_rad_s",
    }

    missing = [
        column
        for column in required.values()
        if column not in columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required motion columns: "
            + ", ".join(missing)
        )

    return {
        "accel": [
            required["accel_x"],
            required["accel_y"],
            required["accel_z"],
        ],
        "gyro": [
            required["gyro_x"],
            required["gyro_y"],
            required["gyro_z"],
        ],
        "accel_x": [required["accel_x"]],
        "accel_y": [required["accel_y"]],
        "accel_z": [required["accel_z"]],
        "gyro_x": [required["gyro_x"]],
        "gyro_y": [required["gyro_y"]],
        "gyro_z": [required["gyro_z"]],
    }

def require_sensor_groups(columns):
    if len(columns["accel"]) < 3:
        raise RuntimeError(
            "Could not identify all three accelerometer axes. "
            f"Detected: {columns['accel']}"
        )

    if len(columns["gyro"]) < 3:
        raise RuntimeError(
            "Could not identify all three gyroscope axes. "
            f"Detected: {columns['gyro']}"
        )


def choose_event_start(
    rng: np.random.Generator,
    row_count: int,
    duration: int,
    event_number: int,
    total_events: int,
) -> int:

    margin = max(12, duration + 3)

    low = margin
    high = row_count - margin - duration

    if high <= low:
        return max(0, min(row_count - duration - 1, row_count // 2))

    regions = np.linspace(
        low,
        high,
        total_events + 1,
        dtype=int,
    )

    region_low = regions[event_number]
    region_high = max(
        region_low + 1,
        regions[event_number + 1],
    )

    return int(
        rng.integers(
            region_low,
            region_high,
        )
    )


def axis_sign(
    profile: str,
    axis_index: int,
) -> float:

    if profile == "FALL_FORWARD":
        return [1.0, 0.35, -1.0][axis_index]

    if profile == "FALL_BACKWARD":
        return [-1.0, 0.35, 1.0][axis_index]

    if profile == "FALL_SIDE":
        return [0.25, 1.0, -0.8][axis_index]

    if profile == "TRIP_RECOVERY":
        return [1.0, -0.6, 0.35][axis_index]

    if profile == "SUDDEN_COLLAPSE":
        return [-0.5, 0.25, -1.0][axis_index]

    if profile == "HIGH_IMPACT":
        return [0.7, -0.8, 1.0][axis_index]

    if profile == "LOSS_OF_BALANCE":
        return [0.9, -0.9, 0.5][axis_index]

    return 1.0


def apply_event(
    output: pd.DataFrame,
    profile: str,
    start: int,
    duration: int,
    acceleration_scale: float,
    gyro_scale: float,
    recovery: float,
    columns: dict,
    rng: np.random.Generator,
):

    end = min(
        len(output),
        start + duration,
    )

    if end <= start:
        return

    progress = np.linspace(
        0.0,
        1.0,
        end - start,
    )

    attack = np.sin(
        np.pi * progress
    )

    attack = np.power(
        np.clip(attack, 0.0, 1.0),
        0.70,
    )

    recovery_curve = np.ones_like(attack)

    if len(recovery_curve) > 2:
        recovery_start = max(
            1,
            int(len(recovery_curve) * 0.60),
        )

        recovery_curve[recovery_start:] = np.linspace(
            1.0,
            recovery,
            len(recovery_curve) - recovery_start,
        )

    event_curve = attack * recovery_curve

    accel_columns = [
        columns["accel_x"],
        columns["accel_y"],
        columns["accel_z"],
    ]

    gyro_columns = [
        columns["gyro_x"],
        columns["gyro_y"],
        columns["gyro_z"],
    ]

    for axis_index, axis_columns in enumerate(accel_columns):
        if not axis_columns:
            continue

        column = axis_columns[0]
        values = safe_numeric(output[column])

        baseline = np.nanmedian(
            values[
                np.isfinite(values)
                & (values != SENTINEL)
            ]
        )

        if not np.isfinite(baseline):
            baseline = 0.0

        sign = axis_sign(
            profile,
            axis_index,
        )

        disturbance = (
            baseline
            * (acceleration_scale - 1.0)
            * sign
            * event_curve
        )

        disturbance += rng.normal(
            0.0,
            max(abs(baseline) * 0.03, 0.01),
            len(disturbance),
        )

        target = values[start:end]

        valid = (
            np.isfinite(target)
            & (target != SENTINEL)
        )

        target[valid] += disturbance[valid]

        values[start:end] = target

        output[column] = values

    for axis_index, axis_columns in enumerate(gyro_columns):
        if not axis_columns:
            continue

        column = axis_columns[0]
        values = safe_numeric(output[column])

        baseline = np.nanmedian(
            np.abs(
                values[
                    np.isfinite(values)
                    & (values != SENTINEL)
                ]
            )
        )

        if not np.isfinite(baseline):
            baseline = 1.0

        sign = axis_sign(
            profile,
            axis_index,
        )

        disturbance = (
            baseline
            * (gyro_scale - 1.0)
            * sign
            * event_curve
        )

        disturbance += rng.normal(
            0.0,
            max(abs(baseline) * 0.05, 0.01),
            len(disturbance),
        )

        target = values[start:end]

        valid = (
            np.isfinite(target)
            & (target != SENTINEL)
        )

        target[valid] += disturbance[valid]

        values[start:end] = target

        output[column] = values


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--profile",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    source = Path(args.source)
    output_path = Path(args.output)
    config_path = Path(args.config)

    config = load_config(config_path)

    if args.profile not in config["profiles"]:
        raise ValueError(
            f"Unknown hazard profile: {args.profile}"
        )

    profile_config = config["profiles"][args.profile]

    source_df = pd.read_csv(source)
    output_df = source_df.copy(deep=True)

    sensor_columns = discover_sensor_columns(
        source_df
    )

    require_sensor_groups(
        sensor_columns
    )

    rng = np.random.default_rng(
        args.seed
    )

    hazard_event_ids = np.full(
        len(output_df),
        -1,
        dtype=int,
    )

    hazard_labels = np.zeros(
        len(output_df),
        dtype=int,
    )

    hazard_types = np.full(
        len(output_df),
        "NONE",
        dtype=object,
    )

    event_count = int(
        profile_config["events"]
    )

    min_duration, max_duration = (
        profile_config["duration_rows"]
    )

    used_ranges = []

    for event_number in range(event_count):

        duration = int(
            rng.integers(
                min_duration,
                max_duration + 1,
            )
        )

        start = choose_event_start(
            rng,
            len(output_df),
            duration,
            event_number,
            event_count,
        )

        end = min(
            len(output_df),
            start + duration,
        )

        for old_start, old_end in used_ranges:
            if start < old_end and end > old_start:
                start = min(
                    len(output_df) - duration - 1,
                    old_end + 2,
                )
                end = min(
                    len(output_df),
                    start + duration,
                )

        used_ranges.append(
            (start, end)
        )

        acceleration_scale = rng.uniform(
            profile_config["acceleration_scale"][0],
            profile_config["acceleration_scale"][1],
        )

        gyro_scale = rng.uniform(
            profile_config["gyro_scale"][0],
            profile_config["gyro_scale"][1],
        )

        apply_event(
            output_df,
            args.profile,
            start,
            duration,
            acceleration_scale,
            gyro_scale,
            float(profile_config["recovery"]),
            sensor_columns,
            rng,
        )

        hazard_event_ids[start:end] = (
            event_number + 1
        )

        hazard_labels[start:end] = 1

        hazard_types[start:end] = (
            profile_config["hazard_type"]
        )

        if profile_config.get(
            "impact_row",
            False,
        ):
            impact_index = start + duration // 2

            if impact_index < len(output_df):
                hazard_event_ids[
                    impact_index
                ] = event_number + 1

    output_df["hazard_label"] = hazard_labels
    output_df["hazard_type"] = hazard_types
    output_df["hazard_event_id"] = hazard_event_ids
    output_df["is_hazard"] = hazard_labels

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        output_path,
        index=False,
    )

    changed_mask = (
        output_df.drop(
            columns=[
                "hazard_label",
                "hazard_type",
                "hazard_event_id",
                "is_hazard",
            ],
            errors="ignore",
        )
        != source_df
    ).any(axis=1)

    changed_values = int(
        (
            output_df.drop(
                columns=[
                    "hazard_label",
                    "hazard_type",
                    "hazard_event_id",
                    "is_hazard",
                ],
                errors="ignore",
            )
            != source_df
        )
        .sum()
        .sum()
    )

    provenance = {
        "stage": "2.10.7",
        "semantic_model": "hazard_scenario_generation",
        "status": "SYNTHETIC_UNVALIDATED",
        "profile": args.profile,
        "hazard_type": profile_config["hazard_type"],
        "seed": args.seed,
        "source": str(source),
        "rows": int(len(output_df)),
        "columns": int(len(output_df.columns)),
        "source_sensor_columns": sensor_columns,
        "event_count": event_count,
        "hazard_rows": int(
            (hazard_labels == 1).sum()
        ),
        "changed_rows": int(
            changed_mask.sum()
        ),
        "changed_values": changed_values,
        "hazard_label_column": "hazard_label",
        "hazard_type_column": "hazard_type",
        "hazard_event_id_column": "hazard_event_id",
        "is_hazard_column": "is_hazard",
    }

    provenance_path = Path(
        str(output_path)
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
        f"HAZARD SCENARIO GENERATION PASS: {args.profile}"
    )
    print(
        f"Rows: {len(output_df)} | "
        f"Columns: {len(output_df.columns)}"
    )
    print(
        f"Hazard rows: {(hazard_labels == 1).sum()}"
    )
    print(
        f"Changed rows: {changed_mask.sum()}"
    )
    print(
        f"Changed values: {changed_values}"
    )
    print(
        f"Hazard type: {profile_config['hazard_type']}"
    )
    print(
        "Status: SYNTHETIC_UNVALIDATED"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
