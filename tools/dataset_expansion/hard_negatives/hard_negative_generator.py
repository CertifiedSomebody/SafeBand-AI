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

ACTIVITY_COLUMNS = {
    "activity",
}

SENTINEL = -999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate label-preserving hard-negative sensor sessions."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def validate_source(df: pd.DataFrame) -> None:
    if len(df) == 0:
        raise ValueError("Source CSV contains zero rows.")

    if "sample_index" not in df.columns:
        raise ValueError("Source CSV must contain sample_index.")

    if "elapsed_ms" not in df.columns:
        raise ValueError("Source CSV must contain elapsed_ms.")

    if "activity" not in df.columns:
        raise ValueError("Source CSV must contain activity.")


def choose_event_start(
    n: int,
    event_rows: int,
    rng: np.random.Generator,
) -> int:

    if n <= event_rows:
        return 0

    return int(
        rng.integers(
            0,
            n - event_rows,
        )
    )


def smoothstep(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length, dtype=float)

    x = np.linspace(
        0.0,
        1.0,
        length,
    )

    return x * x * (3.0 - 2.0 * x)


def safe_numeric(
    series: pd.Series,
) -> np.ndarray:

    values = np.array(
        pd.to_numeric(
            series,
            errors="coerce",
        ).to_numpy(dtype=float),
        dtype=float,
        copy=True,
    )

    return values


def apply_scaled_event(
    output: pd.DataFrame,
    columns: list[str],
    start: int,
    length: int,
    scale: float,
    ramp_rows: int,
) -> int:

    end = min(
        len(output),
        start + length,
    )

    if end <= start:
        return 0

    local_length = end - start

    envelope = np.ones(
        local_length,
        dtype=float,
    )

    ramp = min(
        max(int(ramp_rows), 0),
        local_length // 2,
    )

    if ramp > 0:
        up = smoothstep(ramp)
        down = smoothstep(ramp)[::-1]

        envelope[:ramp] = up
        envelope[-ramp:] = down

    multiplier = (
        1.0
        + (scale - 1.0) * envelope
    )

    changed = 0

    for column in columns:

        if column not in output.columns:
            continue

        values = safe_numeric(
            output[column]
        )

        original = values[
            start:end
        ].copy()

        valid = (
            np.isfinite(original)
            & (original != SENTINEL)
        )

        values[
            start:end
        ][valid] = (
            original[valid]
            * multiplier[valid]
        )

        output[column] = values

        changed += int(
            np.sum(
                values[start:end]
                != original
            )
        )

    return changed


def apply_stop_start(
    output: pd.DataFrame,
    groups: list[list[str]],
    start: int,
    length: int,
    minimum_scale: float,
    recovery_scale: float,
    ramp_rows: int,
) -> int:

    end = min(
        len(output),
        start + length,
    )

    if end <= start:
        return 0

    local_length = end - start

    midpoint = max(
        1,
        local_length // 2,
    )

    envelope = np.ones(
        local_length,
        dtype=float,
    )

    down = smoothstep(midpoint)

    for index in range(midpoint):
        envelope[index] = (
            1.0
            - (1.0 - minimum_scale)
            * down[index]
        )

    recovery_length = (
        local_length - midpoint
    )

    if recovery_length > 0:
        recovery = smoothstep(
            recovery_length
        )

        envelope[midpoint:] = (
            minimum_scale
            + (
                recovery_scale
                - minimum_scale
            )
            * recovery
        )

    changed = 0

    for group in groups:

        for column in group:

            if column not in output.columns:
                continue

            values = safe_numeric(
                output[column]
            )

            original = values[
                start:end
            ].copy()

            valid = (
                np.isfinite(original)
                & (original != SENTINEL)
            )

            values[
                start:end
            ][valid] = (
                original[valid]
                * envelope[valid]
            )

            output[column] = values

            changed += int(
                np.sum(
                    values[start:end]
                    != original
                )
            )

    return changed


def apply_ppg_artifact(
    output: pd.DataFrame,
    columns: list[str],
    start: int,
    length: int,
    scale: float,
    ramp_rows: int,
) -> int:

    end = min(
        len(output),
        start + length,
    )

    if end <= start:
        return 0

    local_length = end - start

    envelope = np.ones(
        local_length,
        dtype=float,
    )

    ramp = min(
        max(int(ramp_rows), 0),
        local_length // 2,
    )

    if ramp > 0:
        up = smoothstep(ramp)
        down = smoothstep(ramp)[::-1]

        envelope[:ramp] = up
        envelope[-ramp:] = down

    multiplier = (
        1.0
        + (scale - 1.0) * envelope
    )

    changed = 0

    for column in columns:

        if column not in output.columns:
            continue

        values = safe_numeric(
            output[column]
        )

        original = values[
            start:end
        ].copy()

        valid = (
            np.isfinite(original)
            & (original != SENTINEL)
        )

        values[
            start:end
        ][valid] = (
            original[valid]
            * multiplier[valid]
        )

        output[column] = values

        changed += int(
            np.sum(
                values[start:end]
                != original
            )
        )

    return changed


def apply_timing_stalls(
    output: pd.DataFrame,
    profile: dict,
    rng: np.random.Generator,
) -> int:

    elapsed = safe_numeric(
        output["elapsed_ms"]
    )

    if len(elapsed) < 2:
        return 0

    intervals = np.diff(
        elapsed
    )

    event_count = int(
        profile["event_count"]
    )

    minimum = float(
        profile["stall_interval_min_ms"]
    )

    maximum = float(
        profile["stall_interval_max_ms"]
    )

    valid_positions = np.arange(
        len(intervals)
    )

    count = min(
        event_count,
        len(valid_positions),
    )

    if count <= 0:
        return 0

    selected = rng.choice(
        valid_positions,
        size=count,
        replace=False,
    )

    for position in selected:

        intervals[position] = (
            rng.uniform(
                minimum,
                maximum,
            )
        )

    new_elapsed = np.zeros(
        len(elapsed),
        dtype=float,
    )

    new_elapsed[0] = elapsed[0]

    if len(intervals) > 0:
        new_elapsed[1:] = (
            elapsed[0]
            + np.cumsum(intervals)
        )

    changed = int(
        np.sum(
            new_elapsed
            != elapsed
        )
    )

    output["elapsed_ms"] = new_elapsed

    return changed


def generate(
    source: pd.DataFrame,
    profile: dict,
    profile_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:

    rng = np.random.default_rng(
        seed
    )

    output = source.copy(
        deep=True
    )

    changed_columns: set[str] = set()
    changed_rows: set[int] = set()
    event_starts: list[int] = []
    changed_values = 0

    event_count = int(
        profile["event_count"]
    )

    min_rows = int(
        profile["min_event_rows"]
    )

    max_rows = int(
        profile["max_event_rows"]
    )

    for _ in range(event_count):

        length = int(
            rng.integers(
                min_rows,
                max_rows + 1,
            )
        )

        start = choose_event_start(
            len(output),
            length,
            rng,
        )

        event_starts.append(start)

        before = output.copy(
            deep=True
        )

        if profile_name == "ABRUPT_ACCELERATION":

            scale = float(
                rng.uniform(
                    profile["amplitude_scale_min"],
                    profile["amplitude_scale_max"],
                )
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_columns"],
                start,
                length,
                scale,
                profile["ramp_rows"],
            )

        elif profile_name == "ABRUPT_DECELERATION":

            scale = float(
                rng.uniform(
                    profile["amplitude_scale_min"],
                    profile["amplitude_scale_max"],
                )
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_columns"],
                start,
                length,
                scale,
                profile["ramp_rows"],
            )

        elif profile_name == "SHARP_TURN":

            scale = float(
                rng.uniform(
                    profile["amplitude_scale_min"],
                    profile["amplitude_scale_max"],
                )
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_columns"],
                start,
                length,
                scale,
                profile["ramp_rows"],
            )

        elif profile_name == "MOTION_SPIKE":

            accel_scale = float(
                rng.uniform(
                    profile["accel_scale_min"],
                    profile["accel_scale_max"],
                )
            )

            gyro_scale = float(
                rng.uniform(
                    profile["gyro_scale_min"],
                    profile["gyro_scale_max"],
                )
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_groups"][0],
                start,
                length,
                accel_scale,
                profile["ramp_rows"],
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_groups"][1],
                start,
                length,
                gyro_scale,
                profile["ramp_rows"],
            )

        elif profile_name == "STOP_START":

            changed_values += apply_stop_start(
                output,
                profile["target_groups"],
                start,
                length,
                profile["minimum_scale"],
                profile["recovery_scale"],
                profile["ramp_rows"],
            )

        elif profile_name == "PPG_ARTIFACT":

            scale = float(
                rng.uniform(
                    profile["amplitude_scale_min"],
                    profile["amplitude_scale_max"],
                )
            )

            changed_values += apply_ppg_artifact(
                output,
                profile["target_columns"],
                start,
                length,
                scale,
                profile["ramp_rows"],
            )

        elif profile_name == "TIMING_STALL_NO_HAZARD":

            changed_values += apply_timing_stalls(
                output,
                profile,
                rng,
            )

            break

        elif profile_name == "NEAR_FALL_RECOVERY":

            accel_scale = float(
                rng.uniform(
                    profile["accel_scale_min"],
                    profile["accel_scale_max"],
                )
            )

            gyro_scale = float(
                rng.uniform(
                    profile["gyro_scale_min"],
                    profile["gyro_scale_max"],
                )
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_groups"][0],
                start,
                length,
                accel_scale,
                profile["ramp_rows"],
            )

            changed_values += apply_scaled_event(
                output,
                profile["target_groups"][1],
                start,
                length,
                gyro_scale,
                profile["ramp_rows"],
            )

            recovery_start = min(
                len(output) - 1,
                start + length,
            )

            recovery_length = min(
                3,
                len(output) - recovery_start,
            )

            if recovery_length > 0:
                changed_values += apply_scaled_event(
                    output,
                    profile["target_groups"][0],
                    recovery_start,
                    recovery_length,
                    profile["recovery_scale"],
                    1,
                )

        for column in output.columns:

            if column in INDEX_COLUMNS:
                continue

            before_values = before[column].to_numpy()
            after_values = output[column].to_numpy()

            changed_mask = (
                before_values
                != after_values
            )

            if changed_mask.any():
                changed_columns.add(column)

                changed_rows.update(
                    np.flatnonzero(
                        changed_mask
                    ).tolist()
                )

    provenance = {
        "stage": "2.10.6",
        "component": "hard_negative_generation",
        "profile": profile_name,
        "seed": seed,
        "semantic_model": "label_preserving_hard_negative",
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "source_columns": int(len(source.columns)),
        "output_columns": int(len(output.columns)),
        "source_activity_values": sorted(
            source["activity"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "output_activity_values": sorted(
            output["activity"]
            .astype(str)
            .unique()
            .tolist()
        ),
        "event_count": event_count,
        "event_starts": event_starts,
        "changed_rows": len(changed_rows),
        "changed_columns": sorted(
            changed_columns
        ),
        "changed_values": int(changed_values),
        "label_preserving": True,
        "schema_preserved": True,
        "status": "SYNTHETIC_UNVALIDATED",
    }

    return output, provenance


def main() -> None:

    args = parse_args()

    source_path = Path(
        args.source
    )

    output_path = Path(
        args.output
    )

    config_path = Path(
        args.config
    )

    config = load_config(
        config_path
    )

    profiles = config["profiles"]

    if args.profile not in profiles:
        raise ValueError(
            f"Unknown profile: {args.profile}. "
            f"Available: {', '.join(profiles)}"
        )

    source = pd.read_csv(
        source_path
    )

    validate_source(
        source
    )

    output, provenance = generate(
        source,
        profiles[args.profile],
        args.profile,
        args.seed,
    )
    provenance["source"] = str(source_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

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
        f"HARD-NEGATIVE GENERATION PASS: "
        f"{args.profile}"
    )
    print(
        f"Rows: {len(output)} | "
        f"Columns: {len(output.columns)}"
    )
    print(
        f"Changed rows: "
        f"{provenance['changed_rows']}"
    )
    print(
        f"Changed values: "
        f"{provenance['changed_values']}"
    )
    print(
        "Activity preserved: "
        f"{provenance['source_activity_values']}"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
