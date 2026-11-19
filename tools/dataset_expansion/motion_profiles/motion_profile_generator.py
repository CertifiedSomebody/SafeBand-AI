from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACCEL_COLUMNS = [
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
]

GYRO_COLUMNS = [
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
]


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_noise(
    n: int,
    rng: np.random.Generator,
    window: int = 25,
) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=float)

    raw = rng.normal(0.0, 1.0, n)

    window = max(3, min(window, n))

    kernel = np.ones(window, dtype=float) / window

    smoothed = np.convolve(
        raw,
        kernel,
        mode="same",
    )

    std = np.std(smoothed)

    if std > 1e-12:
        smoothed /= std

    return smoothed


def build_envelope(
    n: int,
    profile: str,
    target_scale: float,
    transition_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:

    if n == 0:
        return np.empty(0, dtype=float)

    transition_rows = max(
        2,
        min(transition_rows, max(2, n // 2)),
    )

    t = np.linspace(
        0.0,
        1.0,
        n,
    )

    if profile == "WALKING_START":

        ramp = smoothstep(
            np.linspace(
                0.0,
                1.0,
                transition_rows,
            )
        )

        envelope = np.full(
            n,
            target_scale,
            dtype=float,
        )

        envelope[:transition_rows] = (
            0.35
            + (target_scale - 0.35) * ramp
        )

        return envelope

    if profile == "WALKING_STOP":

        ramp = smoothstep(
            np.linspace(
                0.0,
                1.0,
                transition_rows,
            )
        )

        envelope = np.full(
            n,
            target_scale,
            dtype=float,
        )

        envelope[-transition_rows:] = (
            target_scale
            + (0.25 - target_scale) * ramp
        )

        return envelope

    if profile == "WALKING_ACCELERATION":

        start_scale = 0.90
        end_scale = target_scale

        envelope = (
            start_scale
            + (
                end_scale
                - start_scale
            )
            * smoothstep(t)
        )

        return envelope

    if profile == "WALKING_DECELERATION":

        start_scale = 1.15
        end_scale = target_scale

        envelope = (
            start_scale
            + (
                end_scale
                - start_scale
            )
            * smoothstep(t)
        )

        return envelope

    if profile == "WALKING_IRREGULAR":

        variation = smooth_noise(
            n,
            rng,
            window=max(
                9,
                min(35, n // 8),
            ),
        )

        envelope = (
            target_scale
            + 0.10 * variation
        )

        return np.clip(
            envelope,
            0.75,
            1.35,
        )

    if profile in {
        "WALKING_SLOW",
        "WALKING_NORMAL",
        "WALKING_BRISK",
    }:

        envelope = np.ones(
            n,
            dtype=float,
        )

        if transition_rows > 1:

            ramp = smoothstep(
                np.linspace(
                    0.0,
                    1.0,
                    transition_rows,
                )
            )

            envelope[:transition_rows] = (
                1.0
                + (
                    target_scale
                    - 1.0
                )
                * ramp
            )

            reverse = ramp[::-1]

            envelope[-transition_rows:] = (
                1.0
                + (
                    target_scale
                    - 1.0
                )
                * reverse
            )

        if n > 2 * transition_rows:

            envelope[
                transition_rows:
                -transition_rows
            ] = target_scale

        return envelope

    return np.ones(
        n,
        dtype=float,
    )


def apply_profile(
    accel: np.ndarray,
    gyro: np.ndarray,
    envelope: np.ndarray,
    profile: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    accel_out = accel * envelope[:, None]
    gyro_out = gyro * envelope[:, None]

    if profile == "WALKING_TURNING":

        n = len(gyro_out)

        turn_shape = smooth_noise(
            n,
            rng,
            window=max(
                15,
                min(45, n // 6),
            ),
        )

        turn_strength = (
            0.20
            * turn_shape
        )

        gyro_out[:, 2] += (
            np.abs(gyro_out[:, 2])
            + 1.0
        ) * turn_strength

        gyro_out[:, 1] *= (
            1.0
            + 0.15
            * np.abs(turn_shape)
        )

    return accel_out, gyro_out


def validate_numeric(
    frame: pd.DataFrame,
) -> None:

    numeric = frame.select_dtypes(
        include=[np.number]
    )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():

        raise ValueError(
            "Generated dataset contains "
            "non-finite numeric values."
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "SafeBand 2.10.2 final semantic "
            "motion profile generator"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--profile",
        required=True,
    )

    args = parser.parse_args()

    source = pd.read_csv(
        args.source
    )

    config = json.loads(
        args.config.read_text(
            encoding="utf-8-sig"
        )
    )

    profiles = config["profiles"]

    if args.profile not in profiles:

        raise ValueError(
            f"Unknown profile "
            f"'{args.profile}'. "
            f"Available: "
            f"{', '.join(profiles)}"
        )

    profile = profiles[
        args.profile
    ]

    missing_accel = [
        column
        for column in ACCEL_COLUMNS
        if column not in source.columns
    ]

    missing_gyro = [
        column
        for column in GYRO_COLUMNS
        if column not in source.columns
    ]

    if missing_accel:

        raise ValueError(
            "Missing acceleration columns: "
            f"{missing_accel}"
        )

    if missing_gyro:

        raise ValueError(
            "Missing gyroscope columns: "
            f"{missing_gyro}"
        )

    output = source.copy()

    transition_low, transition_high = (
        profile["transition_rows"]
    )

    seed = 20261118

    rng = np.random.default_rng(
        seed
    )

    transition_rows = int(
        rng.integers(
            transition_low,
            transition_high + 1,
        )
    )

    acceleration_scale = float(
        profile[
            "acceleration_scale"
        ]
    )

    gyro_scale = float(
        profile[
            "gyro_scale"
        ]
    )

    envelope = build_envelope(
        len(output),
        args.profile,
        acceleration_scale,
        transition_rows,
        rng,
    )

    accel = (
        output[
            ACCEL_COLUMNS
        ]
        .astype(float)
        .to_numpy(copy=True)
    )

    gyro = (
        output[
            GYRO_COLUMNS
        ]
        .astype(float)
        .to_numpy(copy=True)
    )

    accel_scaled, gyro_scaled = (
        apply_profile(
            accel,
            gyro,
            envelope,
            args.profile,
            rng,
        )
    )

    output.loc[
        :,
        ACCEL_COLUMNS
    ] = accel_scaled

    output.loc[
        :,
        GYRO_COLUMNS
    ] = gyro_scaled

    validate_numeric(
        output
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    semantic_models = {

        "WALKING_START":
            "START_LOW_TO_WALKING",

        "WALKING_STOP":
            "WALKING_TO_STOP_LOW",

        "WALKING_ACCELERATION":
            "PROGRESSIVE_MOTION_INCREASE",

        "WALKING_DECELERATION":
            "PROGRESSIVE_MOTION_DECREASE",

        "WALKING_TURNING":
            "GYRO_DOMINANT_DIRECTIONAL_PERTURBATION",

        "WALKING_IRREGULAR":
            "CORRELATED_LOW_FREQUENCY_MOTION_VARIATION",

        "WALKING_SLOW":
            "STEADY_LOW_MOTION",

        "WALKING_NORMAL":
            "CONTROL_PROFILE",

        "WALKING_BRISK":
            "STEADY_HIGHER_MOTION",
    }

    provenance = {

        "stage":
            "2.10.2",

        "status":
            "SYNTHETIC_UNVALIDATED",

        "source":
            str(args.source),

        "output":
            str(args.output),

        "profile":
            args.profile,

        "semantic_model":
            semantic_models.get(
                args.profile,
                "UNKNOWN",
            ),

        "seed":
            seed,

        "transition_rows":
            transition_rows,

        "acceleration_scale":
            acceleration_scale,

        "gyro_scale":
            gyro_scale,

        "stateful":
            True,

        "source_rows":
            len(source),

        "output_rows":
            len(output),

        "source_columns":
            len(source.columns),

        "output_columns":
            len(output.columns),

        "schema_preserved":
            list(source.columns)
            == list(output.columns),
    }

    provenance_path = (
        args.output.with_suffix(
            ".provenance.json"
        )
    )

    provenance_path.write_text(
        json.dumps(
            provenance,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "SAFEband MILESTONE 2.10.2 — "
        "FINAL MOTION PROFILE GENERATOR"
    )

    print(
        f"Profile             : "
        f"{args.profile}"
    )

    print(
        f"Semantic model      : "
        f"{provenance['semantic_model']}"
    )

    print(
        f"Source rows         : "
        f"{len(source)}"
    )

    print(
        f"Output rows         : "
        f"{len(output)}"
    )

    print(
        f"Columns             : "
        f"{len(output.columns)}"
    )

    print(
        f"Acceleration scale  : "
        f"{acceleration_scale}"
    )

    print(
        f"Gyro scale          : "
        f"{gyro_scale}"
    )

    print(
        f"Transition rows     : "
        f"{transition_rows}"
    )

    print(
        "Stateful model      : "
        "YES"
    )

    print(
        "Schema preserved    : "
        f"{list(source.columns) == list(output.columns)}"
    )

    print(
        "Numeric validation  : PASS"
    )

    print(
        "Status              : "
        "SYNTHETIC_UNVALIDATED"
    )

    print(
        f"Dataset             : "
        f"{args.output}"
    )

    print(
        f"Provenance          : "
        f"{provenance_path}"
    )


if __name__ == "__main__":
    main()
