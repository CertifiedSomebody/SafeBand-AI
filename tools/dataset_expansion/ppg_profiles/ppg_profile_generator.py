from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PPG_COLUMNS = [
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
    "raw_hr",
    "raw_spo2",
    "final_hr",
    "final_spo2",
]

VALIDITY_COLUMNS = [
    "finger_detected",
    "raw_hr_valid",
    "raw_spo2_valid",
    "final_hr_valid",
    "final_spo2_valid",
]

VALUE_COLUMNS = [
    "raw_hr",
    "raw_spo2",
    "final_hr",
    "final_spo2",
]

VALUE_VALIDITY_MAP = {
    "raw_hr": "raw_hr_valid",
    "raw_spo2": "raw_spo2_valid",
    "final_hr": "final_hr_valid",
    "final_spo2": "final_spo2_valid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stateful PPG/contact diversity from a real session."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True)

    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def validate_source(df: pd.DataFrame) -> None:
    missing = [column for column in PPG_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            "Source CSV is missing required PPG columns: "
            + ", ".join(missing)
        )

    if len(df) == 0:
        raise ValueError("Source CSV contains zero rows.")


def build_stateful_mask(
    n: int,
    probability: float,
    min_state_rows: int,
    max_state_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=bool)

    min_state_rows = max(1, int(min_state_rows))
    max_state_rows = max(min_state_rows, int(max_state_rows))

    mask = np.zeros(n, dtype=bool)

    cursor = 0
    state = bool(rng.random() < probability)

    while cursor < n:
        block_length = int(
            rng.integers(
                min_state_rows,
                max_state_rows + 1,
            )
        )

        end = min(n, cursor + block_length)

        if state:
            local_probability = float(
                np.clip(
                    probability + rng.normal(0.0, 0.035),
                    0.0,
                    1.0,
                )
            )

            mask[cursor:end] = rng.random(end - cursor) < local_probability

        cursor = end

        state = not state

    return mask


def synthesize_values(
    source: pd.Series,
    valid_mask: np.ndarray,
    noise_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    numeric = pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)

    usable = numeric[
        np.isfinite(numeric)
        & (numeric != -999.0)
    ]

    result = np.full(len(source), -999.0, dtype=float)

    if len(usable) == 0:
        return result

    count = int(valid_mask.sum())

    if count == 0:
        return result

    sampled = rng.choice(
        usable,
        size=count,
        replace=True,
    )

    if noise_fraction > 0:
        scale = max(float(np.std(usable)), 1.0)
        noise = rng.normal(
            0.0,
            scale * noise_fraction,
            size=count,
        )
        sampled = sampled + noise

    result[valid_mask] = sampled

    return result


def build_profile(
    df: pd.DataFrame,
    profile: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)

    n = len(df)

    finger_probability = float(
        profile["finger_valid_probability"]
    )

    hr_probability = float(
        profile["hr_valid_given_finger"]
    )

    spo2_probability = float(
        profile["spo2_valid_given_finger"]
    )

    final_hr_probability = float(
        profile["final_hr_given_raw_hr"]
    )

    final_spo2_probability = float(
        profile["final_spo2_given_raw_spo2"]
    )

    min_state_rows = int(profile["min_state_rows"])
    max_state_rows = int(profile["max_state_rows"])

    value_noise_fraction = float(
        profile["value_noise_fraction"]
    )

    finger = build_stateful_mask(
        n=n,
        probability=finger_probability,
        min_state_rows=min_state_rows,
        max_state_rows=max_state_rows,
        rng=rng,
    )

    raw_hr = (
        finger
        & (
            rng.random(n)
            < hr_probability
        )
    )

    raw_spo2 = (
        finger
        & (
            rng.random(n)
            < spo2_probability
        )
    )

    final_hr = (
        raw_hr
        & (
            rng.random(n)
            < final_hr_probability
        )
    )

    final_spo2 = (
        raw_spo2
        & (
            rng.random(n)
            < final_spo2_probability
        )
    )

    # Small recovery exceptions model transient algorithmic recovery
    # while maintaining the fundamental finger/contact dependency.
    recovery_hr = (
        finger
        & ~raw_hr
        & (
            rng.random(n)
            < 0.035
        )
    )

    recovery_spo2 = (
        finger
        & ~raw_spo2
        & (
            rng.random(n)
            < 0.015
        )
    )

    final_hr |= recovery_hr
    final_spo2 |= recovery_spo2

    output = df.copy()

    output["finger_detected"] = finger.astype(int)
    output["raw_hr_valid"] = raw_hr.astype(int)
    output["raw_spo2_valid"] = raw_spo2.astype(int)
    output["final_hr_valid"] = final_hr.astype(int)
    output["final_spo2_valid"] = final_spo2.astype(int)

    output["raw_hr"] = synthesize_values(
        df["raw_hr"],
        raw_hr,
        value_noise_fraction,
        rng,
    )

    output["raw_spo2"] = synthesize_values(
        df["raw_spo2"],
        raw_spo2,
        value_noise_fraction,
        rng,
    )

    output["final_hr"] = synthesize_values(
        df["final_hr"],
        final_hr,
        value_noise_fraction,
        rng,
    )

    output["final_spo2"] = synthesize_values(
        df["final_spo2"],
        final_spo2,
        value_noise_fraction,
        rng,
    )

    # Explicit mapping avoids accidental tuple-based DataFrame indexing.
    for value_column, validity_column in VALUE_VALIDITY_MAP.items():
        output.loc[
            output[validity_column] < 0.5,
            value_column,
        ] = -999.0

    # Enforce the firmware-style dependency graph.
    output.loc[
        output["finger_detected"] < 0.5,
        [
            "raw_hr_valid",
            "raw_spo2_valid",
            "final_hr_valid",
            "final_spo2_valid",
        ],
    ] = 0

    output.loc[
        output["finger_detected"] < 0.5,
        [
            "raw_hr",
            "raw_spo2",
            "final_hr",
            "final_spo2",
        ],
    ] = -999.0

    # Raw validity is a prerequisite for final validity.
    output.loc[
        output["raw_hr_valid"] < 0.5,
        "final_hr_valid",
    ] = 0

    output.loc[
        output["raw_spo2_valid"] < 0.5,
        "final_spo2_valid",
    ] = 0

    output.loc[
        output["final_hr_valid"] < 0.5,
        "final_hr",
    ] = -999.0

    output.loc[
        output["final_spo2_valid"] < 0.5,
        "final_spo2",
    ] = -999.0

    # Re-assert binary validity after all dependency enforcement.
    for column in VALIDITY_COLUMNS:
        output[column] = (
            output[column]
            .astype(float)
            .round()
            .astype(int)
        )

    provenance = {
        "stage": "2.10.3",
        "component": "ppg_contact_diversity",
        "profile": profile,
        "seed": seed,
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "schema_preserved": list(output.columns) == list(df.columns),
        "stateful_contact_model": True,
        "dependency_graph": {
            "finger_detected": [
                "raw_hr_valid",
                "raw_spo2_valid",
                "final_hr_valid",
                "final_spo2_valid",
            ],
            "raw_hr_valid": ["final_hr_valid"],
            "raw_spo2_valid": ["final_spo2_valid"],
        },
    }

    return output, provenance


def main() -> None:
    args = parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    config_path = Path(args.config)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {source_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found: {config_path}"
        )

    df = pd.read_csv(source_path)

    validate_source(df)

    config = load_config(config_path)

    profiles = config.get("profiles", {})

    if args.profile not in profiles:
        raise ValueError(
            f"Unknown profile '{args.profile}'. "
            f"Available profiles: {', '.join(profiles)}"
        )

    profile = profiles[args.profile]

    seed = int(
        config.get("seed", 20261118)
    )

    output, provenance = build_profile(
        df=df,
        profile=profile,
        seed=seed,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    provenance_path = output_path.with_suffix(
        output_path.suffix + ".provenance.json"
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
        f"PPG generation PASS: "
        f"{args.profile} -> "
        f"{output_path}"
    )

    print(
        f"Rows: {len(output)} | "
        f"Columns: {len(output.columns)}"
    )

    print(
        f"Finger rate: "
        f"{output['finger_detected'].mean():.4f}"
    )

    print(
        f"HR|finger: "
        f"{output.loc[output['finger_detected'] == 1, 'raw_hr_valid'].mean():.4f}"
        if (output["finger_detected"] == 1).any()
        else "HR|finger: 0.0000"
    )

    print(
        f"SpO2|finger: "
        f"{output.loc[output['finger_detected'] == 1, 'raw_spo2_valid'].mean():.4f}"
        if (output["finger_detected"] == 1).any()
        else "SpO2|finger: 0.0000"
    )

    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
