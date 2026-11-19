import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIMING_COLUMNS = {
    "elapsed_ms",
    "sample_index",
}

SENTINEL = -999.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def correlated_noise(
    rng: np.random.Generator,
    size: int,
    std: float,
    correlation: float,
) -> np.ndarray:

    if size <= 0:
        return np.array([], dtype=float)

    white = rng.normal(0.0, std, size)

    output = np.empty(size, dtype=float)
    output[0] = white[0]

    alpha = float(np.clip(correlation, 0.0, 0.99))

    scale = np.sqrt(max(1.0 - alpha * alpha, 1e-9))

    for index in range(1, size):
        output[index] = (
            alpha * output[index - 1]
            + scale * white[index]
        )

    return output


def generate_intervals(
    row_count: int,
    profile: dict,
    rng: np.random.Generator,
) -> np.ndarray:

    interval_count = max(row_count - 1, 0)

    if interval_count == 0:
        return np.array([], dtype=float)

    base = float(profile["base_interval_ms"])
    jitter_std = float(profile["jitter_std_ms"])
    correlation = float(
        profile.get("correlation", 0.0)
    )

    jitter = correlated_noise(
        rng,
        interval_count,
        jitter_std,
        correlation,
    )

    intervals = base + jitter

    minimum = float(
        profile.get("min_interval_ms", 300.0)
    )

    maximum = float(
        profile.get("max_interval_ms", 550.0)
    )

    intervals = np.clip(
        intervals,
        minimum,
        maximum,
    )

    burst_probability = float(
        profile.get("burst_probability", 0.0)
    )

    if burst_probability > 0.0:

        burst_mask = (
            rng.random(interval_count)
            < burst_probability
        )

        burst_interval = float(
            profile.get(
                "burst_interval_ms",
                250.0,
            )
        )

        burst_jitter_std = float(
            profile.get(
                "burst_jitter_ms",
                20.0,
            )
        )

        if burst_mask.any():

            burst_values = (
                burst_interval
                + rng.normal(
                    0.0,
                    burst_jitter_std,
                    int(burst_mask.sum()),
                )
            )

            intervals[burst_mask] = np.clip(
                burst_values,
                minimum,
                maximum,
            )

    stall_probability = float(
        profile.get(
            "stall_probability",
            0.0,
        )
    )

    if stall_probability > 0.0:

        stall_mask = (
            rng.random(interval_count)
            < stall_probability
        )

        stall_interval = float(
            profile.get(
                "stall_interval_ms",
                5429.0,
            )
        )

        stall_jitter = float(
            profile.get(
                "stall_jitter_ms",
                15.0,
            )
        )

        if stall_mask.any():

            stall_values = (
                stall_interval
                + rng.normal(
                    0.0,
                    stall_jitter,
                    int(stall_mask.sum()),
                )
            )

            intervals[stall_mask] = np.maximum(
                stall_values,
                1000.0,
            )

    return intervals


def build_elapsed_ms(
    source: pd.DataFrame,
    intervals: np.ndarray,
) -> np.ndarray:

    row_count = len(source)

    if row_count == 0:
        return np.array([], dtype=float)

    source_elapsed = pd.to_numeric(
        source["elapsed_ms"],
        errors="coerce",
    )

    start_value = float(source_elapsed.iloc[0])

    elapsed = np.empty(
        row_count,
        dtype=float,
    )

    elapsed[0] = start_value

    if row_count > 1:
        elapsed[1:] = (
            start_value
            + np.cumsum(intervals)
        )

    return elapsed


def generate_timing_profile(
    source: pd.DataFrame,
    profile_name: str,
    profile: dict,
    seed: int,
) -> tuple[pd.DataFrame, dict]:

    rng = np.random.default_rng(seed)

    output = source.copy()

    intervals = generate_intervals(
        len(source),
        profile,
        rng,
    )

    output["elapsed_ms"] = build_elapsed_ms(
        source,
        intervals,
    )

    output["sample_index"] = source[
        "sample_index"
    ].to_numpy()

    stall_mask = intervals > 1000.0

    provenance = {
        "stage": "2.10.5",
        "profile": profile_name,
        "seed": seed,
        "semantic_model": "empirical_acquisition_timing",
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "schema_preserved": list(output.columns)
        == list(source.columns),
        "sensor_values_modified": False,
        "sample_index_preserved": bool(
            np.array_equal(
                source["sample_index"].to_numpy(),
                output["sample_index"].to_numpy(),
            )
        ),
        "timing": {
            "base_interval_ms": float(
                profile["base_interval_ms"]
            ),
            "jitter_std_ms": float(
                profile["jitter_std_ms"]
            ),
            "stall_probability": float(
                profile.get("stall_probability", 0.0)
            ),
            "stall_interval_ms": float(
                profile.get("stall_interval_ms", 5429.0)
            ),
            "stall_count": int(stall_mask.sum()),
            "stall_fraction": float(
                stall_mask.mean()
            ) if len(stall_mask) else 0.0,
            "interval_count": int(len(intervals)),
            "interval_mean_ms": float(
                np.mean(intervals)
            ) if len(intervals) else 0.0,
            "interval_median_ms": float(
                np.median(intervals)
            ) if len(intervals) else 0.0,
            "interval_p95_ms": float(
                np.percentile(intervals, 95)
            ) if len(intervals) else 0.0,
            "interval_max_ms": float(
                np.max(intervals)
            ) if len(intervals) else 0.0,
        },
        "provenance_note": (
            "Timing-only augmentation. Sensor values, "
            "validity flags, activity labels, and sample "
            "indices are preserved. elapsed_ms is regenerated "
            "from an empirical SafeBand acquisition model."
        ),
    }

    return output, provenance


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate empirical timing/acquisition "
            "diversity for SafeBand sessions."
        )
    )

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

    source_path = Path(args.source)
    output_path = Path(args.output)
    config_path = Path(args.config)

    source = pd.read_csv(source_path)
    config = load_config(config_path)

    profiles = config["profiles"]

    if args.profile not in profiles:
        raise ValueError(
            f"Unknown timing profile: {args.profile}"
        )

    if "elapsed_ms" not in source.columns:
        raise ValueError(
            "Source dataset must contain elapsed_ms."
        )

    if "sample_index" not in source.columns:
        raise ValueError(
            "Source dataset must contain sample_index."
        )

    output, provenance = generate_timing_profile(
        source,
        args.profile,
        profiles[args.profile],
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
        str(output_path) + ".provenance.json"
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

    timing = provenance["timing"]

    print(
        f"TIMING GENERATION PASS: {args.profile}"
    )

    print(
        f"Rows: {len(output)} | "
        f"Columns: {len(output.columns)}"
    )

    print(
        f"Mean interval: "
        f"{timing['interval_mean_ms']:.2f} ms"
    )

    print(
        f"Median interval: "
        f"{timing['interval_median_ms']:.2f} ms"
    )

    print(
        f"P95 interval: "
        f"{timing['interval_p95_ms']:.2f} ms"
    )

    print(
        f"Max interval: "
        f"{timing['interval_max_ms']:.2f} ms"
    )

    print(
        f"Acquisition stalls: "
        f"{timing['stall_count']}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Provenance: {provenance_path}"
    )


if __name__ == "__main__":
    main()
