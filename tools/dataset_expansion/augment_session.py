from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MOTION_COLUMNS = [
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
]

ORIENTATION_COLUMNS = ["heading_deg", "roll_deg", "pitch_deg"]

ENVIRONMENT_COLUMNS = [
    "temperature_C", "humidity_pct", "pressure_hPa", "gas_kOhm",
]

AUDIO_COLUMNS = ["mic_rms", "mic_dc", "mic_average_rms", "mic_max_rms"]

PPG_VALUE_COLUMNS = ["max_ir", "max_red", "raw_hr", "raw_spo2", "final_hr", "final_spo2"]

SENTINEL = -999.0


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def bounded_noise(rng, n, sigma):
    return rng.normal(0.0, sigma, n)


def augment_motion(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    motion = cfg["motion"]

    accel_scale = rng.normal(motion["acceleration_scale_mean"], motion["acceleration_scale_std"])
    gyro_scale = rng.normal(motion["gyro_scale_mean"], motion["gyro_scale_std"])

    for col in MOTION_COLUMNS[:3]:
        out[col] = out[col].astype(float) * accel_scale
        out[col] += bounded_noise(rng, len(out), motion["acceleration_noise_std"])

    for col in MOTION_COLUMNS[3:]:
        out[col] = out[col].astype(float) * gyro_scale
        out[col] += bounded_noise(rng, len(out), motion["gyro_noise_std"])

    # Small correlated drift preserves temporal structure instead of adding
    # independent noise to every sample.
    drift_sigma = motion["drift_std"]
    drift = np.cumsum(rng.normal(0.0, drift_sigma, len(out)))

    for col in MOTION_COLUMNS[:3]:
        out[col] += drift * motion["acceleration_drift_scale"]

    return out


def augment_orientation(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    orientation = cfg["orientation"]

    for col in ORIENTATION_COLUMNS:
        values = out[col].astype(float).to_numpy(copy=True)
        noise = rng.normal(0.0, orientation["noise_std_deg"], len(values))
        out[col] = values + noise

    # Heading is circular.
    out["heading_deg"] = np.mod(out["heading_deg"].astype(float), 360.0)
    return out


def augment_environment(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    env = cfg["environment"]

    for col in ENVIRONMENT_COLUMNS:
        values = out[col].astype(float).to_numpy(copy=True)
        if col == "temperature_C":
            sigma = env["temperature_noise"]
        elif col == "humidity_pct":
            sigma = env["humidity_noise"]
        elif col == "pressure_hPa":
            sigma = env["pressure_noise"]
        else:
            sigma = env["gas_noise"]

        out[col] = values + rng.normal(0.0, sigma, len(values))

    out["humidity_pct"] = out["humidity_pct"].clip(0.0, 100.0)
    out["pressure_hPa"] = out["pressure_hPa"].clip(lower=0.0)
    out["gas_kOhm"] = out["gas_kOhm"].clip(lower=0.0)
    return out


def augment_audio(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    audio = cfg["audio"]

    for col in AUDIO_COLUMNS:
        values = out[col].astype(float).to_numpy(copy=True)
        values = values * rng.normal(audio["gain_mean"], audio["gain_std"])
        values += rng.normal(0.0, audio["noise_std"], len(values))
        out[col] = np.clip(values, 0.0, None)

    return out


def augment_ppg(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    ppg = cfg["ppg"]

    # Only perturb values that are valid. Firmware sentinels remain untouched.
    for col, sigma in {
        "max_ir": ppg["ir_noise_std"],
        "max_red": ppg["red_noise_std"],
    }.items():
        values = out[col].astype(float).to_numpy(copy=True)
        mask = np.isfinite(values) & (values != SENTINEL)
        values[mask] += rng.normal(0.0, sigma, mask.sum())
        out[col] = values

    for col, sigma in {
        "raw_hr": ppg["hr_noise_std"],
        "final_hr": ppg["hr_noise_std"],
        "raw_spo2": ppg["spo2_noise_std"],
        "final_spo2": ppg["spo2_noise_std"],
    }.items():
        values = out[col].astype(float).to_numpy(copy=True)
        mask = np.isfinite(values) & (values != SENTINEL)
        values[mask] += rng.normal(0.0, sigma, mask.sum())
        out[col] = values

    # Validity flags are intentionally NOT randomly flipped here.
    # That belongs to the stateful PPG/contact augmentation stage.
    out["raw_hr"] = out["raw_hr"].where(out["raw_hr"] == SENTINEL, out["raw_hr"].clip(25, 240))
    out["final_hr"] = out["final_hr"].where(out["final_hr"] == SENTINEL, out["final_hr"].clip(25, 240))
    out["raw_spo2"] = out["raw_spo2"].where(out["raw_spo2"] == SENTINEL, out["raw_spo2"].clip(50, 100))
    out["final_spo2"] = out["final_spo2"].where(out["final_spo2"] == SENTINEL, out["final_spo2"].clip(50, 100))

    return out


def augment_timing(df: pd.DataFrame, rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    timing = cfg["timing"]

    if len(out) < 2:
        return out

    elapsed = out["elapsed_ms"].astype(float).to_numpy()
    intervals = np.diff(elapsed)

    normal = intervals[
        (intervals >= timing["normal_min_ms"]) &
        (intervals <= timing["normal_max_ms"])
    ]
    stalls = intervals[intervals >= timing["stall_min_ms"]]

    if len(normal) == 0:
        normal = intervals
    if len(stalls) == 0:
        stalls = intervals

    stall_fraction = float(np.mean(intervals >= timing["stall_min_ms"]))
    target_stalls = int(round((len(out) - 1) * stall_fraction))

    normal_draws = rng.choice(normal, size=len(out) - 1, replace=True).astype(float)
    if target_stalls:
        stall_idx = rng.choice(len(normal_draws), size=min(target_stalls, len(normal_draws)), replace=False)
        normal_draws[stall_idx] = rng.choice(stalls, size=len(stall_idx), replace=True)

    new_elapsed = np.empty(len(out), dtype=np.int64)
    new_elapsed[0] = int(elapsed[0])
    new_elapsed[1:] = new_elapsed[0] + np.cumsum(np.rint(normal_draws).astype(np.int64))

    out["elapsed_ms"] = new_elapsed
    out["sample_index"] = np.arange(len(out), dtype=np.int64)
    return out


def augment_session(source: Path, output: Path, config_path: Path, seed: int) -> dict:
    cfg = load_config(config_path)
    df = pd.read_csv(source)

    required = cfg["schema"]["required_columns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    original_columns = list(df.columns)
    rng = np.random.default_rng(seed)

    out = augment_motion(df, rng, cfg)
    out = augment_orientation(out, rng, cfg)
    out = augment_environment(out, rng, cfg)
    out = augment_audio(out, rng, cfg)
    out = augment_ppg(out, rng, cfg)
    out = augment_timing(out, rng, cfg)

    # Preserve firmware schema and untouched metadata columns exactly.
    out = out[original_columns]

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    provenance = {
        "stage": "2.10.1",
        "type": "synthetic_augmentation",
        "source_file": str(source),
        "output_file": str(output),
        "seed": seed,
        "source_rows": int(len(df)),
        "output_rows": int(len(out)),
        "source_activity": sorted(df["activity"].dropna().astype(str).unique().tolist()),
        "output_activity": sorted(out["activity"].dropna().astype(str).unique().tolist()),
        "schema_columns": original_columns,
        "status": "SYNTHETIC_UNVALIDATED",
        "note": "Generic diversity augmentation only; not hazard generation and not a substitute for real validation data.",
    }

    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()

    provenance = augment_session(args.source, args.output, args.config, args.seed)
    print("SAFEband MILESTONE 2.10.1 — GENERIC SESSION AUGMENTATION")
    print(f"Source rows : {provenance['source_rows']}")
    print(f"Output rows : {provenance['output_rows']}")
    print(f"Seed        : {provenance['seed']}")
    print(f"Activity    : {provenance['source_activity']}")
    print("Status      : SYNTHETIC_UNVALIDATED")
    print(f"Dataset     : {args.output}")
    print(f"Provenance  : {args.output.with_suffix('.provenance.json')}")


if __name__ == "__main__":
    main()


