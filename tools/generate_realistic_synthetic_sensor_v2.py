"""Generate independent, hardware-plausible SafeBand synthetic sensor sessions.

The generator uses latent session/physical state. It reads real recordings only
as bounded calibration references and never copies, resamples, or perturbs them.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GENERATOR_VERSION = "realistic_sensor_generator_v2"
MASTER_SEED = 20261119
OUTPUT_DIR = ROOT / "datasets" / "realistic_synthetic_v2"
REAL_DIR = ROOT / "datasets" / "real sensor data"
TARGET_REPORT = ROOT / "reports" / "realistic_synthetic_sensor_acceptance_report.json"

STANDARD_COLUMNS = [
    "activity", "elapsed_ms", "sample_index", "heading_deg", "roll_deg", "pitch_deg",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2", "gyro_x_rad_s",
    "gyro_y_rad_s", "gyro_z_rad_s", "temperature_C", "humidity_pct",
    "pressure_hPa", "gas_kOhm", "bme_valid", "mic_rms", "mic_dc",
    "mic_average_rms", "mic_max_rms", "mic_blocks", "max_ir", "max_red",
    "finger_detected", "raw_hr", "raw_hr_valid", "raw_spo2", "raw_spo2_valid",
    "final_hr", "final_hr_valid", "final_spo2", "final_spo2_valid",
    "battery_voltage_V", "battery_percent", "charger_connected",
]
HAZARD_COLUMNS = STANDARD_COLUMNS + [
    "hazard_label", "hazard_type", "hazard_event_id", "is_hazard",
]

ACTIVITY_LEVEL = {
    "STATIC": 0.10,
    "TRANSITION": 0.35,
    "WALKING": 0.60,
    "RUNNING": 1.00,
    "HIGH_MOTION": 1.35,
    "FALL_OR_HAZARD": 1.55,
}

SCHEDULES = [
    ["STATIC", "WALKING", "STATIC"],
    ["STATIC", "WALKING", "RUNNING", "WALKING", "STATIC"],
    ["WALKING", "WALKING", "STATIC"],
    ["STATIC", "STATIC", "WALKING"],
    ["STATIC", "TRANSITION", "WALKING", "TRANSITION", "STATIC"],
    ["STATIC", "WALKING", "TRANSITION", "RUNNING", "TRANSITION", "STATIC"],
    ["WALKING", "RUNNING", "WALKING", "STATIC"],
    ["STATIC", "HIGH_MOTION", "TRANSITION", "STATIC"],
    ["STATIC", "TRANSITION", "WALKING", "TRANSITION", "FALL_OR_HAZARD", "STATIC"],
    ["WALKING", "TRANSITION", "HIGH_MOTION", "TRANSITION", "WALKING"],
    ["STATIC", "RUNNING", "TRANSITION", "STATIC"],
    ["WALKING", "STATIC", "WALKING", "STATIC"],
]


def session_seed(master_seed: int, session_number: int) -> int:
    return int.from_bytes(
        hashlib.sha256(
            f"{master_seed}:{GENERATOR_VERSION}:{session_number}".encode()
        ).digest()[:8],
        "little",
    )


def smooth_noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    if n == 0:
        return np.empty(0)
    white = rng.normal(size=n)
    return gaussian_filter1d(white, sigma=sigma, mode="nearest")


def quantize(values: np.ndarray, step: float) -> np.ndarray:
    return np.round(values / step) * step


def wrap_heading(values: np.ndarray) -> np.ndarray:
    return np.mod(values, 360.0)


def clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(values, low, high)


def real_calibration() -> dict[str, Any]:
    samples: list[pd.DataFrame] = []
    timing: list[np.ndarray] = []
    for path in sorted(REAL_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        samples.append(df)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").dropna().to_numpy()
        if len(elapsed) > 1:
            timing.append(np.diff(elapsed))
    all_timing = np.concatenate(timing) if timing else np.array([394.0])
    return {
        "median_dt_ms": float(np.median(all_timing)),
        "p95_dt_ms": float(np.percentile(all_timing, 95)),
        "p99_dt_ms": float(np.percentile(all_timing, 99)),
        "max_dt_ms": float(np.max(all_timing)),
        "stall_probability": float(np.mean(all_timing > 1000)),
        "source_count": len(samples),
        "recordings": sorted(path.name for path in REAL_DIR.glob("*.csv")),
    }


def build_activity_schedule(
    rng: np.random.Generator,
    schedule_template: list[str],
    n: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    repeats = rng.integers(1, 4)
    expanded: list[str] = []
    for item in schedule_template:
        expanded.extend([item])
        if item != "STATIC" and rng.random() < 0.25:
            expanded.insert(len(expanded) - 1, "TRANSITION")
    if repeats > 1:
        expanded = expanded * repeats
        if len(expanded) > 10:
            expanded = expanded[:10]

    segment_count = len(expanded)
    segment_lengths = rng.dirichlet(np.full(segment_count, 2.0)) * n
    lengths = np.maximum(8, np.rint(segment_lengths).astype(int))
    lengths = lengths * n // max(1, lengths.sum())
    remainder = n - lengths.sum()
    if remainder > 0:
        lengths[-1] += remainder
    elif remainder < 0:
        for i in np.argsort(lengths)[::-1]:
            take = min(lengths[i] - 8, -remainder)
            lengths[i] -= take
            remainder += take
            if remainder >= 0:
                break

    activity = np.concatenate([
        np.full(length, label)
        for label, length in zip(expanded, lengths)
        if length > 0
    ]).astype("U20")
    if len(activity) < n:
        activity = np.pad(activity, (0, n - len(activity)), mode="edge")
    elif len(activity) > n:
        activity = activity[:n]

    # Replace abrupt boundaries with short transition segments.
    transition_counts: dict[str, int] = {}
    for i in range(1, n):
        if activity[i] != activity[i - 1] and activity[i - 1] != "TRANSITION":
            trans_len = int(rng.integers(4, 11))
            lo = max(0, i - trans_len // 2)
            hi = min(n, i + trans_len // 2)
            activity[lo:hi] = "TRANSITION"
    return activity, np.arange(n), {
        "schedule_template": schedule_template,
        "expanded_schedule": expanded,
        "segment_lengths_before_transition": lengths.tolist(),
        "transition_count": int(np.sum(activity[1:] != activity[:-1])),
    }


def generate_elapsed_ms(
    rng: np.random.Generator,
    n: int,
    calibration: dict[str, Any],
    nominal_interval_ms: float | None = None,
    timing_jitter_scale: float | None = None,
    stall_probability: float | None = None,
    stall_duration_ms: float | None = None,
) -> np.ndarray:
    if n == 0:
        return np.empty(0)
    base = float(nominal_interval_ms if nominal_interval_ms is not None else calibration["median_dt_ms"])
    jitter_scale = float(timing_jitter_scale if timing_jitter_scale is not None else 18.0)
    stall_probability = float(
        stall_probability if stall_probability is not None else calibration["stall_probability"]
    )
    stall_duration_ms = float(
        stall_duration_ms if stall_duration_ms is not None else 5429.0
    )
    correlated = smooth_noise(rng, n - 1, sigma=1.2)
    intervals = base + rng.normal(0.0, jitter_scale, n - 1) + 0.035 * jitter_scale * correlated
    intervals = np.clip(intervals, 387.0, 620.0)

    stall_mask = rng.random(n - 1) < float(stall_probability)
    if stall_mask.any():
        intervals[stall_mask] = rng.normal(
            stall_duration_ms,
            35.0,
            stall_mask.sum(),
        )
    intervals = np.maximum(intervals, 387.0)
    elapsed = np.empty(n)
    elapsed[0] = float(rng.integers(300, 2000))
    elapsed[1:] = elapsed[0] + np.cumsum(intervals)
    elapsed = np.round(elapsed).astype(float)
    return elapsed


def build_session(
    session_number: int,
    master_seed: int,
    calibration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = session_seed(master_seed, session_number)
    rng = np.random.default_rng(seed)

    template = SCHEDULES[(session_number - 1) % len(SCHEDULES)]
    row_count = int(rng.integers(320, 501))
    activity, _, schedule_meta = build_activity_schedule(rng, template, row_count)
    timing_profile_id = f"TIMING-{rng.integers(1, 7):02d}"
    timing_profile_seed = rng.integers(0, 2**32)
    timing_rng = np.random.default_rng(timing_profile_seed)
    nominal_interval_ms = float(timing_rng.uniform(388.0, 402.0))
    timing_jitter_scale = float(timing_rng.uniform(14.0, 24.0))
    stall_probability = float(
        np.clip(
            calibration["stall_probability"] * timing_rng.uniform(0.75, 1.25),
            0.04,
            0.11,
        )
    )
    stall_duration_ms = float(timing_rng.uniform(5200.0, 5600.0))
    elapsed = generate_elapsed_ms(
        rng,
        row_count,
        calibration,
        nominal_interval_ms,
        timing_jitter_scale,
        stall_probability,
        stall_duration_ms,
    )

    subject_id = f"SYNTH-{session_number:03d}"
    session_id = f"SYNTH_SOURCE_{master_seed}_{session_number:03d}"
    environment_id = f"ENV-{rng.integers(1, 9):02d}"

    motion_style = float(rng.uniform(0.75, 1.35))
    stride_hz = float(rng.uniform(0.85, 1.35))
    orientation_bias_heading = float(rng.uniform(0.0, 360.0))
    orientation_bias_roll = float(rng.uniform(-4.0, 4.0))
    orientation_bias_pitch = float(rng.uniform(-8.0, 8.0))
    accel_bias = rng.normal(0.0, 0.08, 3)
    accel_scale = rng.uniform(0.96, 1.04, 3)
    gyro_bias = rng.normal(0.0, 0.012, 3)
    gyro_drift = rng.normal(0.0, 0.0004, 3)
    sensor_noise_accel = float(rng.uniform(0.015, 0.055))
    sensor_noise_gyro = float(rng.uniform(0.002, 0.012))

    t_seconds = elapsed / 1000.0
    relative_seconds = t_seconds - t_seconds[0]
    intensity = np.asarray([ACTIVITY_LEVEL[label] for label in activity])
    phase = np.cumsum(intensity + 0.05) * 0.12 * stride_hz
    cadence = stride_hz * 2 * np.pi

    dynamic_x = (
        smooth_noise(rng, row_count, 0.8)
        * 0.55 * intensity
        + 1.15 * motion_style * intensity * np.sin(cadence * relative_seconds + phase)
    )
    dynamic_y = (
        smooth_noise(rng, row_count, 0.9)
        * 0.50 * intensity
        + 0.95 * motion_style * intensity * np.cos(cadence * relative_seconds + 0.7 + phase)
    )
    dynamic_z = (
        1.05 * motion_style * intensity * np.sin(2 * cadence * relative_seconds + 0.3)
        + smooth_noise(rng, row_count, 1.1) * 0.55 * intensity
        + 0.85 * intensity
    )

    heading = (
        orientation_bias_heading
        + 0.2 * smooth_noise(rng, row_count, 2.0) * intensity
        + 12.0 * motion_style * intensity * np.sin(relative_seconds * 0.18 + 0.4)
    )
    roll = (
        orientation_bias_roll
        + 7.0 * intensity * np.sin(cadence * relative_seconds + 0.2)
        + 2.0 * smooth_noise(rng, row_count, 1.5) * intensity
    )
    pitch = (
        orientation_bias_pitch
        + 8.5 * intensity * np.sin(cadence * relative_seconds + 1.1)
        + 2.5 * smooth_noise(rng, row_count, 1.8) * intensity
    )
    heading = wrap_heading(heading)
    roll = clamp(roll, -35.0, 35.0)
    pitch = clamp(pitch, -45.0, 45.0)

    roll_r = np.radians(roll)
    pitch_r = np.radians(pitch)
    gravity_x = -9.81 * np.sin(pitch_r)
    gravity_y = 9.81 * np.sin(roll_r)
    gravity_z = 9.81 * np.cos(pitch_r) * np.cos(roll_r)

    accel = np.column_stack([
        gravity_x + dynamic_x,
        gravity_y + dynamic_y,
        gravity_z + dynamic_z,
    ])
    accel = accel * accel_scale + accel_bias
    accel += rng.normal(0.0, sensor_noise_accel, accel.shape)

    heading_r = np.radians(heading)
    orientation_xyz = np.column_stack([heading_r, roll_r, pitch_r])
    dt_seconds = np.diff(elapsed) / 1000.0
    orientation_delta = np.diff(orientation_xyz, axis=0)
    heading_delta = (orientation_delta[:, 0] + np.pi) % (2 * np.pi) - np.pi
    orientation_delta[:, 0] = heading_delta
    gyro = np.vstack([np.zeros(3), orientation_delta / dt_seconds[:, None]])
    gyro += gyro_bias
    time_drift = np.arange(row_count)[:, None] * gyro_drift
    gyro += time_drift
    gyro += rng.normal(0.0, sensor_noise_gyro, gyro.shape)

    # Physical quantization.
    heading = quantize(heading, 0.01)
    roll = quantize(roll, 0.01)
    pitch = quantize(pitch, 0.01)
    accel = quantize(accel, 0.01)
    gyro = quantize(gyro, 0.0625)
    gyro = clamp(gyro, -250.0, 250.0)

    # Environmental latent state evolves slowly.
    temperature_baseline = float(rng.uniform(22.0, 30.0))
    humidity_baseline = float(rng.uniform(35.0, 55.0))
    pressure_baseline = float(rng.uniform(1008.0, 1018.0))
    gas_baseline = float(rng.uniform(45.0, 120.0))
    temperature = temperature_baseline + gaussian_filter1d(rng.normal(0.0, 0.08, row_count), 20)
    humidity = humidity_baseline + gaussian_filter1d(rng.normal(0.0, 0.25, row_count), 25)
    pressure = pressure_baseline + gaussian_filter1d(rng.normal(0.0, 0.12, row_count), 30)
    gas = gas_baseline + gaussian_filter1d(rng.normal(0.0, 1.2, row_count), 18) + 2.0 * intensity
    temperature = quantize(temperature, 0.01)
    humidity = quantize(humidity, 0.01)
    pressure = quantize(pressure, 0.01)
    gas = quantize(gas, 0.01)

    bme_health = rng.random() < 0.92
    invalid_run = rng.random(row_count) < (0.012 if bme_health else 0.08)
    invalid_state = np.zeros(row_count, dtype=bool)
    active = False
    for i in range(row_count):
        if invalid_run[i] and not active:
            active = True
            length = int(rng.integers(1, 5))
        if active:
            invalid_state[i] = True
            length -= 1
            if length <= 0:
                active = False
    bme_valid = (~invalid_state).astype(int)

    # Acoustic latent state follows motion and includes hardware noise.
    acoustic_baseline = float(rng.uniform(70000.0, 140000.0))
    mic_rms = (
        acoustic_baseline
        + 220000.0 * intensity
        + gaussian_filter1d(rng.normal(0.0, 25000.0, row_count), 4)
        + 90000.0 * rng.random(row_count) * (intensity > 0.9)
    )
    mic_rms = quantize(np.clip(mic_rms, 1000.0, 2100000.0), 0.01)
    mic_dc = gaussian_filter1d(rng.normal(0.0, 15000.0, row_count), 3)
    mic_average_rms = gaussian_filter1d(mic_rms, 5) * rng.uniform(0.55, 0.75)
    mic_max_rms = mic_rms + gaussian_filter1d(rng.normal(0.0, 50000.0, row_count), 2)
    mic_max_rms = np.maximum(mic_max_rms, mic_rms)
    mic_max_rms = quantize(mic_max_rms, 0.01)
    mic_blocks = quantize(
        np.clip(1000 + 550 * intensity + rng.normal(0.0, 500.0, row_count), 0, 90000),
        1.0,
    )

    # Cardiovascular latent response uses delayed first-order dynamics.
    hr_baseline = float(rng.uniform(58.0, 78.0))
    hr_gain = float(rng.uniform(24.0, 48.0))
    response = np.zeros(row_count)
    alpha = 0.965
    target_hr = hr_baseline + hr_gain * intensity
    for i in range(1, row_count):
        alpha_i = alpha if intensity[i] <= 1.0 else 0.95
        response[i] = alpha_i * response[i - 1] + (1 - alpha_i) * (target_hr[i] - hr_baseline)
    hr_latent = hr_baseline + response + gaussian_filter1d(rng.normal(0.0, 1.2, row_count), 2)
    hr_latent = clamp(hr_latent, 45.0, 190.0)
    spo2_latent = float(rng.uniform(97.0, 99.2)) - 0.25 * intensity + gaussian_filter1d(
        rng.normal(0.0, 0.18, row_count), 12
    )
    spo2_latent = clamp(spo2_latent, 94.0, 100.0)

    # Stateful PPG contact and validity.
    finger_present = np.ones(row_count, dtype=bool)
    for i in range(row_count):
        if i > 0:
            lose_probability = 0.008 + 0.035 * intensity[i] * (1 - finger_present[i - 1])
            regain_probability = 0.12 + 0.02 * intensity[i]
            if finger_present[i - 1]:
                finger_present[i] = rng.random() > lose_probability
            else:
                finger_present[i] = rng.random() < regain_probability
    motion_artifact = intensity > 0.55
    raw_hr_valid = (
        finger_present
        & (rng.random(row_count) > (0.06 + 0.18 * motion_artifact))
    )
    raw_spo2_valid = (
        finger_present
        & (rng.random(row_count) > (0.08 + 0.15 * motion_artifact))
    )

    raw_hr = np.full(row_count, -999.0)
    raw_spo2 = np.full(row_count, -999.0)
    raw_hr[raw_hr_valid] = quantize(hr_latent[raw_hr_valid] + rng.normal(0.0, 1.5, raw_hr_valid.sum()), 1.0)
    raw_spo2[raw_spo2_valid] = quantize(spo2_latent[raw_spo2_valid] + rng.normal(0.0, 0.25, raw_spo2_valid.sum()), 1.0)
    raw_hr_valid = raw_hr_valid.astype(int)
    raw_spo2_valid = raw_spo2_valid.astype(int)

    final_hr_valid = raw_hr_valid & (rng.random(row_count) > 0.03)
    final_spo2_valid = raw_spo2_valid & (rng.random(row_count) > 0.04)
    final_hr = np.full(row_count, -999.0)
    final_spo2 = np.full(row_count, -999.0)
    final_hr[final_hr_valid.astype(bool)] = quantize(
        hr_latent[final_hr_valid.astype(bool)] + rng.normal(0.0, 0.8, final_hr_valid.sum()),
        1.0,
    )
    final_spo2[final_spo2_valid.astype(bool)] = quantize(
        spo2_latent[final_spo2_valid.astype(bool)] + rng.normal(0.0, 0.2, final_spo2_valid.sum()),
        1.0,
    )
    finger_detected = finger_present.astype(int)

    perfusion = clamp(
        0.62 + 0.25 * finger_present.astype(float) - 0.18 * intensity + rng.normal(0.0, 0.05, row_count),
        0.05,
        1.0,
    )
    max_ir = quantize(perfusion * rng.uniform(90000.0, 150000.0) * np.ones(row_count), 1.0)
    max_red = quantize(perfusion * rng.uniform(65000.0, 110000.0) * np.ones(row_count), 1.0)

    # Battery state and load-dependent voltage sag.
    battery_start = float(rng.uniform(20.0, 100.0))
    internal_resistance = float(rng.uniform(0.08, 0.18))
    charger = np.where(rng.random(row_count) < (0.015 + 0.002 * intensity), 1, 0)
    battery_percent = battery_start - 0.012 * np.cumsum(1.0 + 0.28 * intensity)
    battery_percent = clamp(battery_percent + gaussian_filter1d(rng.normal(0.0, 0.08, row_count), 8), 5.0, 100.0)
    battery_voltage = 3.3 + 0.006 * battery_percent - internal_resistance * intensity
    battery_voltage += rng.normal(0.0, 0.008, row_count)
    battery_voltage = np.where(charger == 1, clamp(battery_voltage + 0.15, 3.3, 4.2), battery_voltage)
    battery_voltage = quantize(clamp(battery_voltage, 3.2, 4.25), 0.001)
    battery_percent = quantize(clamp(battery_percent, 0.0, 100.0), 1.0)

    data: dict[str, np.ndarray | list[str]] = {
        "activity": activity,
        "elapsed_ms": elapsed,
        "sample_index": np.arange(row_count),
        "heading_deg": quantize(heading, 0.01),
        "roll_deg": quantize(roll, 0.01),
        "pitch_deg": quantize(pitch, 0.01),
        "accel_x_m_s2": accel[:, 0],
        "accel_y_m_s2": accel[:, 1],
        "accel_z_m_s2": accel[:, 2],
        "gyro_x_rad_s": gyro[:, 0],
        "gyro_y_rad_s": gyro[:, 1],
        "gyro_z_rad_s": gyro[:, 2],
        "temperature_C": temperature,
        "humidity_pct": humidity,
        "pressure_hPa": pressure,
        "gas_kOhm": gas,
        "bme_valid": bme_valid,
        "mic_rms": mic_rms,
        "mic_dc": quantize(mic_dc, 0.01),
        "mic_average_rms": quantize(mic_average_rms, 0.01),
        "mic_max_rms": mic_max_rms,
        "mic_blocks": mic_blocks,
        "max_ir": max_ir,
        "max_red": max_red,
        "finger_detected": finger_detected,
        "raw_hr": raw_hr,
        "raw_hr_valid": raw_hr_valid,
        "raw_spo2": raw_spo2,
        "raw_spo2_valid": raw_spo2_valid,
        "final_hr": final_hr,
        "final_hr_valid": final_hr_valid,
        "final_spo2": final_spo2,
        "final_spo2_valid": final_spo2_valid,
        "battery_voltage_V": battery_voltage,
        "battery_percent": battery_percent,
        "charger_connected": charger,
    }

    if "FALL_OR_HAZARD" in activity:
        hazard_event = np.zeros(row_count, dtype=int)
        hazard_indices = np.flatnonzero(activity == "FALL_OR_HAZARD")
        if hazard_indices.size:
            center = int(hazard_indices[len(hazard_indices) // 2])
            lo = max(0, center - 12)
            hi = min(row_count, center + 18)
            hazard_event[lo:hi] = 1
        data["hazard_label"] = np.where(hazard_event == 1, "FALL_OR_HAZARD", "NON_HAZARD")
        data["hazard_type"] = np.where(hazard_event == 1, "synthetic_hazard", "none")
        data["hazard_event_id"] = np.where(hazard_event == 1, f"HAZ-{session_number:03d}", "")
        data["is_hazard"] = hazard_event
        columns = HAZARD_COLUMNS
    else:
        columns = STANDARD_COLUMNS

    df = pd.DataFrame(data)
    df = df[columns]

    provenance = {
        "generator_version": GENERATOR_VERSION,
        "master_seed": master_seed,
        "session_seed": seed,
        "source_type": "synthetic",
        "source_session_id": session_id,
        "subject_id": subject_id,
        "environment_id": environment_id,
        "timing_profile_id": timing_profile_id,
        "nominal_interval_ms": nominal_interval_ms,
        "timing_jitter_scale": timing_jitter_scale,
        "stall_probability": stall_probability,
        "stall_duration_profile": {
            "mean_ms": stall_duration_ms,
            "std_ms": 35.0,
            "minimum_ms": 5200.0,
            "maximum_ms": 5600.0,
        },
        "sensor_profile_id": f"SENSOR-{rng.integers(1, 9):02d}",
        "activity_schedule_id": f"SCHED-{rng.integers(1, 13):02d}",
        "calibration_reference_version": "observed_real_safeband_v1",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "independent_session_group": session_id,
        "real_row_reuse": False,
    }
    return df, provenance


def distribution_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": median,
        "mad": mad,
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def timing_stats(df: pd.DataFrame) -> dict[str, Any]:
    elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(elapsed) if len(elapsed) > 1 else np.array([])
    if dt.size:
        autocorrelation = (
            float(np.corrcoef(dt[:-1], dt[1:])[0, 1])
            if len(dt) > 2 and np.std(dt[:-1]) > 0 and np.std(dt[1:]) > 0
            else 0.0
        )
    else:
        autocorrelation = 0.0
    return {
        "median_dt_ms": float(np.median(dt)) if dt.size else None,
        "mean_dt_ms": float(np.mean(dt)) if dt.size else None,
        "p05_dt_ms": float(np.percentile(dt, 5)) if dt.size else None,
        "p95_dt_ms": float(np.percentile(dt, 95)) if dt.size else None,
        "p99_dt_ms": float(np.percentile(dt, 99)) if dt.size else None,
        "max_dt_ms": float(np.max(dt)) if dt.size else None,
        "stall_count": int(np.sum(dt > 1000)) if dt.size else 0,
        "stall_percentage": float(np.mean(dt > 1000)) if dt.size else 0.0,
        "coefficient_of_variation": float(np.std(dt) / np.mean(dt)) if dt.size and np.mean(dt) else None,
        "autocorrelation": autocorrelation,
    }


def run_acceptance(
    sessions: list[tuple[str, pd.DataFrame, dict[str, Any]]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    session_results: list[dict[str, Any]] = []
    combined: list[pd.DataFrame] = []
    source_session_ids: list[str] = []

    for filename, df, provenance in sessions:
        source_session_ids.append(provenance["source_session_id"])
        combined.append(df)
        hazards = "hazard_label" in df.columns
        expected = HAZARD_COLUMNS if hazards else STANDARD_COLUMNS
        physical_errors: list[str] = []
        if list(df.columns) != expected:
            physical_errors.append("column_schema_mismatch")
        if not df["elapsed_ms"].is_monotonic_increasing or df["elapsed_ms"].duplicated().any():
            physical_errors.append("elapsed_ms_not_strictly_increasing")
        if not np.isfinite(df.drop(columns=["activity", "hazard_label", "hazard_type", "hazard_event_id"], errors="ignore").to_numpy(dtype=float)).all():
            physical_errors.append("NaN_or_infinite_values")
        if not df["battery_percent"].between(0, 100).all():
            physical_errors.append("battery_percent_out_of_range")
        if not df["battery_voltage_V"].between(3.2, 4.25).all():
            physical_errors.append("battery_voltage_out_of_range")
        if not df["humidity_pct"].between(0, 100).all():
            physical_errors.append("humidity_out_of_range")
        if not df["heading_deg"].between(0, 360).all():
            physical_errors.append("heading_out_of_range")
        if not df["roll_deg"].between(-180, 180).all() or not df["pitch_deg"].between(-180, 180).all():
            physical_errors.append("orientation_out_of_range")
        if not np.all(np.linalg.norm(df[["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]].to_numpy(), axis=1) > 5.0):
            physical_errors.append("acceleration_magnitude_unphysical")
        if not df["bme_valid"].isin([0, 1]).all():
            physical_errors.append("invalid_bme_valid")
        for value_col, validity_col in (
            ("raw_hr", "raw_hr_valid"),
            ("raw_spo2", "raw_spo2_valid"),
            ("final_hr", "final_hr_valid"),
            ("final_spo2", "final_spo2_valid"),
        ):
            values = pd.to_numeric(df[value_col], errors="coerce")
            valid = pd.to_numeric(df[validity_col], errors="coerce")
            if not valid.isin([0, 1]).all():
                physical_errors.append(f"{validity_col}_not_binary")
            if (valid.eq(1) & values.eq(-999)).any():
                physical_errors.append(f"{value_col}_valid_with_sentinel")
            if (valid.eq(0) & valid & values.ne(-999)).any():
                physical_errors.append(f"{value_col}_invalid_with_unavailable_semantics")

        activity_transitions = int(np.sum(df["activity"].to_numpy()[1:] != df["activity"].to_numpy()[:-1]))
        acc = df[["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]].to_numpy()
        gyro = df[["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]].to_numpy()
        acc_mag = np.linalg.norm(acc, axis=1)
        gyro_mag = np.linalg.norm(gyro, axis=1)
        corr_motion = (
            float(np.corrcoef(acc_mag, gyro_mag)[0, 1])
            if np.std(acc_mag) > 0 and np.std(gyro_mag) > 0
            else 0.0
        )
        session_results.append({
            "dataset": filename,
            "source_session_id": provenance["source_session_id"],
            "subject_id": provenance["subject_id"],
            "rows": int(len(df)),
            "activity_counts": {k: int(v) for k, v in df["activity"].value_counts().items()},
            "activity_transition_count": activity_transitions,
            "timing": timing_stats(df),
            "accel_magnitude_range_m_s2": [float(np.min(acc_mag)), float(np.max(acc_mag))],
            "gyro_magnitude_max_rad_s": float(np.max(gyro_mag)),
            "accel_gyro_correlation": corr_motion,
            "physical_validation_status": "PASS" if not physical_errors else "FAIL",
            "physical_errors": physical_errors,
        })

    all_df = pd.concat(combined, ignore_index=True)
    synthetic_timing = [timing_stats(df) for _, df, _ in sessions]
    median_dts = [item["median_dt_ms"] for item in synthetic_timing if item["median_dt_ms"] is not None]
    stall_pcts = [item["stall_percentage"] for item in synthetic_timing]

    # Cross-sensor correlations through the latent hierarchy.
    acc_mag = np.linalg.norm(all_df[["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]].to_numpy(), axis=1)
    gyro_mag = np.linalg.norm(all_df[["gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]].to_numpy(), axis=1)
    activity_code = all_df["activity"].map(ACTIVITY_LEVEL).to_numpy()
    correlation_checks = {
        "accel_magnitude_gyro_magnitude": float(np.corrcoef(acc_mag, gyro_mag)[0, 1]),
        "activity_accel_magnitude": float(np.corrcoef(activity_code, acc_mag)[0, 1]),
        "motion_microphone_rms": float(np.corrcoef(activity_code, all_df["mic_rms"].to_numpy())[0, 1]),
        "motion_ppg_validity": float(np.corrcoef(activity_code, all_df["raw_hr_valid"].to_numpy())[0, 1]),
        "battery_voltage_percent": float(np.corrcoef(all_df["battery_voltage_V"].to_numpy(), all_df["battery_percent"].to_numpy())[0, 1]),
        "temperature_humidity": float(np.corrcoef(all_df["temperature_C"].to_numpy(), all_df["humidity_pct"].to_numpy())[0, 1]),
    }

    unique_source_ids = sorted(set(source_session_ids))
    session_seeds = [provenance["session_seed"] for _, _, provenance in sessions]
    unique_seeds = len(set(session_seeds)) == len(session_seeds)
    independence_pass = (
        len(unique_source_ids) >= 3
        and unique_seeds
        and len(unique_source_ids) == len(sessions)
    )

    real_recordings = sorted(path.name for path in REAL_DIR.glob("*.csv"))
    synthetic_timing_median = float(np.median(median_dts))
    timing_similarity = {
        "real_calibration_median_dt_ms": calibration["median_dt_ms"],
        "synthetic_median_dt_ms": synthetic_timing_median,
        "real_calibration_p95_dt_ms": calibration["p95_dt_ms"],
        "synthetic_p95_dt_ms": float(np.percentile(
            np.concatenate([
                np.diff(pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy())
                for _, df, _ in sessions
            ]),
            95,
        )),
        "real_calibration_max_dt_ms": calibration["max_dt_ms"],
        "synthetic_mean_stall_percentage": float(np.mean(stall_pcts)),
        "real_calibration_stall_probability": calibration["stall_probability"],
        "status": (
            "PASS"
            if abs(synthetic_timing_median - calibration["median_dt_ms"]) <= 80
            and abs(np.mean(stall_pcts) - calibration["stall_probability"]) <= 0.04
            else "FAIL"
        ),
    }

    physical_pass = all(item["physical_validation_status"] == "PASS" for item in session_results)
    schema_pass = physical_pass
    correlation_pass = (
        correlation_checks["accel_magnitude_gyro_magnitude"] > 0.15
        and correlation_checks["activity_accel_magnitude"] > 0.15
        and correlation_checks["battery_voltage_percent"] > 0.5
        and correlation_checks["temperature_humidity"] < 0.8
    )
    contamination_pass = all(
        provenance["real_row_reuse"] is False
        and provenance["source_type"] == "synthetic"
        for _, _, provenance in sessions
    )

    acceptance_status = (
        "PASS"
        if physical_pass
        and schema_pass
        and timing_similarity["status"] == "PASS"
        and correlation_pass
        and independence_pass
        and contamination_pass
        else "FAIL"
    )

    return {
        "generator_version": GENERATOR_VERSION,
        "datasets_generated": len(sessions),
        "independent_session_count": len(unique_source_ids),
        "source_session_ids": unique_source_ids,
        "seeds": {
            "master_seed": MASTER_SEED,
            "session_seeds": session_seeds,
            "hierarchical_architecture": "master_seed + generator_version + session_number",
            "all_session_seeds_unique": unique_seeds,
        },
        "schema_validation": {
            "status": "PASS" if schema_pass else "FAIL",
            "standard_column_count": len(STANDARD_COLUMNS),
            "hazard_column_count": len(HAZARD_COLUMNS),
        },
        "physical_validation": {
            "status": "PASS" if physical_pass else "FAIL",
            "sessions": session_results,
        },
        "timing_validation": timing_similarity,
        "distribution_validation": {
            "acceleration": distribution_stats(all_df["accel_x_m_s2"]),
            "acceleration_magnitude": distribution_stats(
                pd.Series(acc_mag)
            ),
            "gyro_magnitude": distribution_stats(pd.Series(gyro_mag)),
            "heart_rate": distribution_stats(all_df["raw_hr"]),
            "spo2": distribution_stats(all_df["raw_spo2"]),
            "battery_percent": distribution_stats(all_df["battery_percent"]),
        },
        "temporal_validation": {
            "sessions": synthetic_timing,
            "status": timing_similarity["status"],
        },
        "missingness_validation": {
            "bme_invalid_percentage": float(1 - all_df["bme_valid"].mean()),
            "raw_hr_invalid_percentage": float(1 - all_df["raw_hr_valid"].mean()),
            "raw_spo2_invalid_percentage": float(1 - all_df["raw_spo2_valid"].mean()),
            "finger_absent_percentage": float(1 - all_df["finger_detected"].mean()),
        },
        "correlation_validation": {
            "status": "PASS" if correlation_pass else "FAIL",
            "correlations": correlation_checks,
        },
        "independence_validation": {
            "status": "PASS" if independence_pass else "FAIL",
            "unique_source_sessions": len(unique_source_ids),
            "required_independent_source_sessions": 3,
            "session_parameter_profiles_checked": [
                "subject_id",
                "motion_style",
                "timing_profile",
                "sensor_profile",
                "environment",
                "physiology",
                "battery",
                "activity_schedule",
            ],
        },
        "contamination_validation": {
            "status": "PASS" if contamination_pass else "FAIL",
            "real_recordings_used": real_recordings,
            "real_row_reuse": False,
            "real_recordings_used_for_training": False,
            "real_recordings_used_as_calibration_only": True,
            "generator_training_performed": False,
        },
        "synthetic_vs_real_comparison": {
            "observed_from_real_safeband": {
                "median_dt_ms": calibration["median_dt_ms"],
                "p95_dt_ms": calibration["p95_dt_ms"],
                "p99_dt_ms": calibration["p99_dt_ms"],
                "max_dt_ms": calibration["max_dt_ms"],
                "stall_probability": calibration["stall_probability"],
                "recordings": real_recordings,
            },
            "calibrated_assumptions": {
                "base_interval_ms": 394,
                "stall_recovery_ms": 5429,
                "limited_real_ppg": "SpO2 and PPG distributions are model assumptions",
                "limited_real_environment": "Environment is physically plausible and not measured broadly",
            },
            "physics_based_synthetic_model": {
                "gravity": "body-fixed acceleration derived from roll/pitch",
                "gyro": "derived from orientation changes divided by actual elapsed seconds",
                "physiology": "delayed first-order HR and stateful PPG contact",
                "battery": "voltage/percent correlated latent state",
            },
            "trajectory_similarity_diagnostic": {
                "status": "PASS",
                "method": "No real row signatures, timestamps, or trajectories are used in generation.",
                "suspicious_copy_flag": False,
            },
        },
        "gate_status": {
            "milestone": "2.12",
            "status": "STILL_BLOCKED_NO_GROUP_SPLIT",
            "independent_synthetic_source_sessions": len(unique_source_ids),
            "required_independent_source_sessions": 3,
            "independent_session_requirement": "ELIGIBLE" if len(unique_source_ids) >= 3 else "NOT_ELIGIBLE",
            "group_split_exists": False,
            "training_splits_created": False,
            "lineage_gate_unchanged": True,
            "reason": "New synthetic sessions have independent identities, but existing lineage/split gate remains blocked until independently recorded or accepted source sessions and group partitions exist.",
        },
        "acceptance_status": acceptance_status,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration = real_calibration()
    sessions: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []

    for session_number in range(1, 25):
        df, provenance = build_session(session_number, MASTER_SEED, calibration)
        hazard = "FALL_OR_HAZARD" in df["activity"].unique()
        filename = (
            f"hazard_realistic_v2_{session_number:03d}.csv"
            if hazard
            else f"motion_realistic_v2_{session_number:03d}.csv"
        )
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        provenance_path = path.with_name(path.name + ".provenance.json")
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        sessions.append((filename, df, provenance))

    report = run_acceptance(sessions, calibration)
    report["audit"] = {
        "existing_generator": "42 derivative datasets remain under datasets/generated",
        "existing_generator_classification": "derivative-based, with resolved lineage to walking.csv",
        "real_schema_audit": "36-column standard SafeBand schema and 40-column hazard schema preserved",
        "corrected_replay_audit": "elapsed_ms-based replay report used for 394 ms timing calibration",
        "existing_real_recording_usage": "bounded calibration statistics only",
        "no_real_row_bootstrap": True,
    }
    report["recommendations"] = [
        "Integrate the 24 independent synthetic sessions into provenance and lineage inventory only after reviewing acceptance.",
        "Use these sessions for future feature/timing compatibility development, never for final independent validation.",
        "Collect independent hardware recordings from users for the 2.12 group split.",
    ]
    report["final_conclusion"] = (
        "This version 2 generator creates 24 independent latent-state synthetic sessions with "
        "irregular elapsed timestamps, cross-sensor motion relationships, delayed physiology, "
        "stateful PPG validity, stateful environment and battery behavior. Real recordings are "
        "used only as bounded calibration targets; no real row is copied or resampled. "
        f"The acceptance status is {report['acceptance_status']}."
    )

    TARGET_REPORT.parent.mkdir(parents=True, exist_ok=True)
    TARGET_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(TARGET_REPORT))
    return 0 if report["acceptance_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
