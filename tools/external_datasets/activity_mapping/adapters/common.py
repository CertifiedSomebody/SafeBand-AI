from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class Recording:
    dataset_id: str
    subject_id: str
    session_id: str
    source_label: str
    timestamps_s: np.ndarray
    accel_xyz_m_s2: np.ndarray
    sample_rate_hz: float
    source_unit: str
    timestamp_source_unit: str
    source_path: str
    source_path_resolution: str = "direct"


def validate_recording(recording: Recording) -> None:
    timestamps = np.asarray(recording.timestamps_s, dtype=np.float64)
    accel = np.asarray(recording.accel_xyz_m_s2, dtype=np.float64)
    if timestamps.ndim != 1 or accel.shape != (len(timestamps), 3):
        raise ValueError(f"{recording.session_id}: invalid time/ACC array shapes")
    if len(timestamps) < 2:
        raise ValueError(f"{recording.session_id}: fewer than two samples")
    if not np.isfinite(timestamps).all() or not np.isfinite(accel).all():
        raise ValueError(f"{recording.session_id}: non-finite timestamps or ACC values")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{recording.session_id}: timestamps must strictly increase")
    if recording.sample_rate_hz <= 0:
        raise ValueError(f"{recording.session_id}: sample rate must be positive")


def contiguous_segments(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    *,
    max_gap_s: float,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")
    cuts = np.flatnonzero(np.diff(timestamps_s) > max_gap_s) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(timestamps_s)]))
    for start, end in zip(starts, ends):
        if end - start >= 2:
            yield timestamps_s[start:end], values[start:end]


def resample_segment(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    if len(timestamps_s) < 2 or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("source timestamps must strictly increase")
    count = int(np.floor((timestamps_s[-1] - timestamps_s[0]) * target_hz)) + 1
    grid = timestamps_s[0] + np.arange(count, dtype=np.float64) / target_hz
    if len(grid) < 2:
        return grid, np.empty((len(grid), 3), dtype=np.float64)
    output = np.column_stack(
        [np.interp(grid, timestamps_s, values[:, axis]) for axis in range(3)]
    )
    return grid, output


def safe_band_activity(dataset_id: str, source_label: str) -> tuple[str, str]:
    label = source_label.strip()
    lowered = label.lower()
    if dataset_id == "aicos_har":
        if lowered in {"standing", "sitting", "lyingdown"}:
            return "STATIC", label.upper()
        if lowered in {
            "walking", "upstairs", "downstairs", "rampup", "rampdown",
        }:
            return "WALKING", label.upper()
        if lowered == "running":
            return "RUNNING", label.upper()
        if lowered in {"liftup", "liftdown", "elevatorup", "elevatordown"}:
            return "UNMAPPED", label.upper()
    elif dataset_id == "sisfall":
        code = label.upper()
        if code in {"D01", "D02", "D05", "D06"}:
            return "WALKING", code
        if code in {"D03", "D04"}:
            return "RUNNING", code
        if code in {f"D{i:02d}" for i in range(7, 18)}:
            return "TRANSITION", code
        if code in {"D18", "D19"}:
            return "HIGH_MOTION", code
        if code.startswith("F"):
            return "FALL_OR_HAZARD", code
    return "UNKNOWN", label.upper()


def window_features(
    timestamps_s: np.ndarray,
    accel_xyz_m_s2: np.ndarray,
    *,
    rolling_mean: np.ndarray,
    rolling_std: np.ndarray,
    rolling_max: np.ndarray,
) -> dict[str, float]:
    dt = np.diff(timestamps_s)
    if len(dt) == 0 or np.any(dt <= 0):
        raise ValueError("window timestamps must strictly increase")
    magnitudes = np.linalg.norm(accel_xyz_m_s2, axis=1)
    jerk = np.diff(magnitudes) / dt
    interval_ms = dt * 1000.0
    return {
        "feature__accel_magnitude_mean_m_s2": float(np.mean(magnitudes)),
        "feature__accel_magnitude_std_m_s2": float(np.std(magnitudes)),
        "feature__accel_magnitude_max_m_s2": float(np.max(magnitudes)),
        "feature__accel_magnitude_range_m_s2": float(np.ptp(magnitudes)),
        "feature__accel_jerk_abs_mean_m_s3": float(np.mean(np.abs(jerk))),
        "feature__accel_jerk_abs_max_m_s3": float(np.max(np.abs(jerk))),
        "feature__rolling_accel_mean_1s_m_s2": float(np.mean(rolling_mean)),
        "feature__rolling_accel_std_1s_m_s2": float(np.mean(rolling_std)),
        "feature__rolling_accel_max_1s_m_s2": float(np.mean(rolling_max)),
        "feature__timing_interval_ms_mean": float(np.mean(interval_ms)),
        "feature__timing_interval_ms_std": float(np.std(interval_ms)),
        "feature__timing_interval_ms_max": float(np.max(interval_ms)),
        "feature__accel_available_fraction": 1.0,
        "feature__gyro_available_fraction": 0.0,
    }


def rolling_magnitude_statistics(
    timestamps_s: np.ndarray,
    magnitudes: np.ndarray,
    *,
    duration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration_s <= 0:
        raise ValueError("rolling duration must be positive")
    prefix = np.concatenate(([0.0], np.cumsum(magnitudes, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(magnitudes**2, dtype=np.float64)))
    starts = np.empty(len(timestamps_s), dtype=np.int64)
    left = 0
    for right, timestamp in enumerate(timestamps_s):
        while left < right and timestamp - timestamps_s[left] > duration_s:
            left += 1
        starts[right] = left
    ends = np.arange(1, len(timestamps_s) + 1, dtype=np.int64)
    counts = ends - starts
    means = (prefix[ends] - prefix[starts]) / counts
    variances = (prefix_sq[ends] - prefix_sq[starts]) / counts - means**2
    stds = np.sqrt(np.maximum(variances, 0.0))

    maxima = np.empty(len(magnitudes), dtype=np.float64)
    from collections import deque

    queue: deque[int] = deque()
    left = 0
    for right, timestamp in enumerate(timestamps_s):
        while left < right and timestamp - timestamps_s[left] > duration_s:
            if queue and queue[0] == left:
                queue.popleft()
            left += 1
        while queue and magnitudes[queue[-1]] <= magnitudes[right]:
            queue.pop()
        queue.append(right)
        maxima[right] = magnitudes[queue[0]]
    return means, stds, maxima
