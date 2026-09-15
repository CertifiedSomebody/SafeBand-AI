"""SafeBand activity-model feature extraction.

The first trainable SafeBand activity model is intentionally motion-first.
BITS-2 provides wrist accelerometer data at the right modality for this task;
physiology/environment/audio are kept out of this first model rather than
inventing synchronization between unlike sampling rates.

Feature extraction is deterministic and shared by training and runtime.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np

# Keep this ordered and stable: it becomes part of the serialized model contract.
FEATURE_COLUMNS = [
    "ax_mean", "ax_std", "ax_min", "ax_max", "ax_rms",
    "ay_mean", "ay_std", "ay_min", "ay_max", "ay_rms",
    "az_mean", "az_std", "az_min", "az_max", "az_rms",
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_mag_rms",
    "acc_mag_p10", "acc_mag_p25", "acc_mag_p50", "acc_mag_p75", "acc_mag_p90",
    "acc_sma", "acc_range", "acc_jerk_mean", "acc_jerk_std", "acc_jerk_max",
    "acc_diff_energy", "acc_zero_crossings",
    "acc_x_y_corr", "acc_x_z_corr", "acc_y_z_corr",
    "acc_fft_low", "acc_fft_mid", "acc_fft_high",
]


def _number(sample: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = sample.get(key, default)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _zero_crossings(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    centered = x - np.mean(x)
    return float(np.count_nonzero(centered[:-1] * centered[1:] < 0))


def _fft_band_energy(magnitude: np.ndarray) -> tuple[float, float, float]:
    """Return normalized low/mid/high spectral energy.

    The exact physical frequency depends on the dataset's documented sample
    rate. BITS-2 motion is treated as approximately 20 Hz, while the feature
    itself is deliberately expressed as relative band energy.
    """
    n = len(magnitude)
    if n < 4:
        return 0.0, 0.0, 0.0
    x = magnitude - np.mean(magnitude)
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / 20.0)
    total = float(np.sum(power[1:]))
    if total <= 1e-12:
        return 0.0, 0.0, 0.0
    low = float(np.sum(power[(freqs >= 0.5) & (freqs < 3.0)])) / total
    mid = float(np.sum(power[(freqs >= 3.0) & (freqs < 7.0)])) / total
    high = float(np.sum(power[freqs >= 7.0])) / total
    return low, mid, high


def extract_features(window: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract the fixed SafeBand motion feature vector from one window."""
    if not window:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    ax = np.asarray([_number(s, "acceleration_x") for s in window], dtype=float)
    ay = np.asarray([_number(s, "acceleration_y") for s in window], dtype=float)
    az = np.asarray([_number(s, "acceleration_z") for s in window], dtype=float)
    mag = np.sqrt(ax * ax + ay * ay + az * az)

    def axis_stats(x: np.ndarray) -> tuple[float, float, float, float, float]:
        return float(np.mean(x)), _safe_std(x), float(np.min(x)), float(np.max(x)), _rms(x)

    axm, axs, axn, axx, axr = axis_stats(ax)
    aym, ays, ayn, ayx, ayr = axis_stats(ay)
    azm, azs, azn, azx, azr = axis_stats(az)
    mm, ms, mn, mx, mr = axis_stats(mag)

    if len(mag) > 1:
        diff = np.diff(mag)
        jerk_mean = float(np.mean(np.abs(diff)))
        jerk_std = _safe_std(diff)
        jerk_max = float(np.max(np.abs(diff)))
        diff_energy = float(np.mean(diff * diff))
    else:
        jerk_mean = jerk_std = jerk_max = diff_energy = 0.0

    low, mid, high = _fft_band_energy(mag)

    values = {
        "ax_mean": axm, "ax_std": axs, "ax_min": axn, "ax_max": axx, "ax_rms": axr,
        "ay_mean": aym, "ay_std": ays, "ay_min": ayn, "ay_max": ayx, "ay_rms": ayr,
        "az_mean": azm, "az_std": azs, "az_min": azn, "az_max": azx, "az_rms": azr,
        "acc_mag_mean": mm, "acc_mag_std": ms, "acc_mag_min": mn, "acc_mag_max": mx, "acc_mag_rms": mr,
        "acc_mag_p10": float(np.percentile(mag, 10)),
        "acc_mag_p25": float(np.percentile(mag, 25)),
        "acc_mag_p50": float(np.percentile(mag, 50)),
        "acc_mag_p75": float(np.percentile(mag, 75)),
        "acc_mag_p90": float(np.percentile(mag, 90)),
        "acc_sma": float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az))),
        "acc_range": mx - mn,
        "acc_jerk_mean": jerk_mean,
        "acc_jerk_std": jerk_std,
        "acc_jerk_max": jerk_max,
        "acc_diff_energy": diff_energy,
        "acc_zero_crossings": _zero_crossings(mag),
        "acc_x_y_corr": _corr(ax, ay),
        "acc_x_z_corr": _corr(ax, az),
        "acc_y_z_corr": _corr(ay, az),
        "acc_fft_low": low,
        "acc_fft_mid": mid,
        "acc_fft_high": high,
    }
    return {name: float(values[name]) for name in FEATURE_COLUMNS}


__all__ = ["FEATURE_COLUMNS", "extract_features"]
