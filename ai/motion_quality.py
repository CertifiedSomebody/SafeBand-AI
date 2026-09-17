"""Wrist-accelerometer motion quality features for PPG HR estimation."""
from __future__ import annotations
import numpy as np


def extract_motion_quality(acc, fs):
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"ACC must be finite (N,3), got {a.shape}")
    mag = np.linalg.norm(a, axis=1)
    centered = mag - np.median(mag)
    d = np.diff(mag, prepend=mag[:1])
    # Motion energy is used as a quality/context feature, not as a proxy target.
    return {
        "motion_rms": float(np.sqrt(np.mean(centered**2))),
        "motion_std": float(np.std(centered)),
        "motion_range": float(np.ptp(mag)),
        "motion_diff_rms": float(np.sqrt(np.mean(d*d))),
        "motion_diff_abs_mean": float(np.mean(np.abs(d))),
        "motion_fs": float(fs),
    }
