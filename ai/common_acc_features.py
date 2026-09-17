"""Common accelerometer feature contract for SafeBand activity cross-domain experiments.

The contract intentionally mirrors the ACC-only feature family already present in the
BITS-2 activity_windows.csv artifact. FORTH-TRACE raw ACC is resampled to 20 Hz and
windowed at 40 samples (2 s) before these features are calculated.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = [
    "ax_mean","ax_std","ax_min","ax_max","ax_rms",
    "ay_mean","ay_std","ay_min","ay_max","ay_rms",
    "az_mean","az_std","az_min","az_max","az_rms",
    "acc_mag_mean","acc_mag_std","acc_mag_min","acc_mag_max","acc_mag_rms",
    "acc_mag_p10","acc_mag_p25","acc_mag_p50","acc_mag_p75","acc_mag_p90",
    "acc_sma","acc_range","acc_jerk_mean","acc_jerk_std","acc_jerk_max",
    "acc_diff_energy","acc_zero_crossings",
    "acc_x_y_corr","acc_x_z_corr","acc_y_z_corr",
    "acc_fft_low","acc_fft_mid","acc_fft_high",
]

def _safe_std(x):
    return float(np.std(x, ddof=0))

def _rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def _corr(a,b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a,b)[0,1])

def _band_energy(x, fs=20.0, lo=0.3, hi=10.0):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 4:
        return 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= lo) & (freqs < hi)
    total = float(np.sum(spec[(freqs >= lo) & (freqs < hi)]))
    return total

def extract_acc_features(acc_xyz, fs=20.0):
    a = np.asarray(acc_xyz, dtype=float)
    if a.shape[0] < 4 or a.shape[1] != 3:
        raise ValueError("acc_xyz must have shape (N, 3) with N >= 4")
    ax, ay, az = a.T
    mag = np.sqrt(ax*ax + ay*ay + az*az)
    diff = np.diff(mag)
    jerk = diff * fs

    row = {}
    for name, x in (("ax",ax),("ay",ay),("az",az)):
        row[f"{name}_mean"] = float(np.mean(x))
        row[f"{name}_std"] = _safe_std(x)
        row[f"{name}_min"] = float(np.min(x))
        row[f"{name}_max"] = float(np.max(x))
        row[f"{name}_rms"] = _rms(x)

    row.update({
        "acc_mag_mean": float(np.mean(mag)),
        "acc_mag_std": _safe_std(mag),
        "acc_mag_min": float(np.min(mag)),
        "acc_mag_max": float(np.max(mag)),
        "acc_mag_rms": _rms(mag),
        "acc_mag_p10": float(np.percentile(mag,10)),
        "acc_mag_p25": float(np.percentile(mag,25)),
        "acc_mag_p50": float(np.percentile(mag,50)),
        "acc_mag_p75": float(np.percentile(mag,75)),
        "acc_mag_p90": float(np.percentile(mag,90)),
        "acc_sma": float(np.mean(np.abs(ax)+np.abs(ay)+np.abs(az))),
        "acc_range": float(np.max(mag)-np.min(mag)),
        "acc_jerk_mean": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        "acc_jerk_std": _safe_std(jerk) if len(jerk) else 0.0,
        "acc_jerk_max": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
        "acc_diff_energy": float(np.mean(np.square(diff))) if len(diff) else 0.0,
        "acc_zero_crossings": float(np.sum(np.diff(np.sign(diff)) != 0)) if len(diff) else 0.0,
        "acc_x_y_corr": _corr(ax,ay),
        "acc_x_z_corr": _corr(ax,az),
        "acc_y_z_corr": _corr(ay,az),
        "acc_fft_low": _band_energy(mag,fs,0.3,2.0),
        "acc_fft_mid": _band_energy(mag,fs,2.0,5.0),
        "acc_fft_high": _band_energy(mag,fs,5.0,10.0),
    })
    return row

def feature_frame_from_windows(windows, labels, subjects, source, starts=None):
    rows=[]
    for i,w in enumerate(windows):
        r=extract_acc_features(w,fs=20.0)
        r["activity_label"]=labels[i]
        r["subject_id"]=str(subjects[i])
        r["source"]=source
        if starts is not None:
            r["window_start"]=int(starts[i])
        rows.append(r)
    return pd.DataFrame(rows, columns=FEATURES+["activity_label","subject_id","source"]+([] if starts is None else ["window_start"]))
