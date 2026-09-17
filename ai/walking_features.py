"""Walking-focused PPG + motion feature extractor for SafeBand.

Keeps the established 8 s / 2 s protocol and the V4-style feature philosophy,
but uses the native single wrist PPG channel in the PhysioNet walking dataset.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram
from scipy.stats import skew, kurtosis

def _clean(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def _norm(x):
    x = _clean(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = np.std(x)
    if scale < 1e-9:
        scale = 1.0
    return (x - med) / scale

def _bp(x, fs, low=0.5, high=5.0):
    x = _clean(x)
    nyq = fs / 2.0
    high = min(high, nyq * 0.9)
    if not (0 < low < high < nyq):
        return _norm(x)
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return _norm(x)
    return filtfilt(b, a, x)

def _basic(x, p):
    x = _clean(x)
    return {
        p+"_mean": float(np.mean(x)),
        p+"_std": float(np.std(x)),
        p+"_rms": float(np.sqrt(np.mean(x*x))),
        p+"_ptp": float(np.ptp(x)),
        p+"_skew": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        p+"_kurtosis": float(kurtosis(x, bias=False)) if len(x) > 3 else 0.0,
    }

def _spec(x, fs, p):
    f, power = periodogram(_clean(x), fs=fs, detrend="constant", scaling="density")
    m = (f >= 0.5) & (f <= 5.0)
    if not np.any(m):
        return {p+"_dom_bpm": 0.0, p+"_spec_centroid": 0.0, p+"_band_power": 0.0}
    fb, pb = f[m], power[m]
    den = np.sum(pb) + 1e-12
    return {
        p+"_dom_bpm": float(fb[np.argmax(pb)] * 60.0),
        p+"_spec_centroid": float(np.sum(fb * pb) / den),
        p+"_band_power": float(np.trapezoid(pb, fb)),
    }

def _ppg(x, fs):
    z = _norm(_bp(x, fs))
    out = {}
    out.update(_basic(z, "ppg"))
    out.update(_spec(z, fs, "ppg"))
    distance = max(1, int(round(fs * 60.0 / 220.0)))
    peaks, props = find_peaks(z, distance=distance, prominence=0.15)
    out["ppg_peak_count"] = float(len(peaks))
    out["ppg_peak_hr"] = 0.0
    out["ppg_peak_hr_iqr"] = 0.0
    out["ppg_prominence_median"] = 0.0
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        hr = 60.0 / rr
        hr = hr[(hr >= 30.0) & (hr <= 220.0)]
        if hr.size:
            out["ppg_peak_hr"] = float(np.median(hr))
            out["ppg_peak_hr_iqr"] = float(np.percentile(hr, 75) - np.percentile(hr, 25))
            prom = props.get("prominences", [])
            if len(prom):
                out["ppg_prominence_median"] = float(np.median(prom))
    dc = np.median(_clean(x))
    ac = np.std(_bp(x, fs))
    out["ppg_ac_dc"] = float(ac / (abs(dc) + 1e-9))
    return out, z

def _motion(data, fs, p):
    data = _clean(data)
    out = {}
    for i, axis in enumerate(("x", "y", "z")):
        out.update(_basic(data[:, i], p+"_"+axis))
    mag = np.sqrt(np.sum(data*data, axis=1))
    out.update(_basic(mag, p+"_mag"))
    out.update(_spec(mag, fs, p+"_mag"))
    return out, mag

def _corr(a, b):
    a, b = _clean(a), _clean(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0

def extract_features(ppg, acc, gyro, ppg_fs=256.0, motion_fs=256.0):
    ppg = _clean(ppg).reshape(-1)
    acc, gyro = _clean(acc), _clean(gyro)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"ACC shape must be (N,3), got {acc.shape}")
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError(f"GYRO shape must be (N,3), got {gyro.shape}")
    if not (len(ppg) == len(acc) == len(gyro)):
        raise ValueError("PPG/ACC/GYRO lengths differ")
    out, ppg_bp = _ppg(ppg, ppg_fs)
    acc_out, acc_mag = _motion(acc, motion_fs, "acc")
    gyro_out, gyro_mag = _motion(gyro, motion_fs, "gyro")
    out.update(acc_out)
    out.update(gyro_out)
    out["corr_ppg_accmag"] = _corr(ppg_bp, acc_mag)
    out["corr_ppg_gyromag"] = _corr(ppg_bp, gyro_mag)
    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in out.items()}
