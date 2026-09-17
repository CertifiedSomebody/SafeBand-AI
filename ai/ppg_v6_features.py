"""SafeBand PPG V6 feature extraction.

Signal-quality-aware, motion-aware HR features for the PPG reference pipeline.
The representation is intentionally sensor-agnostic enough to bridge the
MAX30101-family PTT reference toward later MAX30102 recordings.

Boundaries:
- ECG waveform is never used as an input feature.
- ECG R-peak annotations provide the HR regression target in the reference data.
- SpO2 is not trained here.
- This module does not constitute MAX30102 hardware validation.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import butter, find_peaks, periodogram, sosfiltfilt
from scipy.stats import kurtosis, skew

PPG_CHANNELS = (
    "distal_red", "distal_ir", "distal_green",
    "proximal_red", "proximal_ir", "proximal_green",
)
MOTION_CHANNELS = ("x", "y", "z")

FS_DEFAULT = 500.0
LOW_HZ = 0.5
HIGH_HZ = 5.0
HR_MIN = 35.0
HR_MAX = 220.0


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if not np.isfinite(x).all():
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    return x


def _norm(x: np.ndarray) -> np.ndarray:
    x = _clean(x)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(x))
    if scale < 1e-9:
        scale = 1.0
    return (x - med) / scale


def bandpass(x: np.ndarray, fs: float = FS_DEFAULT,
             low: float = LOW_HZ, high: float = HIGH_HZ) -> np.ndarray:
    """Third-order zero-phase physiological bandpass with short-window fallback."""
    x = _clean(x)
    nyq = float(fs) / 2.0
    high = min(float(high), nyq * 0.90)
    if not (0.0 < low < high < nyq) or len(x) < 32:
        return _norm(x)
    sos = butter(3, [low / nyq, high / nyq], btype="band", output="sos")
    padlen = min(len(x) - 1, 3 * (2 * len(sos) + 1))
    if padlen < 8:
        return _norm(x)
    return sosfiltfilt(sos, x, padlen=padlen)


def _integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _safe_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    p = p[np.isfinite(p) & (p >= 0)]
    if p.size < 2:
        return 0.0
    s = float(p.sum())
    if s <= 0.0:
        return 0.0
    q = p / s
    return float(-np.sum(q * np.log(q + 1e-12)) / np.log(len(q)))


def _basic(y: np.ndarray, prefix: str) -> dict[str, float]:
    y = _clean(y)
    return {
        f"{prefix}_mean": float(np.mean(y)),
        f"{prefix}_std": float(np.std(y)),
        f"{prefix}_rms": float(np.sqrt(np.mean(y * y))),
        f"{prefix}_ptp": float(np.ptp(y)),
        f"{prefix}_p10": float(np.percentile(y, 10)),
        f"{prefix}_p25": float(np.percentile(y, 25)),
        f"{prefix}_p50": float(np.percentile(y, 50)),
        f"{prefix}_p75": float(np.percentile(y, 75)),
        f"{prefix}_p90": float(np.percentile(y, 90)),
        f"{prefix}_skew": float(skew(y, bias=False)) if len(y) > 2 else 0.0,
        f"{prefix}_kurtosis": float(kurtosis(y, bias=False)) if len(y) > 3 else 0.0,
    }


def _spectrum(y: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    """Extract HR-band spectral shape; dominant frequency uses local interpolation."""
    y = _clean(y)
    f, p = periodogram(y, fs=fs, detrend="constant", scaling="density")
    mask = (f >= LOW_HZ) & (f <= HIGH_HZ)
    if not np.any(mask):
        return {f"{prefix}_{k}": 0.0 for k in (
            "dom_bpm", "centroid_hz", "band_power", "entropy", "peak_ratio",
            "peak_snr_db", "harmonic_ratio", "low_ratio", "mid_ratio", "high_ratio",
        )}

    fb = f[mask]
    pb = np.maximum(p[mask], 0.0)
    sum_power = float(pb.sum()) + 1e-12
    band_power = _integral(pb, fb) + 1e-12
    dom_i = int(np.argmax(pb))
    dom_hz = float(fb[dom_i])

    # Parabolic interpolation around the periodogram maximum gives a smoother
    # HR candidate than a raw FFT-bin lookup without using the target label.
    if 0 < dom_i < len(pb) - 1:
        y0, y1, y2 = np.log(pb[dom_i - 1] + 1e-18), np.log(pb[dom_i] + 1e-18), np.log(pb[dom_i + 1] + 1e-18)
        den = y0 - 2.0 * y1 + y2
        if abs(den) > 1e-12:
            delta = 0.5 * (y0 - y2) / den
            delta = float(np.clip(delta, -0.5, 0.5))
            dom_hz += delta * float(fb[1] - fb[0])
    dom_hz = float(np.clip(dom_hz, LOW_HZ, HIGH_HZ))

    centroid = float(np.sum(fb * pb) / sum_power)
    peak_ratio = float(pb[dom_i] / sum_power)
    noise = float(np.median(pb) + 1e-12)
    peak_snr_db = float(10.0 * np.log10((pb[dom_i] + 1e-12) / noise))

    harmonic_mask = np.abs(fb - 2.0 * dom_hz) <= max(0.10 * dom_hz, float(fb[1] - fb[0]))
    half_mask = np.abs(fb - 0.5 * dom_hz) <= max(0.08 * dom_hz, float(fb[1] - fb[0]))
    harmonic = max(
        float(pb[harmonic_mask].max()) if np.any(harmonic_mask) else 0.0,
        float(pb[half_mask].max()) if np.any(half_mask) else 0.0,
    )
    harmonic_ratio = float(harmonic / (pb[dom_i] + 1e-12))

    low_m = (fb >= 0.5) & (fb < 1.0)
    mid_m = (fb >= 1.0) & (fb < 2.0)
    high_m = (fb >= 2.0) & (fb <= 5.0)
    low = _integral(pb[low_m], fb[low_m]) if np.count_nonzero(low_m) >= 2 else 0.0
    mid = _integral(pb[mid_m], fb[mid_m]) if np.count_nonzero(mid_m) >= 2 else 0.0
    high = _integral(pb[high_m], fb[high_m]) if np.count_nonzero(high_m) >= 2 else 0.0

    return {
        f"{prefix}_dom_bpm": float(dom_hz * 60.0),
        f"{prefix}_centroid_hz": centroid,
        f"{prefix}_band_power": band_power,
        f"{prefix}_entropy": _safe_entropy(pb),
        f"{prefix}_peak_ratio": peak_ratio,
        f"{prefix}_peak_snr_db": peak_snr_db,
        f"{prefix}_harmonic_ratio": harmonic_ratio,
        f"{prefix}_low_ratio": float(low / band_power),
        f"{prefix}_mid_ratio": float(mid / band_power),
        f"{prefix}_high_ratio": float(high / band_power),
    }


def _autocorr_hr(y: np.ndarray, fs: float) -> tuple[float, float]:
    """FFT autocorrelation, avoiding O(N^2) np.correlate on long windows."""
    y = _clean(y) - float(np.mean(y))
    n = len(y)
    sd = float(np.std(y))
    if n < 8 or sd < 1e-9:
        return 0.0, 0.0
    y = y / sd
    size = 1 << int(np.ceil(np.log2(max(2, 2 * n - 1))))
    spec = np.fft.rfft(y, size)
    ac = np.fft.irfft(spec * np.conj(spec), size)[:n]
    ac /= max(float(ac[0]), 1e-12)
    lo = max(1, int(fs * 60.0 / HR_MAX))
    hi = min(n - 1, int(fs * 60.0 / HR_MIN))
    if hi <= lo:
        return 0.0, 0.0
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    return float(60.0 * fs / lag), float(np.clip(ac[lag], -1.0, 1.0))


def _peak_features(y: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    y = _clean(y)
    distance = max(1, int(round(fs * 60.0 / HR_MAX)))
    prom = max(0.05, 0.10 * float(np.std(y)))
    peaks, props = find_peaks(y, distance=distance, prominence=prom)
    out = {
        f"{prefix}_peak_count": float(len(peaks)),
        f"{prefix}_peak_hr": 0.0,
        f"{prefix}_peak_hr_iqr": 0.0,
        f"{prefix}_peak_prom_median": 0.0,
        f"{prefix}_rr_cv": 0.0,
        f"{prefix}_pulse_amp_median": 0.0,
    }
    if len(peaks) < 2:
        return out
    rr = np.diff(peaks).astype(float) / float(fs)
    hr = 60.0 / np.maximum(rr, 1e-9)
    valid = (hr >= HR_MIN) & (hr <= HR_MAX)
    if np.any(valid):
        valid_rr = rr[valid]
        valid_hr = hr[valid]
        out[f"{prefix}_peak_hr"] = float(np.median(valid_hr))
        out[f"{prefix}_peak_hr_iqr"] = float(np.percentile(valid_hr, 75) - np.percentile(valid_hr, 25))
        out[f"{prefix}_rr_cv"] = float(np.std(valid_rr) / (np.mean(valid_rr) + 1e-12))
    prominences = np.asarray(props.get("prominences", []), dtype=float)
    if prominences.size:
        out[f"{prefix}_peak_prom_median"] = float(np.median(prominences))
    if len(peaks) >= 3:
        out[f"{prefix}_pulse_amp_median"] = float(np.median(y[peaks]))
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _clean(a), _clean(b)
    if len(a) != len(b) or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0


def _channel(raw: np.ndarray, filtered: np.ndarray, fs: float, name: str):
    out: dict[str, float] = {}
    out.update(_basic(filtered, name))
    out.update(_spectrum(filtered, fs, name))
    out.update(_peak_features(filtered, fs, name))
    ac_hr, ac_peak = _autocorr_hr(filtered, fs)
    out[f"{name}_autocorr_hr"] = float(ac_hr)
    out[f"{name}_autocorr_peak"] = float(ac_peak)

    dc = float(np.median(raw))
    out[f"{name}_ac_dc"] = float(np.std(filtered) / (abs(dc) + 1e-9))

    spectral = float(np.clip(out[f"{name}_peak_ratio"] * 10.0, 0.0, 1.0))
    periodic = float(np.clip(ac_peak, 0.0, 1.0))
    peak_hr = out[f"{name}_peak_hr"]
    dom_hr = out[f"{name}_dom_bpm"]
    agreement = 0.0 if not (peak_hr and dom_hr) else max(0.0, 1.0 - abs(peak_hr - dom_hr) / 40.0)
    rr_quality = float(np.clip(1.0 - out[f"{name}_rr_cv"], 0.0, 1.0)) if out[f"{name}_rr_cv"] else 0.0
    sqi = 0.35 * periodic + 0.30 * spectral + 0.20 * agreement + 0.15 * rr_quality
    out[f"{name}_sqi"] = float(np.clip(sqi, 0.0, 1.0))

    candidates = [v for v in (dom_hr, peak_hr, ac_hr) if HR_MIN <= v <= HR_MAX]
    candidate = float(np.median(candidates)) if candidates else 0.0
    spread = float(np.std(candidates)) if len(candidates) > 1 else 0.0
    out[f"{name}_candidate_hr"] = candidate
    out[f"{name}_candidate_spread"] = spread
    return out, float(out[f"{name}_sqi"]), candidate


def _motion_features(data: np.ndarray, fs: float, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, axis in enumerate(MOTION_CHANNELS):
        out.update(_basic(data[:, i], f"{prefix}_{axis}"))
        out.update(_spectrum(data[:, i], fs, f"{prefix}_{axis}"))
    mag = np.linalg.norm(data, axis=1)
    out.update(_basic(mag, f"{prefix}_mag"))
    out.update(_spectrum(mag, fs, f"{prefix}_mag"))
    return out


def extract_features(ppg: np.ndarray, acc: np.ndarray, gyro: np.ndarray,
                     ppg_fs: float = FS_DEFAULT, motion_fs: float = FS_DEFAULT) -> dict[str, float]:
    """Extract V6 features from one fixed-length synchronized window."""
    ppg = np.asarray(ppg, dtype=float)
    acc = np.asarray(acc, dtype=float)
    gyro = np.asarray(gyro, dtype=float)
    if ppg.ndim != 2 or ppg.shape[1] != 6:
        raise ValueError(f"PPG shape must be (N,6), got {ppg.shape}")
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError(f"ACC shape must be (N,3), got {acc.shape}")
    if gyro.ndim != 2 or gyro.shape[1] != 3:
        raise ValueError(f"GYRO shape must be (N,3), got {gyro.shape}")
    if not (len(ppg) == len(acc) == len(gyro)):
        raise ValueError("PPG/ACC/GYRO lengths differ")

    # Filter each PPG channel exactly once.  V4/V6 initial implementation
    # repeatedly filtered the same channel during correlation extraction;
    # caching makes dataset preparation substantially cheaper and deterministic.
    ppg_raw = [_clean(ppg[:, i]) for i in range(6)]
    ppg_bp = [bandpass(x, ppg_fs) for x in ppg_raw]
    acc_clean = np.asarray([_clean(acc[:, i]) for i in range(3)]).T
    gyro_clean = np.asarray([_clean(gyro[:, i]) for i in range(3)]).T

    out: dict[str, float] = {}
    sqis, candidates = [], []
    for i, name in enumerate(PPG_CHANNELS):
        f, sqi, cand = _channel(ppg_raw[i], ppg_bp[i], ppg_fs, name)
        out.update(f)
        sqis.append(sqi)
        if HR_MIN <= cand <= HR_MAX:
            candidates.append((cand, sqi, name))

    # Morphology agreement across all six channels.  The full 6x6 set is useful
    # for detecting whether multiple optical paths agree on the same pulse rate.
    for i in range(6):
        for j in range(i + 1, 6):
            out[f"corr_{PPG_CHANNELS[i]}_{PPG_CHANNELS[j]}"] = _corr(ppg_bp[i], ppg_bp[j])

    if candidates:
        vals = np.asarray([v[0] for v in candidates], dtype=float)
        weights = np.asarray([max(v[1], 1e-3) for v in candidates], dtype=float)
        order = np.argsort(vals)
        vals, weights = vals[order], weights[order]
        cdf = np.cumsum(weights) / np.sum(weights)
        consensus = float(vals[np.searchsorted(cdf, 0.5)])
        spread = float(np.sqrt(np.average((vals - consensus) ** 2, weights=weights)))
        best_idx = int(np.argmax([v[1] for v in candidates]))
        best_sqi = float(candidates[best_idx][1])
        best_name = candidates[best_idx][2]
    else:
        consensus, spread, best_sqi, best_name = 0.0, 0.0, 0.0, "none"

    out["consensus_hr"] = consensus
    out["consensus_spread_bpm"] = spread
    out["best_channel_sqi"] = best_sqi
    out["mean_channel_sqi"] = float(np.mean(sqis)) if sqis else 0.0
    out["max_channel_sqi"] = float(np.max(sqis)) if sqis else 0.0
    out["channel_sqi_std"] = float(np.std(sqis)) if sqis else 0.0
    out["best_channel_index"] = float(PPG_CHANNELS.index(best_name)) if best_name in PPG_CHANNELS else -1.0

    out.update(_motion_features(acc_clean, motion_fs, "acc"))
    out.update(_motion_features(gyro_clean, motion_fs, "gyro"))
    acc_mag = np.linalg.norm(acc_clean, axis=1)
    gyro_mag = np.linalg.norm(gyro_clean, axis=1)
    out["motion_acc_mag_std_norm"] = float(np.std(acc_mag) / (np.mean(np.abs(acc_mag)) + 1e-9))
    out["motion_gyro_mag_std_norm"] = float(np.std(gyro_mag) / (np.mean(np.abs(gyro_mag)) + 1e-9))

    for i, name in enumerate(PPG_CHANNELS):
        out[f"corr_{name}_accmag"] = _corr(ppg_bp[i], acc_mag)
        out[f"corr_{name}_gyromag"] = _corr(ppg_bp[i], gyro_mag)
        out[f"abs_corr_{name}_accmag"] = abs(out[f"corr_{name}_accmag"])
        out[f"abs_corr_{name}_gyromag"] = abs(out[f"corr_{name}_gyromag"])

    return {k: float(v) if np.isfinite(v) else 0.0 for k, v in out.items()}
