"""Enhanced, sensor-agnostic PPG + motion feature extraction.

The representation intentionally avoids raw-amplitude dependence so that the
later MAX30102 adapter can normalize its own optical signal before using the
same feature contract.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram, welch


def _clean(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 32 or not np.isfinite(x).all():
        raise ValueError("Signal is too short or contains NaN/Inf.")
    x = x - np.median(x)
    s = np.std(x)
    if s < 1e-10:
        raise ValueError("Signal has near-zero variance.")
    return x / s


def _bandpass(x, fs, lo=0.6, hi=4.0):
    nyq = fs / 2.0
    if not (0 < lo < hi < nyq):
        raise ValueError(f"Invalid band [{lo}, {hi}] for fs={fs}.")
    b, a = butter(3, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x)


def _safe_corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-10 or sb < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spectral_features(x, fs):
    # Zero padding gives a denser frequency grid; it does not create new
    # information, but improves peak localization between FFT bins.
    nfft = max(4096, 2 ** int(np.ceil(np.log2(len(x))) + 3))
    f, p = periodogram(x, fs=fs, window="hann", detrend="linear",
                       nfft=nfft, scaling="spectrum")
    band = (f >= 0.6) & (f <= 4.0)
    fb, pb = f[band], p[band]
    if pb.size == 0 or np.sum(pb) <= 0:
        return {"dom_bpm": np.nan, "spec_entropy": np.nan,
                "fund_power_frac": np.nan, "harmonic_ratio": np.nan,
                "low_high_ratio": np.nan}

    k = int(np.argmax(pb))
    dom = float(fb[k] * 60.0)
    prob = pb / np.sum(pb)
    entropy = float(-np.sum(prob * np.log(prob + 1e-12)) /
                    np.log(len(prob)))

    total = float(np.sum(pb)) + 1e-12
    fund = float(pb[k]) / total

    # Energy near the first two harmonics of the dominant component.
    f0 = fb[k]
    h2 = np.abs(fb - 2*f0) <= 0.12
    harmonic_ratio = float(np.sum(pb[h2]) / (pb[k] + 1e-12))

    low = np.sum(pb[(fb >= 0.6) & (fb < 1.5)])
    high = np.sum(pb[(fb >= 1.5) & (fb <= 4.0)])
    return {
        "dom_bpm": dom,
        "spec_entropy": entropy,
        "fund_power_frac": fund,
        "harmonic_ratio": harmonic_ratio,
        "low_high_ratio": float(low / (high + 1e-12)),
    }


def _autocorr_hr(x, fs):
    # Autocorrelation lag search gives sub-FFT-bin temporal periodicity.
    lo_lag = max(1, int(fs * 60 / 180))
    hi_lag = min(len(x)-1, int(fs * 60 / 42))
    if hi_lag <= lo_lag:
        return np.nan, np.nan
    y = x - np.mean(x)
    ac = np.correlate(y, y, mode="full")[len(y)-1:]
    ac0 = ac[0]
    if ac0 <= 0:
        return np.nan, np.nan
    ac = ac / ac0
    region = ac[lo_lag:hi_lag+1]
    if region.size == 0:
        return np.nan, np.nan
    j = int(np.argmax(region))
    lag = lo_lag + j
    # Parabolic interpolation around the autocorrelation maximum.
    frac = 0.0
    if 0 < lag < len(ac)-1:
        y1, y2, y3 = ac[lag-1], ac[lag], ac[lag+1]
        den = y1 - 2*y2 + y3
        if abs(den) > 1e-12:
            frac = 0.5 * (y1 - y3) / den
            frac = float(np.clip(frac, -0.5, 0.5))
    period = (lag + frac) / fs
    return float(60.0 / period), float(ac[lag])


def _peak_features(x, fs):
    # x is normalized filtered PPG. Multiple quality-aware statistics are
    # retained rather than relying on one fragile peak estimate.
    min_dist = max(1, int(fs * 60 / 180))
    prom = max(0.12 * np.std(x), 0.08)
    peaks, props = find_peaks(x, distance=min_dist, prominence=prom)

    out = {
        "peak_hr": np.nan,
        "peak_count": float(len(peaks)),
        "peak_prom_median": np.nan,
        "ibi_std": np.nan,
        "ibi_median": np.nan,
        "peak_regularity": np.nan,
    }
    if len(peaks) < 2:
        return out

    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi >= 60/180) & (ibi <= 60/42)]
    if len(ibi) == 0:
        return out

    med = float(np.median(ibi))
    mad = float(np.median(np.abs(ibi - med)))
    out["peak_hr"] = 60.0 / med
    out["ibi_std"] = float(np.std(ibi))
    out["ibi_median"] = med
    out["peak_regularity"] = float(1.0 / (1.0 + mad))
    if "prominences" in props and len(props["prominences"]):
        out["peak_prom_median"] = float(np.median(props["prominences"]))
    return out


def extract_ppg_features(bvp, fs):
    raw = np.asarray(bvp, dtype=np.float64).reshape(-1)
    x = _clean(raw)
    f = _bandpass(x, fs, 0.6, 4.0)

    sf = _spectral_features(f, fs)
    ac_hr, ac_strength = _autocorr_hr(f, fs)
    pf = _peak_features(f, fs)

    # Morphology / quality statistics.
    dx = np.diff(f)
    rms = np.sqrt(np.mean(f*f))
    q = {
        "ppg_rms": float(rms),
        "ppg_ptp": float(np.ptp(f)),
        "ppg_iqr": float(np.percentile(f, 75) - np.percentile(f, 25)),
        "ppg_skew_proxy": float(np.mean(f**3)),
        "ppg_kurt_proxy": float(np.mean(f**4)),
        "ppg_diff_abs_mean": float(np.mean(np.abs(dx))),
        "ppg_diff_std": float(np.std(dx)),
        "autocorr_hr": ac_hr,
        "autocorr_strength": ac_strength,
    }
    q.update(sf)
    q.update(pf)

    # Cross-check candidate HRs. These are model inputs, not ground truth.
    candidates = np.array(
        [sf["dom_bpm"], ac_hr, pf["peak_hr"]], dtype=float
    )
    valid = candidates[np.isfinite(candidates)]
    q["candidate_hr_median"] = float(np.median(valid)) if len(valid) else np.nan
    q["candidate_hr_spread"] = float(np.std(valid)) if len(valid) > 1 else 99.0
    return q


def extract_acc_features(acc, fs):
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise ValueError(f"ACC must be finite shape (N,3), got {a.shape}")

    mag = np.linalg.norm(a, axis=1)
    dm = np.diff(mag, prepend=mag[:1])
    # Remove the DC/gravity component for motion-energy measures.
    mag_c = mag - np.median(mag)

    return {
        "acc_mag_mean": float(np.mean(mag)),
        "acc_mag_std": float(np.std(mag)),
        "acc_mag_rms": float(np.sqrt(np.mean(mag*mag))),
        "acc_mag_range": float(np.ptp(mag)),
        "acc_dynamic_std": float(np.std(mag_c)),
        "acc_diff_abs_mean": float(np.mean(np.abs(dm))),
        "acc_diff_std": float(np.std(dm)),
        "acc_dynamic_energy": float(np.mean(mag_c*mag_c)),
        "acc_fs": float(fs),
    }


def extract_features(bvp, acc, bvp_fs, acc_fs):
    out = extract_ppg_features(bvp, bvp_fs)
    out.update(extract_acc_features(acc, acc_fs))
    return out
