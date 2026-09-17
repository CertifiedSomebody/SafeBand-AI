"""Robust PPG -> HR candidate extraction.

All estimators return candidates plus quality evidence. They do not know the
ECG ground truth and are suitable for later replacement by a MAX30102 input
adapter.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram


HR_MIN, HR_MAX = 42.0, 180.0


def clean_ppg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(x) < 128 or not np.isfinite(x).all():
        raise ValueError("PPG must be finite and contain at least 128 samples.")
    # Robust centering/scaling prevents optical DC/amplitude from dominating.
    x = x - np.median(x)
    scale = np.median(np.abs(x)) * 1.4826
    if scale < 1e-10:
        scale = np.std(x)
    if scale < 1e-10:
        raise ValueError("PPG has near-zero variation.")
    return x / scale


def bandpass(x, fs, lo=0.6, hi=4.0):
    ny = fs / 2.0
    if not 0 < lo < hi < ny:
        raise ValueError(f"Invalid band [{lo}, {hi}] at {fs} Hz.")
    b, a = butter(3, [lo/ny, hi/ny], btype="band")
    return filtfilt(b, a, x)


def spectral_candidate(x, fs):
    nfft = max(4096, 2 ** int(np.ceil(np.log2(len(x))) + 3))
    f, p = periodogram(x, fs=fs, window="hann", detrend="linear",
                       nfft=nfft, scaling="spectrum")
    m = (f >= HR_MIN/60) & (f <= HR_MAX/60)
    f, p = f[m], p[m]
    if len(p) < 3 or not np.isfinite(p).all() or p.sum() <= 0:
        return np.nan, 0.0, np.nan
    k = int(np.argmax(p))
    # Quadratic interpolation around the spectral maximum.
    delta = 0.0
    if 0 < k < len(p)-1:
        den = p[k-1] - 2*p[k] + p[k+1]
        if abs(den) > 1e-14:
            delta = float(np.clip(0.5*(p[k-1]-p[k+1])/den, -0.5, 0.5))
    df = f[1]-f[0]
    hz = f[k] + delta*df
    # Local peak prominence proxy.
    lo = max(0, k-2); hi = min(len(p), k+3)
    baseline = np.median(np.delete(p[lo:hi], min(2, k-lo))) if hi-lo > 2 else np.median(p)
    quality = float(max(0.0, (p[k]-baseline)/(np.sum(p)+1e-12)))
    return float(hz*60), quality, float(p[k]/(p.sum()+1e-12))


def autocorr_candidate(x, fs):
    lo = max(1, int(fs*60/HR_MAX))
    hi = min(len(x)-2, int(fs*60/HR_MIN))
    if hi <= lo:
        return np.nan, 0.0
    y = x - np.mean(x)
    ac = np.correlate(y, y, mode="full")[len(y)-1:]
    if ac[0] <= 0:
        return np.nan, 0.0
    ac /= ac[0]
    r = ac[lo:hi+1]
    k = lo + int(np.argmax(r))
    delta = 0.0
    if 0 < k < len(ac)-1:
        den = ac[k-1]-2*ac[k]+ac[k+1]
        if abs(den) > 1e-12:
            delta = float(np.clip(0.5*(ac[k-1]-ac[k+1])/den, -0.5, 0.5))
    lag = k + delta
    return float(60*fs/lag), float(np.clip(ac[k], 0, 1))


def peak_candidate(x, fs):
    distance = max(1, int(fs*60/HR_MAX))
    peaks, props = find_peaks(x, distance=distance,
                              prominence=max(0.08, 0.08*np.std(x)))
    if len(peaks) < 2:
        return np.nan, 0.0, np.nan, np.nan
    ibi = np.diff(peaks)/fs
    valid = (ibi >= 60/HR_MAX) & (ibi <= 60/HR_MIN)
    ibi = ibi[valid]
    if len(ibi) == 0:
        return np.nan, 0.0, np.nan, np.nan
    med = np.median(ibi)
    mad = np.median(np.abs(ibi-med))
    regularity = float(1/(1+mad))
    hr = float(60/med)
    prom = float(np.median(props["prominences"])) if len(props["prominences"]) else np.nan
    return hr, regularity, float(np.std(ibi)), prom


def extract_candidates(ppg, fs):
    raw = clean_ppg(ppg)
    filt = bandpass(raw, fs)
    sh, sq, sp = spectral_candidate(filt, fs)
    ah, aq = autocorr_candidate(filt, fs)
    ph, pq, ibi_std, prom = peak_candidate(filt, fs)
    vals = np.array([sh, ah, ph], dtype=float)
    good = vals[np.isfinite(vals)]
    median = float(np.median(good)) if len(good) else np.nan
    spread = float(np.std(good)) if len(good) > 1 else 99.0

    # Motion-independent signal quality: candidate agreement is deliberately
    # separated from raw amplitude, which is not portable to MAX30102.
    agreement = float(np.exp(-spread/12.0)) if np.isfinite(median) else 0.0
    return {
        "spectral_hr": sh, "spectral_quality": sq, "spectral_fraction": sp,
        "autocorr_hr": ah, "autocorr_quality": aq,
        "peak_hr": ph, "peak_regularity": pq, "ibi_std": ibi_std,
        "peak_prominence": prom, "candidate_median": median,
        "candidate_spread": spread, "candidate_agreement": agreement,
        "filtered": filt,
    }
