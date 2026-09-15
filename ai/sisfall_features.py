"""SisFall -> SafeBand V6 feature compatibility layer.

IMPORTANT:
- BITS-2's canonical accelerometer values are preserved exactly as released by
  the BITS-2 ingestion adapter. The V6 model was trained on that numeric scale.
- Therefore SisFall is converted to *g*, not m/s^2, before feature extraction.
- SisFall raw acquisition is 200 Hz; it is reduced to 20 Hz with non-overlapping
  10-sample block means so the V6 window/FFT contract remains 20 Hz.
- No gyro features are mixed into the accelerometer-only BITS-2 V6 model.
"""

from __future__ import annotations
import math
from typing import Dict, Sequence

BASE_FEATURES = [
    "ax_mean","ax_std","ax_min","ax_max","ax_rms",
    "ay_mean","ay_std","ay_min","ay_max","ay_rms",
    "az_mean","az_std","az_min","az_max","az_rms",
    "mag_mean","mag_std","mag_min","mag_max","mag_rms",
    "mag_p10","mag_p25","mag_p50","mag_p75","mag_p90",
    "mag_sma","mag_range","jerk_mean","jerk_std","jerk_max",
    "diff_energy","mag_zero_crossings","xy_corr","xz_corr","yz_corr",
    "fft_low","fft_mid","fft_high",
]
EVENT_FEATURES = [
    "pre_peak_mean","peak_magnitude","post_peak_mean","peak_to_baseline",
    "peak_index_ratio","pre_post_change","post_peak_std","peak_width",
    "high_energy_fraction","recovery_ratio",
]
V6_FEATURES = [
    "v6_mag_mean","v6_mag_std","v6_mag_min","v6_mag_max","v6_mag_rms",
    "v6_mag_p10","v6_mag_p25","v6_mag_p50","v6_mag_p75","v6_mag_p90",
    "v6_peak_prominence","v6_peak_to_rms","v6_peak_to_median",
    "v6_peak_index_ratio","v6_peak_width_half","v6_high_fraction",
    "v6_pre_mean","v6_post_mean","v6_pre_std","v6_post_std",
    "v6_pre_post_ratio","v6_pre_post_delta","v6_post_quiet_ratio",
    "v6_pre_energy","v6_post_energy","v6_post_energy_ratio",
    "v6_jerk_abs_mean","v6_jerk_abs_max","v6_jerk_std",
    "v6_jerk_sign_changes","v6_diff_energy","v6_impulse_area",
    "v6_axis_peak_ratio","v6_axis_std_ratio","v6_tilt_change",
    "v6_spectral_entropy",
]
FEATURE_COLUMNS = BASE_FEATURES + EVENT_FEATURES + V6_FEATURES

G0 = 9.80665
# SisFall published sensor specs:
# ADXL345: +/-16 g, 13-bit
# MMA8451Q: +/-8 g, 14-bit
ADXL345_G_PER_COUNT = (2.0 * 16.0) / (2 ** 13)
MMA8451Q_G_PER_COUNT = (2.0 * 8.0) / (2 ** 14)


def adxl_counts_to_g(v: float) -> float:
    """Convert SisFall ADXL345 counts to g."""
    return float(v) * ADXL345_G_PER_COUNT


def mma_counts_to_g(v: float) -> float:
    """Convert SisFall MMA8451Q counts to g."""
    return float(v) * MMA8451Q_G_PER_COUNT


def downsample_200_to_20(samples):
    """Block-average 200 Hz samples into 20 Hz samples."""
    out = []
    n = len(samples) - (len(samples) % 10)
    for i in range(0, n, 10):
        block = samples[i:i + 10]
        out.append(tuple(sum(row[j] for row in block) / 10.0 for j in range(3)))
    return out


def _mean(v): return sum(v) / len(v) if v else 0.0

def _std(v):
    if not v:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))

def _rms(v):
    return math.sqrt(_mean([x * x for x in v])) if v else 0.0

def _pct(v, q):
    if not v:
        return 0.0
    a = sorted(v)
    p = (len(a) - 1) * q
    lo, hi = int(math.floor(p)), int(math.ceil(p))
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (p - lo)

def _corr(a, b):
    if len(a) < 2:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    den = math.sqrt(sum(x * x for x in da) * sum(x * x for x in db))
    return 0.0 if den <= 1e-12 else sum(x * y for x, y in zip(da, db)) / den

def _zero_crossings(v):
    if len(v) < 2:
        return 0
    m = _mean(v)
    c = [x - m for x in v]
    return sum(1 for a, b in zip(c, c[1:]) if (a < 0 <= b) or (a > 0 >= b))

def _fft_bands(v, sample_rate=20.0):
    n = len(v)
    if n < 4:
        return 0.0, 0.0, 0.0
    x = [z - _mean(v) for z in v]
    low = mid = high = 0.0
    for k in range(1, n // 2 + 1):
        re = sum(x[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        im = -sum(x[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        p = (re * re + im * im) / (n * n)
        hz = sample_rate * k / n
        if hz <= 2:
            low += p
        elif hz <= 5:
            mid += p
        else:
            high += p
    return low, mid, high

def _spectral_entropy(v):
    n = len(v)
    if n < 8:
        return 0.0
    x = [z - _mean(v) for z in v]
    powers = []
    for k in range(1, n // 2 + 1):
        re = sum(x[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        im = -sum(x[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        powers.append((re * re + im * im) / (n * n))
    total = sum(powers)
    probs = [p / total for p in powers if p > 0] if total > 1e-12 else []
    return (
        -sum(p * math.log(p) for p in probs) / math.log(len(probs))
        if len(probs) > 1 else 0.0
    )

def extract_features(samples):
    """Return the exact 84-column V6 contract for 20 Hz accelerometer data in g."""
    if len(samples) < 4:
        return {k: 0.0 for k in FEATURE_COLUMNS}

    ax = [float(s[0]) for s in samples]
    ay = [float(s[1]) for s in samples]
    az = [float(s[2]) for s in samples]
    mag = [math.sqrt(x*x + y*y + z*z) for x, y, z in zip(ax, ay, az)]

    def five(v):
        return _mean(v), _std(v), min(v), max(v), _rms(v)

    axm, axs, axmin, axmax, axr = five(ax)
    aym, ays, aymin, aymax, ayr = five(ay)
    azm, azs, azmin, azmax, azr = five(az)
    mm, ms, mmin, mmax, mr = five(mag)

    jerk = [mag[i] - mag[i - 1] for i in range(1, len(mag))]
    low, mid, high = _fft_bands(mag)

    base = {
        "ax_mean": axm, "ax_std": axs, "ax_min": axmin, "ax_max": axmax, "ax_rms": axr,
        "ay_mean": aym, "ay_std": ays, "ay_min": aymin, "ay_max": aymax, "ay_rms": ayr,
        "az_mean": azm, "az_std": azs, "az_min": azmin, "az_max": azmax, "az_rms": azr,
        "mag_mean": mm, "mag_std": ms, "mag_min": mmin, "mag_max": mmax, "mag_rms": mr,
        "mag_p10": _pct(mag, .1), "mag_p25": _pct(mag, .25), "mag_p50": _pct(mag, .5),
        "mag_p75": _pct(mag, .75), "mag_p90": _pct(mag, .9),
        "mag_sma": sum(abs(x) + abs(y) + abs(z) for x, y, z in zip(ax, ay, az)) / len(ax),
        "mag_range": mmax - mmin, "jerk_mean": _mean(jerk), "jerk_std": _std(jerk),
        "jerk_max": max(jerk) if jerk else 0.0, "diff_energy": _mean([x*x for x in jerk]),
        "mag_zero_crossings": float(_zero_crossings(mag)),
        "xy_corr": _corr(ax, ay), "xz_corr": _corr(ax, az), "yz_corr": _corr(ay, az),
        "fft_low": low, "fft_mid": mid, "fft_high": high,
    }

    n = len(mag)
    peak = max(mag)
    pi = mag.index(peak)
    pre = mag[:max(1, pi)]
    post = mag[min(n - 1, pi + 1):]
    pre_m = _mean(pre)
    post_m = _mean(post)
    baseline = (pre_m + post_m) / 2.0
    delta = max(0.0, peak - baseline)
    width = sum(1 for x in mag if x >= baseline + 0.5 * delta)
    high_n = sum(1 for x in mag if x >= baseline + 0.75 * delta)

    base.update({
        "pre_peak_mean": pre_m, "peak_magnitude": peak, "post_peak_mean": post_m,
        "peak_to_baseline": peak / (baseline + 1e-9),
        "peak_index_ratio": pi / max(1, n - 1),
        "pre_post_change": abs(post_m - pre_m) / (baseline + 1e-9),
        "post_peak_std": _std(post),
        "peak_width": float(width), "high_energy_fraction": high_n / n,
        "recovery_ratio": post_m / (pre_m + 1e-9),
    })

    rms = math.sqrt(_mean([x*x for x in mag]))
    med = _pct(mag, .5)
    axis_max = max(
        max(abs(x) for x in ax),
        max(abs(x) for x in ay),
        max(abs(x) for x in az),
    )
    axis_stds = [_std(ax), _std(ay), _std(az)]
    cut = max(1, n // 4)
    early = (_mean(ax[:cut]), _mean(ay[:cut]), _mean(az[:cut]))
    late = (_mean(ax[-cut:]), _mean(ay[-cut:]), _mean(az[-cut:]))

    def tilt(v):
        return math.atan2(v[1], math.sqrt(v[0]*v[0] + v[2]*v[2]))

    v6base = max(
        1e-9,
        (_mean(mag[:max(1, pi)]) + _mean(mag[min(n - 1, pi + 1):])) / 2
    )
    half = v6base + 0.5 * (peak - v6base)
    high_thr = v6base + 0.75 * (peak - v6base)
    pre_e = _mean([x*x for x in pre])
    post_e = _mean([x*x for x in post])

    base.update({
        "v6_mag_mean": mm, "v6_mag_std": ms, "v6_mag_min": mmin,
        "v6_mag_max": mmax, "v6_mag_rms": rms,
        "v6_mag_p10": _pct(mag, .1), "v6_mag_p25": _pct(mag, .25),
        "v6_mag_p50": med, "v6_mag_p75": _pct(mag, .75), "v6_mag_p90": _pct(mag, .9),
        "v6_peak_prominence": (peak - v6base) / (v6base + 1e-9),
        "v6_peak_to_rms": peak / (rms + 1e-9),
        "v6_peak_to_median": peak / (med + 1e-9),
        "v6_peak_index_ratio": pi / max(1, n - 1),
        "v6_peak_width_half": float(sum(1 for x in mag if x >= half)),
        "v6_high_fraction": sum(1 for x in mag if x >= high_thr) / n,
        "v6_pre_mean": pre_m, "v6_post_mean": post_m,
        "v6_pre_std": _std(pre), "v6_post_std": _std(post),
        "v6_pre_post_ratio": post_m / (pre_m + 1e-9),
        "v6_pre_post_delta": abs(post_m - pre_m) / (v6base + 1e-9),
        "v6_post_quiet_ratio": 1.0 / (1.0 + _std(post)),
        "v6_pre_energy": pre_e, "v6_post_energy": post_e,
        "v6_post_energy_ratio": post_e / (pre_e + 1e-9),
        "v6_jerk_abs_mean": _mean([abs(x) for x in jerk]),
        "v6_jerk_abs_max": max([abs(x) for x in jerk] or [0.0]),
        "v6_jerk_std": _std(jerk),
        "v6_jerk_sign_changes": float(sum(
            1 for a, b in zip(jerk, jerk[1:])
            if (a < 0 <= b) or (a > 0 >= b)
        )),
        "v6_diff_energy": _mean([x*x for x in jerk]),
        "v6_impulse_area": sum(max(0.0, x - v6base) for x in mag),
        "v6_axis_peak_ratio": axis_max / (rms + 1e-9),
        "v6_axis_std_ratio": max(axis_stds) / (sum(axis_stds) + 1e-9),
        "v6_tilt_change": abs(tilt(late) - tilt(early)),
        "v6_spectral_entropy": _spectral_entropy(mag),
    })
    return {k: float(base.get(k, 0.0)) for k in FEATURE_COLUMNS}
