"""
SafeBand AI - BITS-2 Motion Features v2

Feature extraction is deliberately accelerometer-first for BITS-2.
The released BITS-2 timestamps do not provide reliable sub-second
cross-sensor synchronization, so this module never fabricates alignment.

The same deterministic feature contract is used during training and
runtime inference.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

FEATURE_COLUMNS = [
    "ax_mean", "ax_std", "ax_min", "ax_max", "ax_rms",
    "ay_mean", "ay_std", "ay_min", "ay_max", "ay_rms",
    "az_mean", "az_std", "az_min", "az_max", "az_rms",
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_rms",
    "mag_p10", "mag_p25", "mag_p50", "mag_p75", "mag_p90",
    "mag_sma", "mag_range",
    "jerk_mean", "jerk_std", "jerk_max",
    "diff_energy", "mag_zero_crossings",
    "xy_corr", "xz_corr", "yz_corr",
    "fft_low", "fft_mid", "fft_high",
]

def _stats(v: Sequence[float]) -> Tuple[float, float, float, float, float]:
    if not v:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    n = len(v)
    m = sum(v) / n
    var = sum((x - m) ** 2 for x in v) / n
    return m, math.sqrt(var), min(v), max(v), math.sqrt(sum(x*x for x in v)/n)

def _percentile(v: Sequence[float], q: float) -> float:
    if not v:
        return 0.0
    a = sorted(v)
    pos = (len(a)-1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return a[lo]
    return a[lo] + (a[hi]-a[lo]) * (pos-lo)

def _corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2:
        return 0.0
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    da = [x-ma for x in a]
    db = [x-mb for x in b]
    den = math.sqrt(sum(x*x for x in da) * sum(x*x for x in db))
    return 0.0 if den <= 1e-12 else sum(x*y for x,y in zip(da,db))/den

def _zero_crossings(v: Sequence[float]) -> int:
    if len(v) < 2:
        return 0
    m = sum(v)/len(v)
    centered = [x-m for x in v]
    return sum(1 for a,b in zip(centered, centered[1:]) if (a < 0 <= b) or (a > 0 >= b))

def _fft_band_energy(v: Sequence[float]) -> Tuple[float,float,float]:
    # Pure-Python DFT keeps the feature contract dependency-light.
    # For a 40-60 sample window this is small enough for preprocessing.
    n = len(v)
    if n < 4:
        return 0.0, 0.0, 0.0
    mean = sum(v)/n
    x = [z-mean for z in v]
    powers = []
    for k in range(1, n//2 + 1):
        re = sum(x[t]*math.cos(2*math.pi*k*t/n) for t in range(n))
        im = -sum(x[t]*math.sin(2*math.pi*k*t/n) for t in range(n))
        powers.append((k, (re*re+im*im)/(n*n)))
    # Bands are relative to the 20 Hz BITS-2 acquisition rate:
    # low <= 2 Hz, mid 2-5 Hz, high > 5 Hz.
    low = mid = high = 0.0
    for k,p in powers:
        hz = 20.0*k/n
        if hz <= 2.0:
            low += p
        elif hz <= 5.0:
            mid += p
        else:
            high += p
    return low, mid, high

def extract_motion_features(samples: Sequence[Sequence[float]]) -> Dict[str,float]:
    """Extract BITS-2 accelerometer window features.

    samples contains (ax, ay, az), in the same units as the canonical
    BITS-2 adapter. No timestamp interpolation is performed.
    """
    if not samples:
        return {c: 0.0 for c in FEATURE_COLUMNS}

    ax = [float(s[0]) for s in samples]
    ay = [float(s[1]) for s in samples]
    az = [float(s[2]) for s in samples]
    mag = [math.sqrt(a*a+b*b+c*c) for a,b,c in zip(ax,ay,az)]

    axm, axs, axmin, axmax, axr = _stats(ax)
    aym, ays, aymin, aymax, ayr = _stats(ay)
    azm, azs, azmin, azmax, azr = _stats(az)
    mm, ms, mmin, mmax, mr = _stats(mag)

    jerk = [
        mag[i]-mag[i-1]
        for i in range(1, len(mag))
    ]
    jm, js, _, jmax, _ = _stats(jerk)
    diff_energy = sum(d*d for d in jerk)/max(1, len(jerk))
    low, mid, high = _fft_band_energy(mag)

    return {
        "ax_mean": axm, "ax_std": axs, "ax_min": axmin, "ax_max": axmax, "ax_rms": axr,
        "ay_mean": aym, "ay_std": ays, "ay_min": aymin, "ay_max": aymax, "ay_rms": ayr,
        "az_mean": azm, "az_std": azs, "az_min": azmin, "az_max": azmax, "az_rms": azr,
        "mag_mean": mm, "mag_std": ms, "mag_min": mmin, "mag_max": mmax, "mag_rms": mr,
        "mag_p10": _percentile(mag,.10), "mag_p25": _percentile(mag,.25),
        "mag_p50": _percentile(mag,.50), "mag_p75": _percentile(mag,.75),
        "mag_p90": _percentile(mag,.90),
        "mag_sma": sum(abs(x)+abs(y)+abs(z) for x,y,z in zip(ax,ay,az))/len(ax),
        "mag_range": mmax-mmin,
        "jerk_mean": jm, "jerk_std": js, "jerk_max": jmax,
        "diff_energy": diff_energy,
        "mag_zero_crossings": float(_zero_crossings(mag)),
        "xy_corr": _corr(ax,ay), "xz_corr": _corr(ax,az), "yz_corr": _corr(ay,az),
        "fft_low": low, "fft_mid": mid, "fft_high": high,
    }
