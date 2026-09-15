"""
SafeBand AI — Fall event features v3.

A fall is treated as a temporal event. This module summarizes the
pre-peak / peak / post-peak behavior inside a candidate window.
"""

from __future__ import annotations
import math
from typing import Sequence, Dict

FALL_EVENT_FEATURES = [
    "pre_peak_mean","peak_magnitude","post_peak_mean",
    "peak_to_baseline","peak_index_ratio",
    "pre_post_change","post_peak_std",
    "peak_width","high_energy_fraction",
    "recovery_ratio",
]

def fall_event_features(samples: Sequence[Sequence[float]]) -> Dict[str,float]:
    if len(samples)<4:
        return {k:0.0 for k in FALL_EVENT_FEATURES}
    mag=[math.sqrt(float(x[0])**2+float(x[1])**2+float(x[2])**2) for x in samples]
    n=len(mag)
    peak=max(mag)
    pi=mag.index(peak)
    pre=mag[:max(1,pi)]
    post=mag[min(n-1,pi+1):]
    pre_m=sum(pre)/len(pre)
    post_m=sum(post)/len(post)
    baseline=(pre_m+post_m)/2.0
    post_std=math.sqrt(sum((x-post_m)**2 for x in post)/len(post))
    width=sum(1 for x in mag if x >= baseline + 0.5*max(0.0,peak-baseline))
    high=sum(1 for x in mag if x >= baseline + 0.75*max(0.0,peak-baseline))
    return {
        "pre_peak_mean":pre_m,
        "peak_magnitude":peak,
        "post_peak_mean":post_m,
        "peak_to_baseline":peak/(baseline+1e-9),
        "peak_index_ratio":pi/max(1,n-1),
        "pre_post_change":abs(post_m-pre_m)/(baseline+1e-9),
        "post_peak_std":post_std,
        "peak_width":float(width),
        "high_energy_fraction":high/n,
        "recovery_ratio":post_m/(pre_m+1e-9),
    }
