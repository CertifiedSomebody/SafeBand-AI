"""SafeBand AI audio feature extraction for INMP441/TinyML experiments.

The extractor operates on raw PCM samples and returns a deterministic,
dataset-agnostic feature vector. It does not infer a semantic event from
volume alone. Training labels must come from the dataset annotation.
"""
from __future__ import annotations
from typing import Dict, Sequence
import numpy as np

FEATURE_COLUMNS = [
    "rms","peak_abs","crest_factor","zero_crossing_rate","dc_offset",
    "dynamic_range","spectral_centroid_hz","spectral_bandwidth_hz",
    "spectral_flatness","spectral_rolloff_hz",
    "energy_80_300_hz","energy_300_1000_hz","energy_1k_3k_hz","energy_3k_8k_hz",
    "energy_above_8k_hz","dominant_frequency_hz",
]

def _safe(x: float) -> float:
    return float(x) if np.isfinite(x) else 0.0

def extract_audio_features(samples: Sequence[float], sample_rate_hz: float) -> Dict[str,float]:
    x=np.asarray(samples,dtype=np.float64).reshape(-1)
    fs=float(sample_rate_hz)
    if x.size < 8 or fs <= 0:
        return {c:0.0 for c in FEATURE_COLUMNS}
    x=np.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
    peak=float(np.max(np.abs(x)))
    rms=float(np.sqrt(np.mean(x*x)))
    zcr=float(np.mean((x[:-1]*x[1:])<0)) if x.size>1 else 0.0
    dc=float(np.mean(x))
    dynamic=float(np.percentile(x,99)-np.percentile(x,1))
    crest=peak/rms if rms>1e-12 else 0.0

    w=np.hanning(x.size)
    y=(x-dc)*w
    spec=np.abs(np.fft.rfft(y))
    power=spec**2
    freqs=np.fft.rfftfreq(x.size,d=1.0/fs)
    if power.size>1:
        p=power.copy(); p[0]=0.0
        total=float(np.sum(p))
        if total>1e-12:
            centroid=float(np.sum(freqs*p)/total)
            bandwidth=float(np.sqrt(np.sum(((freqs-centroid)**2)*p)/total))
            csum=np.cumsum(p)
            roll=float(freqs[min(len(freqs)-1,int(np.searchsorted(csum,0.85*total)))])
            nz=p[p>1e-12]
            flat=float(np.exp(np.mean(np.log(nz)))/np.mean(nz)) if nz.size else 0.0
            dom=float(freqs[int(np.argmax(p))])
            def band(lo,hi): return float(np.sum(p[(freqs>=lo)&(freqs<hi)])/total)
            vals={
                "rms":rms,"peak_abs":peak,"crest_factor":crest,"zero_crossing_rate":zcr,
                "dc_offset":dc,"dynamic_range":dynamic,"spectral_centroid_hz":centroid,
                "spectral_bandwidth_hz":bandwidth,"spectral_flatness":flat,
                "spectral_rolloff_hz":roll,"energy_80_300_hz":band(80,300),
                "energy_300_1000_hz":band(300,1000),"energy_1k_3k_hz":band(1000,3000),
                "energy_3k_8k_hz":band(3000,8000),"energy_above_8k_hz":band(8000,fs/2+1),
                "dominant_frequency_hz":dom}
            return {k:_safe(v) for k,v in vals.items()}
    return {c:_safe(v) for c,v in {
        "rms":rms,"peak_abs":peak,"crest_factor":crest,"zero_crossing_rate":zcr,
        "dc_offset":dc,"dynamic_range":dynamic}.items()} | {c:0.0 for c in FEATURE_COLUMNS if c not in {
        "rms","peak_abs","crest_factor","zero_crossing_rate","dc_offset","dynamic_range"}}

__all__=["FEATURE_COLUMNS","extract_audio_features"]
