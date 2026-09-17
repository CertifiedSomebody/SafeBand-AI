"""High-information feature extraction for FORTH-TRACE 9-DoF wrist IMU data."""
from __future__ import annotations
import numpy as np

CHANNELS = ["ax","ay","az","gx","gy","gz","mx","my","mz"]
MODALITIES = {"acc": CHANNELS[:3], "gyro": CHANNELS[3:6], "mag": CHANNELS[6:9]}

def _skew_kurt(x):
    x=np.asarray(x,float); m=float(np.mean(x)); d=x-m; sd=float(np.std(x))
    if sd<=1e-12: return 0.0,0.0
    z=d/sd
    return float(np.mean(z**3)), float(np.mean(z**4)-3.0)

def _spectral(x, fs):
    x = np.asarray(x, float); n=len(x)
    if n < 8: return (0.0,0.0,0.0,0.0,0.0)
    y = x - np.mean(x); p=np.abs(np.fft.rfft(y))**2; f=np.fft.rfftfreq(n,1/fs)
    if len(p)>1: p[0]=0
    total=float(p.sum())
    if total<=1e-12: return (0,0,0,0,0)
    pn=p/total
    entropy=float(-(pn[pn>0]*np.log2(pn[pn>0])).sum()/np.log2(len(pn)))
    peak=int(np.argmax(p))
    dom=float(f[peak])
    def band(a,b): return float(p[(f>=a)&(f<b)].sum()/total)
    return band(0.3,3.0), band(3.0,7.0), band(7.0,15.0), entropy, dom

def _series(x, fs, prefix):
    x=np.asarray(x,float); d=np.diff(x) if len(x)>1 else np.array([0.0])
    q=np.percentile(x,[5,10,25,50,75,90,95])
    low,mid,high,ent,dom=_spectral(x,fs)
    sk,ku=_skew_kurt(x)
    vals={
      "mean":np.mean(x),"std":np.std(x),"min":np.min(x),"max":np.max(x),"median":q[3],
      "p05":q[0],"p10":q[1],"p25":q[2],"p75":q[4],"p90":q[5],"p95":q[6],
      "range":np.ptp(x),"iqr":q[4]-q[2],"rms":np.sqrt(np.mean(x*x)),
      "skew":sk,"kurtosis":ku,
      "mad":np.mean(np.abs(x-np.mean(x))),"absdiff_mean":np.mean(np.abs(d)),
      "diff_std":np.std(d),"diff_energy":np.mean(d*d),
      "zero_crossings":np.count_nonzero((x[:-1]-x.mean())*(x[1:]-x.mean())<0) if len(x)>1 else 0,
      "fft_low":low,"fft_mid":mid,"fft_high":high,"spectral_entropy":ent,"dominant_hz":dom,
    }
    return {f"{prefix}_{k}":float(v) for k,v in vals.items()}

def extract_window_features(w, fs=51.2):
    """w: dict/list-like mapping of nine channel names to equal-length arrays."""
    out={}
    x={c:np.asarray(w[c],float) for c in CHANNELS}
    for c in CHANNELS: out.update(_series(x[c],fs,c))
    mags={
      "acc_mag":np.sqrt(x["ax"]**2+x["ay"]**2+x["az"]**2),
      "gyro_mag":np.sqrt(x["gx"]**2+x["gy"]**2+x["gz"]**2),
      "mag_mag":np.sqrt(x["mx"]**2+x["my"]**2+x["mz"]**2),
    }
    for n,v in mags.items(): out.update(_series(v,fs,n))
    # Pairwise axis correlations within each physical sensor.
    for name, cs in MODALITIES.items():
        for i in range(3):
            for j in range(i+1,3):
                a,b=x[cs[i]],x[cs[j]]
                sa,sb=np.std(a),np.std(b)
                out[f"{name}_{i}_{j}_corr"] = float(np.corrcoef(a,b)[0,1]) if sa>1e-12 and sb>1e-12 else 0.0
    # Cross-modality magnitude coupling captures wrist motion dynamics.
    for a,b in [("acc_mag","gyro_mag"),("acc_mag","mag_mag"),("gyro_mag","mag_mag")]:
        va,vb=mags[a],mags[b]; sa,sb=np.std(va),np.std(vb)
        out[f"{a}_{b}_corr"] = float(np.corrcoef(va,vb)[0,1]) if sa>1e-12 and sb>1e-12 else 0.0
    return out
