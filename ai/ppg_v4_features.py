"""PPG V4.2 feature extractor."""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram
from scipy.stats import skew, kurtosis

PPG_CHANNELS = (
    "distal_red","distal_ir","distal_green",
    "proximal_red","proximal_ir","proximal_green",
)

def _clean(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def _norm(x):
    x = _clean(x)
    med = np.median(x)
    mad = np.median(np.abs(x-med))
    scale = 1.4826*mad
    if scale < 1e-9: scale = np.std(x)
    if scale < 1e-9: scale = 1.0
    return (x-med)/scale

def _bp(x, fs, low=0.5, high=5.0):
    x = _clean(x)
    nyq = fs/2.0
    high = min(high, nyq*0.9)
    if not (0 < low < high < nyq): return _norm(x)
    b,a = butter(3, [low/nyq, high/nyq], btype="band")
    if len(x) <= 3*max(len(a),len(b)): return _norm(x)
    return filtfilt(b,a,x)

def _basic(x,p):
    x=_clean(x)
    return {
        p+"_mean":float(np.mean(x)), p+"_std":float(np.std(x)),
        p+"_rms":float(np.sqrt(np.mean(x*x))), p+"_ptp":float(np.ptp(x)),
        p+"_skew":float(skew(x,bias=False)) if len(x)>2 else 0.0,
        p+"_kurtosis":float(kurtosis(x,bias=False)) if len(x)>3 else 0.0,
    }

def _spec(x,fs,p):
    f,pow_=periodogram(_clean(x),fs=fs,detrend="constant",scaling="density")
    m=(f>=0.5)&(f<=5.0)
    if not np.any(m):
        return {p+"_dom_bpm":0.0,p+"_spec_centroid":0.0,p+"_band_power":0.0}
    fb,pb=f[m],pow_[m]
    den=np.sum(pb)+1e-12
    return {
        p+"_dom_bpm":float(fb[np.argmax(pb)]*60),
        p+"_spec_centroid":float(np.sum(fb*pb)/den),
        p+"_band_power":float(np.trapezoid(pb,fb)),
    }

def _channel(x,fs,p):
    z=_norm(_bp(x,fs))
    out={}; out.update(_basic(z,p)); out.update(_spec(z,fs,p))
    distance=max(1,int(round(fs*60/220)))
    peaks,props=find_peaks(z,distance=distance,prominence=0.15)
    out[p+"_peak_count"]=float(len(peaks))
    out[p+"_peak_hr"]=0.0; out[p+"_peak_hr_iqr"]=0.0; out[p+"_prominence_median"]=0.0
    if len(peaks)>=2:
        rr=np.diff(peaks)/fs
        hr=60/rr
        hr=hr[(hr>=30)&(hr<=220)]
        if hr.size:
            out[p+"_peak_hr"]=float(np.median(hr))
            out[p+"_peak_hr_iqr"]=float(np.percentile(hr,75)-np.percentile(hr,25))
            prom=props.get("prominences",[])
            if len(prom): out[p+"_prominence_median"]=float(np.median(prom))
    return out

def _corr(a,b):
    a=_clean(a); b=_clean(b)
    if np.std(a)<1e-12 or np.std(b)<1e-12: return 0.0
    r=float(np.corrcoef(a,b)[0,1])
    return r if np.isfinite(r) else 0.0

def _motion(data,fs,p):
    out={}
    for i,axis in enumerate(("x","y","z")): out.update(_basic(data[:,i],p+"_"+axis))
    mag=np.sqrt(np.sum(data*data,axis=1))
    out.update(_basic(mag,p+"_mag")); out.update(_spec(mag,fs,p+"_mag"))
    return out

def extract_features(ppg,acc,gyro,ppg_fs=500.0,motion_fs=500.0):
    ppg=_clean(ppg); acc=_clean(acc); gyro=_clean(gyro)
    if ppg.ndim!=2 or ppg.shape[1]!=6: raise ValueError(f"PPG shape must be (N,6), got {ppg.shape}")
    if acc.ndim!=2 or acc.shape[1]!=3: raise ValueError(f"ACC shape must be (N,3), got {acc.shape}")
    if gyro.ndim!=2 or gyro.shape[1]!=3: raise ValueError(f"GYRO shape must be (N,3), got {gyro.shape}")
    if not (len(ppg)==len(acc)==len(gyro)): raise ValueError("PPG/ACC/GYRO lengths differ")
    out={}
    for i,name in enumerate(PPG_CHANNELS): out.update(_channel(ppg[:,i],ppg_fs,name))
    pairs=((0,1),(0,2),(1,2),(3,4),(3,5),(4,5),(0,3),(1,4),(2,5))
    for a,b in pairs: out[f"corr_{PPG_CHANNELS[a]}_{PPG_CHANNELS[b]}"]=_corr(ppg[:,a],ppg[:,b])
    for i,name in enumerate(PPG_CHANNELS):
        dc=np.median(ppg[:,i]); ac=np.std(_bp(ppg[:,i],ppg_fs))
        out[name+"_ac_dc"]=float(ac/(abs(dc)+1e-9))
    out.update(_motion(acc,motion_fs,"acc")); out.update(_motion(gyro,motion_fs,"gyro"))
    am=np.sqrt(np.sum(acc*acc,axis=1)); gm=np.sqrt(np.sum(gyro*gyro,axis=1))
    for i,name in enumerate(PPG_CHANNELS):
        out[f"corr_{name}_accmag"]=_corr(ppg[:,i],am)
        out[f"corr_{name}_gyromag"]=_corr(ppg[:,i],gm)
    return {k:(float(v) if np.isfinite(v) else 0.0) for k,v in out.items()}
