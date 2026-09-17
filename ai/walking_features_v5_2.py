"""Walking HR feature extraction for SafeBand PPG V5.2.

Design goal: preserve the established 8 s / 2 s representation while making
motion groups explicitly controllable and interpretable.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram
from scipy.stats import skew, kurtosis

def _clean(x):
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

def robust_norm(x):
    x = _clean(x)
    med = np.median(x)
    mad = np.median(np.abs(x-med))
    scale = 1.4826*mad
    if scale < 1e-9: scale = np.std(x)
    if scale < 1e-9: scale = 1.0
    return (x-med)/scale

def bandpass(x, fs, low=0.5, high=5.0):
    x = _clean(x); nyq=fs/2
    high=min(high, nyq*0.9)
    if not (0 < low < high < nyq): return robust_norm(x)
    b,a=butter(3,[low/nyq,high/nyq],btype="band")
    if len(x) <= 3*max(len(a),len(b)): return robust_norm(x)
    return filtfilt(b,a,x)

def basic(x, prefix):
    x=_clean(x)
    return {
        f"{prefix}_mean":float(np.mean(x)),
        f"{prefix}_std":float(np.std(x)),
        f"{prefix}_rms":float(np.sqrt(np.mean(x*x))),
        f"{prefix}_ptp":float(np.ptp(x)),
        f"{prefix}_skew":float(skew(x,bias=False)) if len(x)>2 else 0.0,
        f"{prefix}_kurtosis":float(kurtosis(x,bias=False)) if len(x)>3 else 0.0,
    }

def spectral(x, fs, prefix, low=0.5, high=5.0):
    f,p=periodogram(_clean(x),fs=fs,detrend="constant",scaling="density")
    m=(f>=low)&(f<=high)
    if not np.any(m):
        return {f"{prefix}_dom_bpm":0.0,f"{prefix}_spec_centroid":0.0,
                f"{prefix}_band_power":0.0}
    fb,pb=f[m],p[m]; den=np.sum(pb)+1e-12
    return {
        f"{prefix}_dom_bpm":float(fb[np.argmax(pb)]*60),
        f"{prefix}_spec_centroid":float(np.sum(fb*pb)/den),
        f"{prefix}_band_power":float(np.trapezoid(pb,fb)),
    }

def _corr(a,b):
    a,b=_clean(a),_clean(b)
    if np.std(a)<1e-12 or np.std(b)<1e-12: return 0.0
    r=float(np.corrcoef(a,b)[0,1])
    return r if np.isfinite(r) else 0.0

def ppg_features(ppg,fs):
    z=bandpass(ppg,fs); out={}
    out.update(basic(z,"ppg")); out.update(spectral(z,fs,"ppg"))
    distance=max(1,int(round(fs*60/220)))
    peaks,props=find_peaks(z,distance=distance,prominence=0.15)
    out["ppg_peak_count"]=float(len(peaks))
    out["ppg_peak_hr"]=0.0
    out["ppg_peak_hr_iqr"]=0.0
    out["ppg_prominence_median"]=0.0
    if len(peaks)>=2:
        rr=np.diff(peaks)/fs
        hr=60/rr; hr=hr[(hr>=30)&(hr<=220)]
        if hr.size:
            out["ppg_peak_hr"]=float(np.median(hr))
            out["ppg_peak_hr_iqr"]=float(np.percentile(hr,75)-np.percentile(hr,25))
            pr=props.get("prominences",[])
            if len(pr): out["ppg_prominence_median"]=float(np.median(pr))
    dc=np.median(_clean(ppg)); ac=np.std(z)
    out["ppg_ac_dc"]=float(ac/(abs(dc)+1e-9))
    return out,z

def motion_features(data,fs,prefix):
    data=_clean(data); out={}
    for i,axis in enumerate(("x","y","z")):
        out.update(basic(data[:,i],f"{prefix}_{axis}"))
    mag=np.sqrt(np.sum(data*data,axis=1))
    out.update(basic(mag,f"{prefix}_mag"))
    out.update(spectral(mag,fs,f"{prefix}_mag",low=0.5,high=5.0))
    return out,mag

def extract_features(ppg,acc,gyro,wide_acc=None,mag=None,fs=256.0):
    ppg=_clean(ppg).reshape(-1); acc=_clean(acc); gyro=_clean(gyro)
    if acc.shape!=(len(ppg),3) or gyro.shape!=(len(ppg),3):
        raise ValueError("PPG/ACC/GYRO shapes are incompatible")
    out,ppgbp=ppg_features(ppg,fs)
    acco,accmag=motion_features(acc,fs,"acc"); gyroo,gyromag=motion_features(gyro,fs,"gyro")
    out.update(acco); out.update(gyroo)
    out["corr_ppg_accmag"]=_corr(ppgbp,accmag)
    out["corr_ppg_gyromag"]=_corr(ppgbp,gyromag)
    if wide_acc is not None:
        wao,wamag=motion_features(wide_acc,fs,"wacc")
        out.update(wao); out["corr_ppg_waccmag"]=_corr(ppgbp,wamag)
    if mag is not None:
        mo,mmag=motion_features(mag,fs,"mag")
        out.update(mo); out["corr_ppg_magmag"]=_corr(ppgbp,mmag)
    return {k:float(v) if np.isfinite(v) else 0.0 for k,v in out.items()}
