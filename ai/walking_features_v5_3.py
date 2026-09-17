
"""SafeBand AI V5.3: signal-derived walking HR features."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass(x, fs=256.0, lo=0.5, hi=5.0):
    x=np.asarray(x,float); x=x-np.nanmedian(x)
    b,a=butter(3,[lo/(fs/2),hi/(fs/2)],btype="band")
    return filtfilt(b,a,np.nan_to_num(x))

def spectral_hr(ppg,fs=256.0,hr_min=40,hr_max=220):
    y=bandpass(ppg,fs)
    f=np.fft.rfftfreq(len(y),1/fs); p=np.abs(np.fft.rfft(y))**2
    m=(f>=hr_min/60)&(f<=hr_max/60)
    if not np.any(m): return np.nan
    return float(f[m][np.argmax(p[m])]*60)

def peak_hr(ppg,fs=256.0,hr_min=40,hr_max=220):
    y=bandpass(ppg,fs); distance=max(1,int(fs*60/hr_max))
    prom=max(1e-12,0.15*np.std(y))
    peaks,_=find_peaks(y,distance=distance,prominence=prom)
    if len(peaks)<2: return np.nan
    rr=np.diff(peaks)/fs; hr=60/np.median(rr)
    return float(hr) if hr_min<=hr<=hr_max else np.nan

def autocorr_hr(ppg,fs=256.0,hr_min=40,hr_max=220):
    y=bandpass(ppg,fs); y=y-np.mean(y)
    ac=np.correlate(y,y,mode="full")[len(y)-1:]
    lo=int(fs*60/hr_max); hi=min(len(ac)-1,int(fs*60/hr_min))
    if hi<=lo: return np.nan
    lag=lo+int(np.argmax(ac[lo:hi+1]))
    return float(60*fs/lag)

def features(ppg,acc=None,gyro=None,fs=256.0):
    p=np.asarray(ppg,float)
    y=bandpass(p,fs)
    vals=[spectral_hr(p,fs),peak_hr(p,fs),autocorr_hr(p,fs)]
    finite=[v for v in vals if np.isfinite(v)]
    cand=np.median(finite) if finite else np.nan
    agree=np.std(finite) if len(finite)>1 else (0.0 if finite else np.nan)
    feats={"spectral_hr":vals[0],"peak_hr":vals[1],"autocorr_hr":vals[2],
           "candidate_hr":cand,"candidate_spread":agree,
           "ppg_std":float(np.std(y)),"ppg_rms":float(np.sqrt(np.mean(y*y)))}
    if acc is not None:
        a=np.asarray(acc,float); amag=np.linalg.norm(a,axis=1)
        feats.update({"acc_mag_std":float(np.std(amag)),"acc_mag_rms":float(np.sqrt(np.mean(amag**2)))})
    if gyro is not None:
        g=np.asarray(gyro,float); gm=np.linalg.norm(g,axis=1)
        feats.update({"gyro_mag_std":float(np.std(gm)),"gyro_mag_rms":float(np.sqrt(np.mean(gm**2)))})
    return feats

def temporal_smooth(hr_series,alpha=0.35,max_jump_bpm=25.0):
    out=[]; prev=None
    for x in hr_series:
        x=float(x)
        if not np.isfinite(x): out.append(np.nan); continue
        if prev is not None:
            x=np.clip(x,prev-max_jump_bpm,prev+max_jump_bpm)
            x=alpha*x+(1-alpha)*prev
        out.append(x); prev=x
    return np.asarray(out)
