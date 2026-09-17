import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, periodogram

def preprocess_bvp(x, fs):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    if len(x)<64 or not np.isfinite(x).all():
        raise ValueError("BVP must be finite and contain >=64 samples")
    x=x-np.median(x)
    s=np.std(x)
    if s<1e-10: raise ValueError("BVP has near-zero variance")
    x=x/s
    ny=fs/2
    b,a=butter(3,[0.6/ny,4.0/ny],btype="band")
    return filtfilt(b,a,x)

def spectral_features(x,fs):
    nfft=max(4096,2**int(np.ceil(np.log2(len(x)))+3))
    f,p=periodogram(x,fs=fs,window="hann",detrend="linear",
                    nfft=nfft,scaling="spectrum")
    m=(f>=0.6)&(f<=4.0); f=f[m];p=p[m]
    if len(p)==0 or p.sum()<=0:
        return np.nan,np.nan,np.nan,np.nan,np.nan
    k=int(np.argmax(p)); dom=float(f[k]*60)
    prob=p/(p.sum()+1e-12)
    ent=float(-np.sum(prob*np.log(prob+1e-12))/np.log(len(prob)))
    fund=float(p[k]/(p.sum()+1e-12))
    f0=f[k]
    harm=float(np.sum(p[np.abs(f-2*f0)<=0.12])/(p[k]+1e-12))
    low=float(np.sum(p[(f>=.6)&(f<1.5)]))
    high=float(np.sum(p[(f>=1.5)&(f<=4)]))
    return dom,ent,fund,harm,float(low/(high+1e-12))

def autocorr_hr(x,fs):
    lo=max(1,int(fs*60/180)); hi=min(len(x)-2,int(fs*60/42))
    if hi<=lo:return np.nan,np.nan
    y=x-np.mean(x); ac=np.correlate(y,y,mode="full")[len(y)-1:]
    if ac[0]<=0:return np.nan,np.nan
    ac=ac/ac[0]; r=ac[lo:hi+1]; k=lo+int(np.argmax(r))
    return float(60*fs/k),float(np.clip(ac[k],0,1))

def peak_features(x,fs):
    distance=max(1,int(fs*60/180))
    peaks,props=find_peaks(x,distance=distance,prominence=max(.08,.08*np.std(x)))
    if len(peaks)<2:return np.nan,np.nan,np.nan,np.nan
    ibi=np.diff(peaks)/fs
    ibi=ibi[(ibi>=60/180)&(ibi<=60/42)]
    if len(ibi)==0:return np.nan,np.nan,np.nan,np.nan
    med=float(np.median(ibi)); mad=float(np.median(np.abs(ibi-med)))
    return float(60/med),float(np.std(ibi)),float(1/(1+mad)),float(np.median(props["prominences"]))

def extract_ppg_features(bvp,fs):
    x=preprocess_bvp(bvp,fs)
    dom,ent,fund,harm,lh=spectral_features(x,fs)
    ah,aq=autocorr_hr(x,fs)
    ph,ibi_std,reg,prom=peak_features(x,fs)
    vals=np.array([dom,ah,ph],float); good=vals[np.isfinite(vals)]
    med=float(np.median(good)) if len(good) else np.nan
    spread=float(np.std(good)) if len(good)>1 else 99.
    return {
      "autocorr_hr":ah,"autocorr_strength":aq,
      "candidate_hr_median":med,"candidate_hr_spread":spread,
      "dom_bpm":dom,"fund_power_frac":fund,"harmonic_ratio":harm,
      "ibi_median":(60/ph if np.isfinite(ph) else np.nan),
      "ibi_std":ibi_std,"low_high_ratio":lh,"peak_count":float(len(find_peaks(x,distance=max(1,int(fs*60/180)))[0])),
      "peak_hr":ph,"peak_prom_median":prom,"peak_regularity":reg,
      "ppg_diff_abs_mean":float(np.mean(np.abs(np.diff(x)))),
      "ppg_diff_std":float(np.std(np.diff(x))),
      "ppg_iqr":float(np.percentile(x,75)-np.percentile(x,25)),
      "ppg_kurt_proxy":float(np.mean(x**4)),
      "ppg_ptp":float(np.ptp(x)),"ppg_rms":float(np.sqrt(np.mean(x*x))),
      "ppg_skew_proxy":float(np.mean(x**3)),
      "spec_entropy":ent,
    }
