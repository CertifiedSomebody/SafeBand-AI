
"""SafeBand V3 — exact 84-column V6-compatible accelerometer feature contract."""
from __future__ import annotations
import math
import numpy as np

BASE = [
"ax_mean","ax_std","ax_min","ax_max","ax_rms","ay_mean","ay_std","ay_min","ay_max","ay_rms",
"az_mean","az_std","az_min","az_max","az_rms","mag_mean","mag_std","mag_min","mag_max","mag_rms",
"mag_p10","mag_p25","mag_p50","mag_p75","mag_p90","mag_sma","mag_range","jerk_mean","jerk_std",
"jerk_max","diff_energy","mag_zero_crossings","xy_corr","xz_corr","yz_corr","fft_low","fft_mid","fft_high"]
EVENT = ["pre_peak_mean","peak_magnitude","post_peak_mean","peak_to_baseline","peak_index_ratio",
"pre_post_change","post_peak_std","peak_width","high_energy_fraction","recovery_ratio"]
V6 = [
"v6_mag_mean","v6_mag_std","v6_mag_min","v6_mag_max","v6_mag_rms","v6_mag_p10","v6_mag_p25","v6_mag_p50","v6_mag_p75","v6_mag_p90",
"v6_peak_prominence","v6_peak_to_rms","v6_peak_to_median","v6_peak_index_ratio","v6_peak_width_half","v6_high_fraction",
"v6_pre_mean","v6_post_mean","v6_pre_std","v6_post_std","v6_pre_post_ratio","v6_pre_post_delta","v6_post_quiet_ratio",
"v6_pre_energy","v6_post_energy","v6_post_energy_ratio","v6_jerk_abs_mean","v6_jerk_abs_max","v6_jerk_std","v6_jerk_sign_changes",
"v6_diff_energy","v6_impulse_area","v6_axis_peak_ratio","v6_axis_std_ratio","v6_tilt_change","v6_spectral_entropy"]
FEATURES=BASE+EVENT+V6
assert len(FEATURES)==84

def _stats(x):
    x=np.asarray(x,dtype=float); m=float(x.mean()) if x.size else 0.0
    return m,float(x.std()),float(x.min()),float(x.max()),float(np.sqrt(np.mean(x*x))) if x.size else 0.0
def _pct(x,q): return float(np.percentile(x,q*100)) if len(x) else 0.0
def _corr(a,b):
    if len(a)<2: return 0.0
    a=a-a.mean(); b=b-b.mean(); den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/den) if den>1e-12 else 0.0
def _bands(mag,fs=20.0):
    if len(mag)<4: return 0.,0.,0.
    x=mag-mag.mean(); p=np.abs(np.fft.rfft(x)[1:])**2/(len(x)**2)
    f=np.fft.rfftfreq(len(x),1/fs)[1:]
    return float(p[f<=2].sum()),float(p[(f>2)&(f<=5)].sum()),float(p[f>5].sum())
def _entropy(mag):
    if len(mag)<8: return 0.0
    p=np.abs(np.fft.rfft(mag-mag.mean())[1:])**2
    s=p.sum()
    if s<=1e-12: return 0.0
    p=p[p>0]/s
    return float(-np.sum(p*np.log(p))/np.log(len(p))) if len(p)>1 else 0.0

def extract_features(samples):
    a=np.asarray(samples,dtype=float)
    if a.shape[0]==0: return {c:0.0 for c in FEATURES}
    ax,ay,az=a[:,0],a[:,1],a[:,2]
    mag=np.sqrt(ax*ax+ay*ay+az*az); n=len(mag)
    out={}
    for name,x in (("ax",ax),("ay",ay),("az",az),("mag",mag)):
        m,s,mn,mx,r=_stats(x)
        out.update({f"{name}_mean":m,f"{name}_std":s,f"{name}_min":mn,f"{name}_max":mx,f"{name}_rms":r})
    out.update({f"mag_p{q}":_pct(mag,q/100) for q in (10,25,50,75,90)})
    out["mag_sma"]=float(np.mean(np.abs(ax)+np.abs(ay)+np.abs(az)))
    out["mag_range"]=out["mag_max"]-out["mag_min"]
    jerk=np.diff(mag)
    jm,js,_,jx,_=_stats(jerk) if len(jerk) else (0.,0.,0.,0.,0.)
    out.update(jerk_mean=jm,jerk_std=js,jerk_max=jx,
               diff_energy=float(np.mean(jerk*jerk)) if len(jerk) else 0.,
               mag_zero_crossings=float(np.sum((mag[:-1]-mag.mean())*(mag[1:]-mag.mean())<0)) if n>1 else 0.,
               xy_corr=_corr(ax,ay),xz_corr=_corr(ax,az),yz_corr=_corr(ay,az))
    lo,mi,hi=_bands(mag); out.update(fft_low=lo,fft_mid=mi,fft_high=hi)
    peak=float(mag.max()); pi=int(np.argmax(mag)); pre=mag[:max(1,pi)]; post=mag[min(n-1,pi+1):]
    pre_m=float(pre.mean()); post_m=float(post.mean()); base=max(1e-9,(pre_m+post_m)/2)
    half=base+.5*(peak-base); high=base+.75*(peak-base)
    out.update(pre_peak_mean=pre_m,peak_magnitude=peak,post_peak_mean=post_m,
               peak_to_baseline=peak/(base+1e-9),peak_index_ratio=pi/max(1,n-1),
               pre_post_change=abs(post_m-pre_m)/(base+1e-9),
               post_peak_std=float(post.std()) if len(post) else 0.,
               peak_width=float(np.sum(mag>=half)),high_energy_fraction=float(np.mean(mag>=high)),
               recovery_ratio=post_m/(pre_m+1e-9))
    rms=float(np.sqrt(np.mean(mag*mag))); med=float(np.median(mag))
    early=max(1,n//4); ev=np.array([ax[:early].mean(),ay[:early].mean(),az[:early].mean()])
    lv=np.array([ax[-early:].mean(),ay[-early:].mean(),az[-early:].mean()])
    tilt=lambda v: math.atan2(float(v[1]),math.sqrt(float(v[0]**2+v[2]**2)))
    out.update(
      v6_mag_mean=float(mag.mean()),v6_mag_std=float(mag.std()),v6_mag_min=float(mag.min()),v6_mag_max=peak,v6_mag_rms=rms,
      v6_mag_p10=float(np.percentile(mag,10)),v6_mag_p25=float(np.percentile(mag,25)),v6_mag_p50=med,
      v6_mag_p75=float(np.percentile(mag,75)),v6_mag_p90=float(np.percentile(mag,90)),
      v6_peak_prominence=(peak-base)/(base+1e-9),v6_peak_to_rms=peak/(rms+1e-9),v6_peak_to_median=peak/(med+1e-9),
      v6_peak_index_ratio=pi/max(1,n-1),v6_peak_width_half=float(np.sum(mag>=half)),v6_high_fraction=float(np.mean(mag>=high)),
      v6_pre_mean=pre_m,v6_post_mean=post_m,v6_pre_std=float(pre.std()),v6_post_std=float(post.std()),
      v6_pre_post_ratio=post_m/(pre_m+1e-9),v6_pre_post_delta=abs(post_m-pre_m)/(base+1e-9),
      v6_post_quiet_ratio=1/(1+float(post.std())),v6_pre_energy=float(np.mean(pre*pre)),v6_post_energy=float(np.mean(post*post)),
      v6_post_energy_ratio=float(np.mean(post*post)/(np.mean(pre*pre)+1e-9)),
      v6_jerk_abs_mean=float(np.mean(np.abs(jerk))) if len(jerk) else 0.,
      v6_jerk_abs_max=float(np.max(np.abs(jerk))) if len(jerk) else 0.,
      v6_jerk_std=float(np.std(jerk)),v6_jerk_sign_changes=float(np.sum((jerk[:-1]*jerk[1:])<0)) if len(jerk)>1 else 0.,
      v6_diff_energy=float(np.mean(jerk*jerk)) if len(jerk) else 0.,
      v6_impulse_area=float(np.sum(np.maximum(0,mag-base))),
      v6_axis_peak_ratio=max(float(np.max(np.abs(ax))),float(np.max(np.abs(ay))),float(np.max(np.abs(az))))/(rms+1e-9),
      v6_axis_std_ratio=max(float(ax.std()),float(ay.std()),float(az.std()))/(float(ax.std()+ay.std()+az.std())+1e-9),
      v6_tilt_change=abs(tilt(lv)-tilt(ev)),v6_spectral_entropy=_entropy(mag))
    return {c:float(out[c]) for c in FEATURES}
