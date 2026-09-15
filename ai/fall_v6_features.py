"""SafeBand AI — BITS-2 fall features v6.

Accelerometer-only event morphology. No timestamp interpolation or
cross-sensor synchronization is introduced.
"""
from __future__ import annotations
import math
from typing import Dict, Sequence

V6_FEATURES = [
    "v6_mag_mean","v6_mag_std","v6_mag_min","v6_mag_max","v6_mag_rms",
    "v6_mag_p10","v6_mag_p25","v6_mag_p50","v6_mag_p75","v6_mag_p90",
    "v6_peak_prominence","v6_peak_to_rms","v6_peak_to_median",
    "v6_peak_index_ratio","v6_peak_width_half","v6_high_fraction",
    "v6_pre_mean","v6_post_mean","v6_pre_std","v6_post_std",
    "v6_pre_post_ratio","v6_pre_post_delta","v6_post_quiet_ratio",
    "v6_pre_energy","v6_post_energy","v6_post_energy_ratio",
    "v6_jerk_abs_mean","v6_jerk_abs_max","v6_jerk_std","v6_jerk_sign_changes",
    "v6_diff_energy","v6_impulse_area","v6_axis_peak_ratio",
    "v6_axis_std_ratio","v6_tilt_change","v6_spectral_entropy",
]

def _mean(v): return sum(v)/len(v) if v else 0.0
def _std(v):
    if not v: return 0.0
    m=_mean(v); return math.sqrt(sum((x-m)**2 for x in v)/len(v))
def _pct(v,q):
    if not v: return 0.0
    a=sorted(v); p=(len(a)-1)*q; lo=int(math.floor(p)); hi=int(math.ceil(p))
    return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(p-lo)
def _energy(v): return _mean([x*x for x in v])
def _sign_changes(v):
    return float(sum(1 for a,b in zip(v,v[1:]) if (a<0<=b) or (a>0>=b)))

def _spectral_entropy(v):
    n=len(v)
    if n<8: return 0.0
    x=[z-_mean(v) for z in v]
    powers=[]
    for k in range(1,n//2+1):
        re=sum(x[t]*math.cos(2*math.pi*k*t/n) for t in range(n))
        im=-sum(x[t]*math.sin(2*math.pi*k*t/n) for t in range(n))
        powers.append((re*re+im*im)/(n*n))
    total=sum(powers)
    if total<=1e-12: return 0.0
    probs=[p/total for p in powers if p>0]
    return -sum(p*math.log(p) for p in probs)/math.log(len(probs)) if len(probs)>1 else 0.0

def extract_fall_v6(samples: Sequence[Sequence[float]]) -> Dict[str,float]:
    if not samples:
        return {k:0.0 for k in V6_FEATURES}
    ax=[float(s[0]) for s in samples]; ay=[float(s[1]) for s in samples]; az=[float(s[2]) for s in samples]
    mag=[math.sqrt(x*x+y*y+z*z) for x,y,z in zip(ax,ay,az)]
    n=len(mag); peak=max(mag); pi=mag.index(peak); med=_pct(mag,.5); rms=math.sqrt(_energy(mag))
    base=max(1e-9,(_mean(mag[:max(1,pi)])+_mean(mag[min(n-1,pi+1):]))/2)
    half=base+0.5*(peak-base); high=base+0.75*(peak-base)
    width=sum(1 for x in mag if x>=half); high_frac=sum(1 for x in mag if x>=high)/n
    pre=mag[:max(1,pi)]; post=mag[min(n-1,pi+1):]
    pre_m=_mean(pre); post_m=_mean(post)
    jerk=[mag[i]-mag[i-1] for i in range(1,n)]
    pre_e=_energy(pre); post_e=_energy(post)
    axis_max=max(max(abs(x) for x in ax),max(abs(y) for y in ay),max(abs(z) for z in az))
    axis_stds=[_std(ax),_std(ay),_std(az)]
    # Gravity-dominated tilt proxy: compare early and late mean vectors.
    cut=max(1,n//4); early=(_mean(ax[:cut]),_mean(ay[:cut]),_mean(az[:cut])); late=(_mean(ax[-cut:]),_mean(ay[-cut:]),_mean(az[-cut:]))
    def tilt(v): return math.atan2(v[1],math.sqrt(v[0]*v[0]+v[2]*v[2]))
    tilt_change=abs(tilt(late)-tilt(early))
    return {
      "v6_mag_mean":_mean(mag),"v6_mag_std":_std(mag),"v6_mag_min":min(mag),"v6_mag_max":peak,"v6_mag_rms":rms,
      "v6_mag_p10":_pct(mag,.1),"v6_mag_p25":_pct(mag,.25),"v6_mag_p50":med,"v6_mag_p75":_pct(mag,.75),"v6_mag_p90":_pct(mag,.9),
      "v6_peak_prominence":(peak-base)/(base+1e-9),"v6_peak_to_rms":peak/(rms+1e-9),"v6_peak_to_median":peak/(med+1e-9),
      "v6_peak_index_ratio":pi/max(1,n-1),"v6_peak_width_half":float(width),"v6_high_fraction":high_frac,
      "v6_pre_mean":pre_m,"v6_post_mean":post_m,"v6_pre_std":_std(pre),"v6_post_std":_std(post),
      "v6_pre_post_ratio":post_m/(pre_m+1e-9),"v6_pre_post_delta":abs(post_m-pre_m)/(base+1e-9),
      "v6_post_quiet_ratio":1.0/(1.0+_std(post)),"v6_pre_energy":pre_e,"v6_post_energy":post_e,"v6_post_energy_ratio":post_e/(pre_e+1e-9),
      "v6_jerk_abs_mean":_mean([abs(x) for x in jerk]),"v6_jerk_abs_max":max([abs(x) for x in jerk] or [0.0]),"v6_jerk_std":_std(jerk),"v6_jerk_sign_changes":_sign_changes(jerk),
      "v6_diff_energy":_energy(jerk),"v6_impulse_area":sum(max(0.0,x-base) for x in mag),
      "v6_axis_peak_ratio":axis_max/(rms+1e-9),"v6_axis_std_ratio":max(axis_stds)/(sum(axis_stds)+1e-9),
      "v6_tilt_change":tilt_change,"v6_spectral_entropy":_spectral_entropy(mag),
    }
