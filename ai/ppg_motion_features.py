import numpy as np

def extract_acc_features(acc,fs):
    a=np.asarray(acc,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all():
        raise ValueError(f"ACC must be finite (N,3), got {a.shape}")
    mag=np.linalg.norm(a,axis=1)
    dm=np.diff(mag,prepend=mag[:1])
    c=mag-np.median(mag)
    return {
      "acc_diff_abs_mean":float(np.mean(np.abs(dm))),
      "acc_diff_std":float(np.std(dm)),
      "acc_dynamic_energy":float(np.mean(c*c)),
      "acc_dynamic_std":float(np.std(c)),
      "acc_fs":float(fs),
      "acc_mag_mean":float(np.mean(mag)),
      "acc_mag_range":float(np.ptp(mag)),
      "acc_mag_rms":float(np.sqrt(np.mean(mag*mag))),
      "acc_mag_std":float(np.std(mag)),
    }

def extract_features(bvp,acc,bvp_fs,acc_fs):
    d=extract_ppg_features(bvp,bvp_fs)
    d.update(extract_acc_features(acc,acc_fs))
    return d
from .ppg_signal import extract_ppg_features
