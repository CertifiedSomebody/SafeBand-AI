from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

CLASSES = ["breath","cough","crying","laugh","screaming","sneeze","yawn"]
EPS = 1e-10

def load_audio(path):
    fs, x = wavfile.read(str(path))
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = max(abs(info.min), info.max)
        x = x.astype(np.float32) / float(scale)
    else:
        x = x.astype(np.float32)
    x = np.nan_to_num(x)
    return int(fs), x

def rms(x):
    return float(np.sqrt(np.mean(x*x) + EPS))

def zcr(x):
    return float(np.mean(x[:-1] * x[1:] < 0)) if len(x)>1 else 0.0

def spectral_features(x, fs):
    if len(x) < 16:
        return {k:0.0 for k in ["centroid","bandwidth","rolloff85","flatness","flux"]}
    win = signal.windows.hann(len(x), sym=False)
    mag = np.abs(np.fft.rfft(x*win))
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    s = mag.sum() + EPS
    centroid = float((freqs*mag).sum()/s)
    bandwidth = float(np.sqrt(((freqs-centroid)**2*mag).sum()/s))
    c = np.cumsum(mag)
    roll = float(freqs[min(len(freqs)-1, np.searchsorted(c, .85*c[-1]))])
    flat = float(np.exp(np.mean(np.log(mag+EPS))) / (np.mean(mag)+EPS))
    return {"centroid":centroid,"bandwidth":bandwidth,"rolloff85":roll,"flatness":flat,"flux":0.0}

def extract_features(path):
    fs,x = load_audio(path)
    duration = len(x)/fs if fs else 0
    s = spectral_features(x,fs)
    f = {
        "sample_rate":float(fs), "duration_s":float(duration),
        "rms":rms(x), "zcr":zcr(x),
        "peak_abs":float(np.max(np.abs(x))) if len(x) else 0.0,
        "crest_factor":float((np.max(np.abs(x))+EPS)/(rms(x)+EPS)) if len(x) else 0.0,
        **s
    }
    # Robust frame statistics capture temporal behavior without using labels.
    n = len(x)
    frame = max(128, int(round(.025*fs)))
    hop = max(64, int(round(.010*fs)))
    if n >= frame:
        vals = np.array([rms(x[i:i+frame]) for i in range(0,n-frame+1,hop)],dtype=np.float32)
        zvals = np.array([zcr(x[i:i+frame]) for i in range(0,n-frame+1,hop)],dtype=np.float32)
        f.update({
            "frame_rms_mean":float(vals.mean()), "frame_rms_std":float(vals.std()),
            "frame_rms_p10":float(np.percentile(vals,10)),
            "frame_rms_p50":float(np.percentile(vals,50)),
            "frame_rms_p90":float(np.percentile(vals,90)),
            "frame_zcr_mean":float(zvals.mean()), "frame_zcr_std":float(zvals.std())
        })
    else:
        f.update({k:0.0 for k in ["frame_rms_mean","frame_rms_std","frame_rms_p10","frame_rms_p50","frame_rms_p90","frame_zcr_mean","frame_zcr_std"]})
    return f
