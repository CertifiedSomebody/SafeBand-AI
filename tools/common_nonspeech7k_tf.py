from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

LABELS = ["breath","cough","crying","laugh","screaming","sneeze","yawn"]
EXPECTED = set(LABELS)

def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def resolve_audio(row):
    # Manifest schema is authoritative: prefer audio_path.
    for col in ("audio_path","path","filepath"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            p = repo_path(str(row[col]).strip())
            if p.exists() and p.is_file():
                return p
            # tolerate absolute paths from a different checkout by basename fallback
            name = Path(str(row[col])).name
            if name:
                hits = list((ROOT/"datasets"/"raw"/"Nonspeech7k").rglob(name))
                if len(hits) == 1:
                    return hits[0]
    raise FileNotFoundError(f"Could not resolve WAV for file_id={row.get('file_id','?')}")

def load_mono(path, target_sr=16000):
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f"Invalid audio: {path}")
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / max(peak, 1e-8)
    return x, sr

def logmel_mfcc_features(x, sr):
    import librosa
    n_fft, hop, n_mels, n_mfcc = 512, 160, 64, 20
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=n_mels, fmin=50, fmax=min(7600, sr//2),
                                       power=2.0)
    L = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=L, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    def stats(A, prefix):
        q = np.percentile(A, [10,25,50,75,90], axis=1)
        out = {}
        for i in range(A.shape[0]):
            out[f"{prefix}{i+1}_mean"] = float(np.mean(A[i]))
            out[f"{prefix}{i+1}_std"] = float(np.std(A[i]))
            out[f"{prefix}{i+1}_p10"] = float(q[0,i])
            out[f"{prefix}{i+1}_p50"] = float(q[2,i])
            out[f"{prefix}{i+1}_p90"] = float(q[4,i])
        return out
    feat = {}
    feat.update(stats(mfcc, "mfcc"))
    feat.update(stats(d1, "dmfcc"))
    feat.update(stats(d2, "ddmfcc"))
    feat["mel_energy_mean"] = float(np.mean(S))
    feat["mel_energy_std"] = float(np.std(S))
    feat["duration_s"] = float(len(x)/sr)
    feat["rms"] = float(np.sqrt(np.mean(x*x)+1e-12))
    return feat
