from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

LABELS = ["breath","cough","crying","laugh","screaming","sneeze","yawn"]

def repo_path(p):
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p

def resolve_audio(row):
    for col in ("audio_path", "path", "filepath"):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            raw = str(row[col]).strip()
            p = repo_path(raw)
            if p.exists() and p.is_file():
                return p
            name = Path(raw).name
            hits = list((ROOT/"datasets"/"raw"/"Nonspeech7k").rglob(name))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise RuntimeError(f"Ambiguous audio basename {name}: {len(hits)} matches")
    raise FileNotFoundError(f"Cannot resolve WAV for file_id={row.get('file_id','?')}")

def load_audio(path, target_sr=16000):
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
        x /= max(peak, 1e-8)
    return x, sr

def make_logmel(x, sr):
    import librosa
    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=512, hop_length=160, n_mels=64,
        fmin=50, fmax=min(7600, sr//2), power=2.0
    )
    S = librosa.power_to_db(S, ref=np.max)
    S = np.clip(S, -80.0, 0.0)
    target_frames = 128
    if S.shape[1] < target_frames:
        S = np.pad(S, ((0,0),(0,target_frames-S.shape[1])),
                   mode="constant", constant_values=-80.0)
    elif S.shape[1] > target_frames:
        start = (S.shape[1]-target_frames)//2
        S = S[:, start:start+target_frames]
    return S.astype(np.float32)
