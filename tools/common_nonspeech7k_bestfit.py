from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

LABELS = ["breath", "cough", "crying", "laugh", "screaming", "sneeze", "yawn"]

def load_audio(path, target_sr=16000):
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError(f"Invalid audio: {path}")
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x /= max(peak, 1e-8)
    return x.astype(np.float32), int(sr)

def make_features(x, sr):
    import librosa
    # Frozen V2 time-frequency representation: 64 log-mel + MFCC/deltas
    mel = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=512, hop_length=160, n_mels=64,
        fmin=50, fmax=min(7600, sr // 2), power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = np.clip(logmel, -80.0, 0.0)

    mfcc = librosa.feature.mfcc(
        S=logmel, sr=sr, n_mfcc=20, n_fft=512, hop_length=160
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = []
    for A in (logmel, mfcc, d1, d2):
        feats.extend([
            A.mean(axis=1), A.std(axis=1),
            np.percentile(A, 10, axis=1),
            np.percentile(A, 50, axis=1),
            np.percentile(A, 90, axis=1),
        ])
    rms = librosa.feature.rms(y=x, frame_length=512, hop_length=160)[0]
    zcr = librosa.feature.zero_crossing_rate(y=x, frame_length=512, hop_length=160)[0]
    feats.extend([
        np.array([rms.mean(), rms.std(), np.percentile(rms, 10),
                  np.percentile(rms, 50), np.percentile(rms, 90)]),
        np.array([zcr.mean(), zcr.std(), np.percentile(zcr, 10),
                  np.percentile(zcr, 50), np.percentile(zcr, 90)]),
    ])
    out = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in feats])
    if not np.isfinite(out).all():
        raise ValueError("Non-finite feature vector")
    return out.astype(np.float32)
