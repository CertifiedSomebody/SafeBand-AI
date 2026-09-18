from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import soundfile as sf


LABELS = ["breath", "cough", "crying", "laugh", "screaming", "sneeze", "yawn"]


def repo_path(path_value):
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path


def resolve_audio(row):
    for column in ("audio_path", "path", "filepath"):
        if column in row.index and pd.notna(row[column]):
            raw = str(row[column]).strip()
            if not raw:
                continue

            direct = repo_path(raw)
            if direct.is_file():
                return direct

            # Last-resort basename discovery, but only accept an unambiguous hit.
            basename = Path(raw).name
            root = ROOT / "datasets" / "raw" / "Nonspeech7k"
            hits = list(root.rglob(basename))
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise RuntimeError(
                    f"Ambiguous audio basename '{basename}': {len(hits)} matches."
                )

    raise FileNotFoundError(
        f"Could not resolve audio for file_id={row.get('file_id', '<unknown>')}"
    )


def load_audio(path, target_sr=16000):
    x, sr = sf.read(path, always_2d=False)
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        x = x.mean(axis=1)

    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"Invalid audio shape: {path}")

    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite audio samples: {path}")

    if sr != target_sr:
        import librosa

        x = librosa.resample(
            x, orig_sr=sr, target_sr=target_sr
        ).astype(np.float32)
        sr = target_sr

    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        x = x / max(peak, 1e-8)

    return x.astype(np.float32), int(sr)


def make_logmel(x, sr):
    import librosa

    spectrogram = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=512,
        hop_length=160,
        n_mels=64,
        fmin=50,
        fmax=min(7600, sr // 2),
        power=2.0,
    )

    # Per-recording dB conversion is intentional and matches V3.
    logmel = librosa.power_to_db(spectrogram, ref=np.max)
    logmel = np.clip(logmel, -80.0, 0.0)

    target_frames = 128
    frames = logmel.shape[1]

    if frames < target_frames:
        logmel = np.pad(
            logmel,
            ((0, 0), (0, target_frames - frames)),
            mode="constant",
            constant_values=-80.0,
        )
    elif frames > target_frames:
        start = (frames - target_frames) // 2
        logmel = logmel[:, start : start + target_frames]

    if logmel.shape != (64, 128):
        raise RuntimeError(f"Unexpected log-mel shape: {logmel.shape}")

    if not np.isfinite(logmel).all():
        raise RuntimeError("Non-finite log-mel values produced.")

    return logmel.astype(np.float32)
