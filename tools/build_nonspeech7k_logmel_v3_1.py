from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.common_nonspeech7k_cnn_fixed import (
    LABELS,
    load_audio,
    make_logmel,
    resolve_audio,
)


EXPECTED_ROWS = 6283
EXPECTED_GROUPS = 1899


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="datasets/processed/nonspeech7k/train_clean_manifest.csv",
    )
    parser.add_argument(
        "--out",
        default="datasets/processed/nonspeech7k/logmel_v3_1.npz",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest

    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest, dtype={"file_id": "string"})

    required = {"file_id", "label", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {sorted(missing)}")

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, found {len(df)}."
        )

    file_ids = df["file_id"].astype("string").str.strip()
    if file_ids.nunique() != EXPECTED_GROUPS:
        raise RuntimeError(
            f"Expected {EXPECTED_GROUPS} groups, found {file_ids.nunique()}."
        )

    label_to_id = {label: i for i, label in enumerate(LABELS)}

    X = np.empty((len(df), 1, 64, 128), dtype=np.float32)
    y = np.empty(len(df), dtype=np.int64)

    # IMPORTANT: fixed-width Unicode can silently truncate strings.
    # Use an object array for groups and convert explicitly before saving.
    groups = np.empty(len(df), dtype=object)
    audio_paths = np.empty(len(df), dtype=object)

    for i, (_, row) in enumerate(df.iterrows()):
        label = str(row["label"])
        if label not in label_to_id:
            raise RuntimeError(f"Unknown label at row {i}: {label}")

        audio = resolve_audio(row)
        waveform, sr = load_audio(audio, target_sr=16000)
        X[i, 0] = make_logmel(waveform, sr)
        y[i] = label_to_id[label]
        groups[i] = str(row["file_id"]).strip()
        audio_paths[i] = str(audio.relative_to(ROOT))

        if (i + 1) % 500 == 0:
            print(f"[INFO] processed {i + 1}/{len(df)}")

    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite values found in generated cache.")

    groups = np.asarray(groups, dtype=str)
    audio_paths = np.asarray(audio_paths, dtype=str)

    if len(set(groups.tolist())) != EXPECTED_GROUPS:
        raise RuntimeError(
            "Group count changed during cache creation. "
            f"Expected {EXPECTED_GROUPS}, got {len(set(groups.tolist()))}."
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        X=X,
        y=y,
        groups=groups,
        labels=np.asarray(LABELS, dtype=str),
        audio_paths=audio_paths,
    )

    print(f"[PASS] cache written       : {out}")
    print(f"[PASS] X shape              : {X.shape}")
    print(f"[PASS] unique File IDs     : {len(set(groups.tolist()))}")
    print("[PASS] class counts:")
    counts = np.bincount(y, minlength=len(LABELS))
    for label, count in zip(LABELS, counts):
        print(f"       {label:10s}: {int(count)}")


if __name__ == "__main__":
    main()
