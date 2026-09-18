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
    make_logmel,
    load_audio,
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
        "--check-all",
        action="store_true",
        help="Decode and validate every training WAV.",
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
            f"Expected {EXPECTED_ROWS} clean training rows, found {len(df)}."
        )

    file_ids = df["file_id"].astype("string").str.strip()
    if file_ids.isna().any() or (file_ids == "").any():
        raise RuntimeError("Empty/null file_id found.")

    group_count = file_ids.nunique()
    if group_count != EXPECTED_GROUPS:
        raise RuntimeError(
            f"Expected {EXPECTED_GROUPS} unique File IDs, found {group_count}."
        )

    labels = set(df["label"].astype(str))
    if labels != set(LABELS):
        raise RuntimeError(
            f"Unexpected labels. Expected {LABELS}, found {sorted(labels)}."
        )

    print(f"[PASS] manifest rows      : {len(df)}")
    print(f"[PASS] unique File IDs     : {group_count}")
    print(f"[PASS] labels              : {LABELS}")
    print("[PASS] file_id loaded as string")

    if args.check_all:
        for index, row in df.iterrows():
            audio = resolve_audio(row)
            x, sr = load_audio(audio, target_sr=16000)
            logmel = make_logmel(x, sr)

            if logmel.shape != (64, 128):
                raise RuntimeError(
                    f"Bad tensor at row {index}: {logmel.shape}"
                )

            if not np.isfinite(logmel).all():
                raise RuntimeError(
                    f"Non-finite tensor at row {index}, file_id={row.file_id}"
                )

            if (index + 1) % 500 == 0:
                print(f"[INFO] validated {index + 1}/{len(df)}")

        print("[PASS] all 6283 WAV files decoded and transformed")
    else:
        # Fast representative check: one file per class.
        for label in LABELS:
            row = df[df["label"].astype(str) == label].iloc[0]
            x, sr = load_audio(resolve_audio(row), target_sr=16000)
            logmel = make_logmel(x, sr)
            if logmel.shape != (64, 128):
                raise RuntimeError(f"Representative check failed: {label}")

        print("[PASS] representative audio/log-mel checks passed")

    print("[PASS] official 725-file test set is not touched")


if __name__ == "__main__":
    main()
