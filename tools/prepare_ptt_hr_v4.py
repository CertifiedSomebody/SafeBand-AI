#!/usr/bin/env python3
"""Prepare PhysioNet PTT/MAX30101-domain data for SafeBand PPG V4.2.

V4.2 keeps the V4.1 feature representation and adds stronger input validation.
It is intentionally conservative: no SpO2 labels are converted into continuous
window targets, and ECG waveform is never used as a feature.

Supported record names: sXX_run.csv, sXX_sit.csv, sXX_walk.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ppg_v4_features import extract_features

FS_HZ = 500.0
PPG_COLS = ["pleth_1", "pleth_2", "pleth_3", "pleth_4", "pleth_5", "pleth_6"]
ACC_COLS = ["a_x", "a_y", "a_z"]
GYRO_COLS = ["g_x", "g_y", "g_z"]
REQUIRED = ["time", "peaks", *PPG_COLS, *ACC_COLS, *GYRO_COLS]
RECORD_RE = re.compile(r"^s(?P<subject>\d+)_(?P<activity>run|sit|walk)$", re.IGNORECASE)


def parse_record_name(path: Path) -> tuple[str, str]:
    """Parse record name without fragile group-number assumptions."""
    match = RECORD_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(
            f"Unexpected record filename '{path.name}'. "
            "Expected sXX_run.csv, sXX_sit.csv or sXX_walk.csv."
        )
    subject = f"s{int(match.group('subject'))}"
    activity = match.group("activity").lower()
    return subject, activity


def ecg_hr_from_peaks(peaks: np.ndarray, fs: float) -> float | None:
    """Derive target HR from ECG R-peak annotation samples."""
    x = np.asarray(peaks)
    idx = np.flatnonzero(x > 0)
    if idx.size < 3:
        return None
    rr = np.diff(idx).astype(float) / fs
    rr = rr[(rr >= 60.0 / 220.0) & (rr <= 60.0 / 30.0)]
    if rr.size < 2:
        return None
    hr = 60.0 / float(np.median(rr))
    return hr if 30.0 <= hr <= 220.0 else None


def process_record(path: Path, window_sec: float, shift_sec: float):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")

    subject, activity = parse_record_name(path)

    numeric_cols = ["peaks", *PPG_COLS, *ACC_COLS, *GYRO_COLS]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"{path.name}: NaN/non-numeric values in {bad}")

    w = int(round(window_sec * FS_HZ))
    step = int(round(shift_sec * FS_HZ))
    if w <= 0 or step <= 0 or step > w:
        raise ValueError("Require 0 < shift_sec <= window_sec.")

    rows = []
    skipped = 0
    possible = max(0, (len(df) - w) // step + 1)

    for start in range(0, len(df) - w + 1, step):
        stop = start + w
        p = df.iloc[start:stop]
        hr = ecg_hr_from_peaks(p["peaks"].to_numpy(), FS_HZ)
        if hr is None:
            skipped += 1
            continue

        features = extract_features(
            p[PPG_COLS].to_numpy(float),
            p[ACC_COLS].to_numpy(float),
            p[GYRO_COLS].to_numpy(float),
            FS_HZ,
            FS_HZ,
        )
        features.update({
            "subject": subject,
            "activity": activity,
            "record": path.stem,
            "start_sample": int(start),
            "start_time": str(p["time"].iloc[0]),
            "hr_bpm": float(hr),
        })
        rows.append(features)

    return rows, {
        "record": path.stem,
        "subject": subject,
        "activity": activity,
        "raw_samples": int(len(df)),
        "possible_windows": int(possible),
        "valid_hr_windows": int(len(rows)),
        "skipped_no_hr": int(skipped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--shift-sec", type=float, default=2.0)
    args = ap.parse_args()

    root = Path(args.ptt_root)
    csv_dir = root / "csv"
    if not csv_dir.is_dir():
        csv_dir = root / "CSV"
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"Neither csv nor CSV exists under {root}")

    records = []
    for path in csv_dir.glob("s*.csv"):
        try:
            parse_record_name(path)
        except ValueError:
            continue
        records.append(path)
    records.sort(key=lambda p: (
        parse_record_name(p)[0][1:].zfill(6),
        parse_record_name(p)[1],
    ))
    if not records:
        raise FileNotFoundError(f"No valid sXX_(run|sit|walk).csv records found in {csv_dir}")

    all_rows, summaries = [], []
    for i, path in enumerate(records, 1):
        print(f"[{i:02d}/{len(records):02d}] {path.name}")
        rows, summary = process_record(path, args.window_sec, args.shift_sec)
        all_rows.extend(rows)
        summaries.append(summary)

    if not all_rows:
        raise RuntimeError("No valid windows produced.")

    df = pd.DataFrame(all_rows)
    meta = ["subject", "activity", "record", "start_sample", "start_time", "hr_bpm"]
    feature_cols = sorted(c for c in df.columns if c not in meta)

    values = df[feature_cols + ["hr_bpm"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise RuntimeError("Prepared table contains NaN/Inf.")

    out = Path(args.out)
    summary_out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    df[meta + feature_cols].to_csv(out, index=False)

    summary = {
        "version": "PPG V4.2",
        "dataset": "Pulse Transit Time PPG Dataset",
        "records_found": len(records),
        "records_processed": len(summaries),
        "windows": len(df),
        "subjects": int(df.subject.nunique()),
        "features": len(feature_cols),
        "activities": sorted(df.activity.unique().tolist()),
        "window_sec": args.window_sec,
        "shift_sec": args.shift_sec,
        "sample_rate_hz": FS_HZ,
        "feature_columns": feature_cols,
        "records": summaries,
        "ecg_waveform_used_as_feature": False,
        "ecg_peaks_used_as_target_source": True,
        "spo2_used_as_target": False,
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== PREPARATION COMPLETE ===")
    print(f"Records:  {len(records)}")
    print(f"Subjects: {df.subject.nunique()}")
    print(f"Windows:  {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print("\nActivity counts:")
    print(df.groupby("activity").size().to_string())
    print("\nHR target:")
    print(df.hr_bpm.describe().to_string())
    print(f"\nCSV:     {out}")
    print(f"Summary: {summary_out}")


if __name__ == "__main__":
    main()
