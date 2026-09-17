#!/usr/bin/env python3
"""Prepare walking-only WFDB records into SafeBand V5.1 feature windows."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
# Make `python tools/script.py` work from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import wfdb
from ai.walking_features import extract_features

WINDOW_SEC = 8.0
SHIFT_SEC = 2.0
FS_EXPECTED = 256.0
META = {"subject","activity","record","start_sample","start_time","hr_bpm"}

def hr_from_peaks(peaks, start, end, fs):
    p = np.asarray(peaks, dtype=int)
    p = p[(p >= start) & (p < end)]
    if len(p) < 2:
        return np.nan
    rr = np.diff(p) / fs
    rr = rr[(rr >= 60/220) & (rr <= 60/30)]
    if len(rr) == 0:
        return np.nan
    return float(np.mean(60.0 / rr))

def walk_records(root):
    records = [x.strip() for x in (Path(root)/"RECORDS").read_text().splitlines() if x.strip()]
    return [r for r in records if r.endswith("_walk")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    rows, audit = [], []
    for rec in walk_records(root):
        path = str(root / rec)
        r = wfdb.rdrecord(path, physical=True)
        a = wfdb.rdann(path, "atr")
        names = list(r.sig_name)
        required = [
            "wrist_ppg",
            "wrist_gyro_x","wrist_gyro_y","wrist_gyro_z",
            "wrist_low_noise_accelerometer_x",
            "wrist_low_noise_accelerometer_y",
            "wrist_low_noise_accelerometer_z",
        ]
        missing = [x for x in required if x not in names]
        if missing:
            raise ValueError(f"{rec}: missing channels {missing}")
        if abs(float(r.fs) - FS_EXPECTED) > 1e-6:
            raise ValueError(f"{rec}: expected {FS_EXPECTED} Hz, got {r.fs}")
        idx = {n:i for i,n in enumerate(names)}
        ppg = r.p_signal[:, idx["wrist_ppg"]]
        gyro = r.p_signal[:, [idx["wrist_gyro_x"],idx["wrist_gyro_y"],idx["wrist_gyro_z"]]]
        acc = r.p_signal[:, [idx["wrist_low_noise_accelerometer_x"],
                             idx["wrist_low_noise_accelerometer_y"],
                             idx["wrist_low_noise_accelerometer_z"]]]
        peaks = np.asarray(a.sample, dtype=int)
        nwin = max(0, int(np.floor((len(ppg)-WINDOW_SEC*r.fs)/(SHIFT_SEC*r.fs)))+1)
        valid = 0
        for k in range(nwin):
            start = int(round(k*SHIFT_SEC*r.fs))
            end = start + int(round(WINDOW_SEC*r.fs))
            target = hr_from_peaks(peaks, start, end, float(r.fs))
            if not np.isfinite(target):
                continue
            feat = extract_features(ppg[start:end], acc[start:end], gyro[start:end], float(r.fs), float(r.fs))
            row = {"subject":rec.split("_")[0], "activity":"walk", "record":rec,
                   "start_sample":start, "start_time":start/float(r.fs), "hr_bpm":target}
            row.update(feat)
            rows.append(row); valid += 1
        audit.append({
            "record":rec, "subject":rec.split("_")[0], "fs_hz":float(r.fs),
            "samples":int(len(ppg)), "duration_sec":float(len(ppg)/r.fs),
            "annotation_count":int(len(peaks)), "windows_total":nwin,
            "windows_valid":valid, "channel_count":len(names)
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid walking windows produced.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    features = [c for c in df.columns if c not in META]
    summary = {"version":"PPG WALKING V5.1 PREPARATION",
               "window_sec":WINDOW_SEC,"shift_sec":SHIFT_SEC,"fs_hz":FS_EXPECTED,
               "records":audit,"rows":int(len(df)),
               "subjects":sorted(df.subject.unique().tolist()),"features":len(features)}
    Path(args.summary_out).write_text(json.dumps(summary,indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
