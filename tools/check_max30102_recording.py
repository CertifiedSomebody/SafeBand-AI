"""Validate a future SafeBand MAX30102 recording before model training.

Expected CSV columns:
    timestamp_ms, ir
Optional:
    red, acc_x, acc_y, acc_z

This checker deliberately does NOT convert E4 BVP into MAX30102 data.
It is a hardware-data gate for future real SafeBand recordings.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    required = {"timestamp_ms", "ir"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required MAX30102 columns: {sorted(missing)}")

    for c in ["timestamp_ms", "ir", "red", "acc_x", "acc_y", "acc_z"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[["timestamp_ms", "ir"]].isna().any().any():
        raise ValueError("timestamp_ms/ir contains invalid values.")

    t = df["timestamp_ms"].to_numpy(dtype=float)
    if len(t) < 10:
        raise ValueError("Recording is too short.")

    dt = np.diff(t) / 1000.0
    positive = dt[dt > 0]
    if positive.size == 0:
        raise ValueError("No increasing timestamps.")

    fs = 1.0 / np.median(positive)
    print(f"Rows: {len(df)}")
    print(f"Estimated IR sampling rate: {fs:.3f} Hz")
    print(f"IR range: [{df.ir.min():.3f}, {df.ir.max():.3f}]")
    print(f"Columns: {df.columns.tolist()}")

    if not 10 <= fs <= 1000:
        print("WARNING: estimated sampling rate is unusual; inspect hardware logging.")

    print("MAX30102 recording schema check: PASS")


if __name__ == "__main__":
    main()
