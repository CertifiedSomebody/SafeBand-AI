#!/usr/bin/env python3
"""Predict HR from one synchronized PPG V4.1 window."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.ppg_v4_features import extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ppg-npy", required=True, help="N x 6: distal R/IR/G, proximal R/IR/G")
    ap.add_argument("--acc-npy", required=True, help="N x 3 accelerometer")
    ap.add_argument("--gyro-npy", required=True, help="N x 3 gyroscope")
    ap.add_argument("--fs", type=float, default=500.0)
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    ppg = np.load(args.ppg_npy)
    acc = np.load(args.acc_npy)
    gyro = np.load(args.gyro_npy)

    features = extract_features(ppg, acc, gyro, args.fs, args.fs)
    X = np.asarray(
        [[features.get(k, 0.0) for k in artifact["feature_names"]]],
        dtype=float,
    )

    pred_raw = float(artifact["model"].predict(X)[0])
    calibration = artifact.get("calibration", {})
    if calibration.get("enabled"):
        pred = calibration["slope"] * pred_raw + calibration["intercept"]
    else:
        pred = pred_raw

    print("=== SafeBand PPG V4.1 HR prediction ===")
    print(f"Raw model prediction: {pred_raw:.2f} BPM")
    print(f"Final prediction:      {pred:.2f} BPM")
    print(f"Calibration enabled:   {bool(calibration.get('enabled'))}")
    print(f"Feature count:         {len(artifact['feature_names'])}")
    print(f"Source domain:         {artifact.get('source_sensor', 'unknown')}")
    print("WARNING: MAX30101-domain reference; not MAX30102 validation.")


if __name__ == "__main__":
    main()
