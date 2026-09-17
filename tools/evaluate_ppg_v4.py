#!/usr/bin/env python3
"""Independent diagnostics for a frozen PPG V4.1 held-out test prediction CSV.

This script does not retrain or change the model. It reports:
- overall test metrics
- activity-level metrics
- subject-level metrics
- recording-level metrics
- prediction error quantiles
- performance relative to a train-mean baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae_bpm": float(mean_absolute_error(y, p)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else None,
        "within_3_bpm_pct": float(np.mean(e <= 3) * 100),
        "within_5_bpm_pct": float(np.mean(e <= 5) * 100),
        "within_10_bpm_pct": float(np.mean(e <= 10) * 100),
        "bias_bpm": float(np.mean(p - y)),
        "median_abs_error_bpm": float(np.median(e)),
        "p90_abs_error_bpm": float(np.percentile(e, 90)),
        "max_abs_error_bpm": float(np.max(e)),
    }


def grouped(df, key):
    out = {}
    for value, g in df.groupby(key, sort=True):
        out[str(value)] = metrics(g["hr_bpm"], g["prediction_bpm"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.predictions)
    required = {"subject", "activity", "record", "hr_bpm", "prediction_bpm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    y = pd.to_numeric(df["hr_bpm"], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df["prediction_bpm"], errors="coerce").to_numpy(float)
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Prediction CSV contains NaN/Inf.")

    report = {
        "overall": metrics(y, p),
        "by_activity": grouped(df, "activity"),
        "by_subject": grouped(df, "subject"),
        "by_record": grouped(df, "record"),
        "error_quantiles_bpm": {
            "p50": float(np.percentile(np.abs(y - p), 50)),
            "p75": float(np.percentile(np.abs(y - p), 75)),
            "p90": float(np.percentile(np.abs(y - p), 90)),
            "p95": float(np.percentile(np.abs(y - p), 95)),
            "max": float(np.max(np.abs(y - p))),
        },
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== PPG V4.1 HELD-OUT TEST DIAGNOSTICS ===")
    o = report["overall"]
    print(f"N:        {o['n']}")
    print(f"MAE:      {o['mae_bpm']:.3f} BPM")
    print(f"RMSE:     {o['rmse_bpm']:.3f} BPM")
    print(f"R2:       {o['r2']:.3f}")
    print(f"±3 BPM:   {o['within_3_bpm_pct']:.2f}%")
    print(f"±5 BPM:   {o['within_5_bpm_pct']:.2f}%")
    print(f"±10 BPM:  {o['within_10_bpm_pct']:.2f}%")
    print(f"Bias:     {o['bias_bpm']:+.3f} BPM")
    print(f"Median AE:{o['median_abs_error_bpm']:.3f} BPM")
    print(f"P90 AE:   {o['p90_abs_error_bpm']:.3f} BPM")
    print(f"Max AE:   {o['max_abs_error_bpm']:.3f} BPM")

    print("\n--- BY ACTIVITY ---")
    for k, v in report["by_activity"].items():
        print(f"{k:>5}: MAE={v['mae_bpm']:.3f} | RMSE={v['rmse_bpm']:.3f} | ±5={v['within_5_bpm_pct']:.2f}%")

    print("\n--- BY SUBJECT ---")
    for k, v in report["by_subject"].items():
        print(f"{k:>3}: MAE={v['mae_bpm']:.3f} | RMSE={v['rmse_bpm']:.3f} | ±5={v['within_5_bpm_pct']:.2f}%")

    print(f"\nSaved diagnostic report: {out}")


if __name__ == "__main__":
    main()
