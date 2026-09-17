#!/usr/bin/env python3
"""Train/evaluate SafeBand PPG V4.1 on PTT/MAX30101-domain features.

Important:
- split is subject-level, never random-window-level
- model selection uses validation MAE only
- held-out test subjects are untouched until final evaluation
- calibration is fitted on validation predictions only and retained only if
  validation MAE improves
- ECG and SpO2 labels are not model features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


META = {"subject", "activity", "record", "start_sample", "start_time", "hr_bpm"}


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        raise ValueError("No finite samples for metrics.")
    e = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae_bpm": float(mean_absolute_error(y, p)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else None,
        "within_3_bpm_pct": float(np.mean(e <= 3.0) * 100.0),
        "within_5_bpm_pct": float(np.mean(e <= 5.0) * 100.0),
        "within_10_bpm_pct": float(np.mean(e <= 10.0) * 100.0),
        "bias_bpm": float(np.mean(p - y)),
        "median_abs_error_bpm": float(np.median(e)),
        "max_abs_error_bpm": float(np.max(e)),
    }


def numeric_subjects(values) -> np.ndarray:
    ids = sorted(
        {str(v) for v in values},
        key=lambda s: int(s[1:]) if s.startswith("s") and s[1:].isdigit() else s,
    )
    return np.asarray(ids, dtype=str)


def split_subjects(subjects: np.ndarray):
    ids = numeric_subjects(subjects)
    if len(ids) < 6:
        raise ValueError("Need at least 6 subjects for a 60/20/20 split.")

    n = len(ids)
    n_train = max(1, round(n * 0.60))
    n_val = max(1, round(n * 0.20))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def build_models():
    return {
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "rf": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=700,
                        min_samples_leaf=2,
                        max_features=0.75,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=700,
                        min_samples_leaf=2,
                        max_features=0.80,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=450,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=1.5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def fit_affine(y_true, pred):
    A = np.column_stack([np.asarray(pred, float), np.ones(len(pred))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y_true, float), rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--predictions-out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = META - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_cols = [c for c in df.columns if c not in META]
    if not feature_cols:
        raise ValueError("No model features found.")

    subjects = df["subject"].astype(str).to_numpy()
    activities = df["activity"].astype(str).to_numpy()
    y = pd.to_numeric(df["hr_bpm"], errors="coerce").to_numpy(dtype=float)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Input contains NaN/Inf. Re-run preparation.")

    train_s, val_s, test_s = split_subjects(subjects)
    tr = np.isin(subjects, train_s)
    va = np.isin(subjects, val_s)
    te = np.isin(subjects, test_s)

    if (set(train_s) & set(val_s)) or (set(train_s) & set(test_s)) or (set(val_s) & set(test_s)):
        raise RuntimeError("Subject leakage detected.")

    print("=== PPG V4.1 TRAINING ===")
    print(f"Rows/features: {len(df)} / {len(feature_cols)}")
    print(f"Subjects: {len(set(subjects))}")
    print(f"TRAIN subjects: {train_s.tolist()} | windows={int(tr.sum())}")
    print(f"VAL   subjects: {val_s.tolist()} | windows={int(va.sum())}")
    print(f"TEST  subjects: {test_s.tolist()} | windows={int(te.sum())}")
    print()

    models = build_models()
    val_scores = {}
    val_predictions = {}

    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X[tr], y[tr])
        pv = model.predict(X[va])
        val_predictions[name] = pv
        val_scores[name] = metrics(y[va], pv)
        print(
            f"  VAL MAE={val_scores[name]['mae_bpm']:.3f} | "
            f"RMSE={val_scores[name]['rmse_bpm']:.3f} | "
            f"R2={val_scores[name]['r2']:.3f} | "
            f"±5={val_scores[name]['within_5_bpm_pct']:.2f}%"
        )

    selected = min(val_scores, key=lambda n: val_scores[n]["mae_bpm"])
    raw_val = val_predictions[selected]
    raw_mae = mean_absolute_error(y[va], raw_val)

    slope, intercept = fit_affine(y[va], raw_val)
    calibrated_val = slope * raw_val + intercept
    calibrated_mae = mean_absolute_error(y[va], calibrated_val)
    calibration_enabled = bool(calibrated_mae + 1e-12 < raw_mae)

    calibration = {
        "enabled": calibration_enabled,
        "slope": slope,
        "intercept": intercept,
        "validation_raw_mae_bpm": float(raw_mae),
        "validation_calibrated_mae_bpm": float(calibrated_mae),
        "selection_rule": "retain only if validation MAE strictly improves",
    }

    # Refit only after model selection. Test remains untouched.
    final_model = models[selected]
    final_model.fit(X[tr | va], y[tr | va])
    test_pred_raw = final_model.predict(X[te])
    test_pred = slope * test_pred_raw + intercept if calibration_enabled else test_pred_raw
    test_scores = metrics(y[te], test_pred)

    # Store test predictions for the independent diagnostic script.
    pred_df = df.loc[te, ["subject", "activity", "record", "start_sample", "start_time", "hr_bpm"]].copy()
    pred_df["prediction_bpm"] = test_pred
    pred_df["abs_error_bpm"] = np.abs(pred_df["hr_bpm"] - pred_df["prediction_bpm"])
    pred_df["error_bpm"] = pred_df["prediction_bpm"] - pred_df["hr_bpm"]

    mo = Path(args.model_out)
    mo.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_model,
        "feature_names": feature_cols,
        "model_type": selected,
        "sample_rate_hz": 500.0,
        "window_sec": 8.0,
        "shift_sec": 2.0,
        "source_sensor": "Maxim MAX30101 (PTT dataset) + MPU-9250 IMU",
        "target": "ECG R-peak-derived HR BPM",
        "calibration": calibration,
        "training_subjects": train_s.tolist(),
        "validation_subjects": val_s.tolist(),
        "test_subjects": test_s.tolist(),
        "version": "PPG V4.1",
        "domain_warning": (
            "MAX30101-domain reference model. This is not MAX30102 validation. "
            "Real MAX30102 recordings are required for final hardware validation."
        ),
    }
    joblib.dump(artifact, mo)

    if args.predictions_out:
        pp = Path(args.predictions_out)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(pp, index=False)

    report = {
        "version": "PPG V4.1",
        "dataset": {
            "windows": int(len(df)),
            "subjects": int(len(set(subjects))),
            "records": int(df["record"].nunique()),
            "activities": sorted(df["activity"].unique().tolist()),
        },
        "target_summary": {
            "train": metrics(y[tr], np.full(tr.sum(), np.mean(y[tr]))),
            "validation": metrics(y[va], np.full(va.sum(), np.mean(y[tr]))),
            "test": metrics(y[te], np.full(te.sum(), np.mean(y[tr]))),
        },
        "split": {
            "method": "deterministic_numeric_subject_level_60_20_20",
            "train_subjects": train_s.tolist(),
            "validation_subjects": val_s.tolist(),
            "test_subjects": test_s.tolist(),
            "subject_overlap": False,
            "train_windows": int(tr.sum()),
            "validation_windows": int(va.sum()),
            "test_windows": int(te.sum()),
        },
        "validation": val_scores,
        "selected_model": selected,
        "validation_calibration": calibration,
        "held_out_test": test_scores,
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "artifact": str(mo),
        "predictions": str(args.predictions_out) if args.predictions_out else None,
        "metric_guide": {
            "mae_bpm": "Mean absolute HR error; lower is better.",
            "rmse_bpm": "Root mean squared HR error; lower is better and penalizes large errors more.",
            "r2": "Variance explained relative to a constant-mean baseline; higher is better, but can be negative.",
            "within_3_bpm_pct": "Percentage of predictions within 3 BPM of ECG-derived HR; higher is better.",
            "within_5_bpm_pct": "Percentage within 5 BPM; higher is better.",
            "within_10_bpm_pct": "Percentage within 10 BPM; higher is better.",
            "bias_bpm": "Mean signed prediction error; near zero means less systematic over/under prediction.",
            "median_abs_error_bpm": "Median absolute HR error; lower is better.",
            "max_abs_error_bpm": "Largest absolute HR error in the evaluated split; lower is better.",
            "regression_accuracy_note": "This is a regression task, so classification accuracy is not an appropriate primary metric.",
        },
        "notes": [
            "ECG waveform is not a model input; ECG R peaks provide the HR target only.",
            "SpO2 start/end numerics are not expanded into continuous window labels.",
            "PPG channels are processed per window; AC/DC features are representation features, not a clinical SpO2 equation.",
            "Model selection is performed using validation MAE only.",
            "The held-out test subjects are evaluated only after model selection and final refitting.",
            "V4.1 uses numeric subject ordering to avoid lexicographic s1,s10,... ordering.",
            "Real MAX30102 recordings remain required for final hardware validation and calibration.",
        ],
    }

    ro = Path(args.report_out)
    ro.parent.mkdir(parents=True, exist_ok=True)
    ro.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RESULT ===")
    print(f"Selected model: {selected}")
    print(f"VAL MAE: {val_scores[selected]['mae_bpm']:.3f} BPM")
    print(f"TEST MAE: {test_scores['mae_bpm']:.3f} BPM")
    print(f"TEST RMSE: {test_scores['rmse_bpm']:.3f} BPM")
    print(f"TEST R2: {test_scores['r2']:.3f}")
    print(f"TEST ±3 BPM: {test_scores['within_3_bpm_pct']:.2f}%")
    print(f"TEST ±5 BPM: {test_scores['within_5_bpm_pct']:.2f}%")
    print(f"TEST ±10 BPM: {test_scores['within_10_bpm_pct']:.2f}%")
    print(f"TEST bias: {test_scores['bias_bpm']:+.3f} BPM")
    print(f"Calibration enabled: {calibration_enabled}")
    print(f"Model: {mo}")
    print(f"Report: {ro}")


if __name__ == "__main__":
    main()
