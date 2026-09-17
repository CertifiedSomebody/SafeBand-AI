#!/usr/bin/env python3
"""SafeBand PPG V6: leakage-safe, signal-quality-aware HR regression."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

META = {"subject", "activity", "record", "start_sample", "start_time", "hr_bpm"}
RNG = 42
BOOTSTRAP_N = 2000


def skey(value: str):
    value = str(value)
    return int(value[1:]) if value.startswith("s") and value[1:].isdigit() else 10**9


def split_subjects(ids):
    ordered = sorted(set(map(str, ids)), key=skey)
    n = len(ordered)
    n_train = max(1, int(round(n * 0.60)))
    n_val = max(1, int(round(n * 0.20)))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    return ordered[:n_train], ordered[n_train:n_train + n_val], ordered[n_train + n_val:]


def bland_altman(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mean = (y + pred) / 2.0
    diff = pred - y
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {
        "mean_difference_bpm": md,
        "loa_lower_bpm": float(md - 1.96 * sd),
        "loa_upper_bpm": float(md + 1.96 * sd),
        "mean_hr_bpm": float(np.mean(mean)),
        "sd_difference_bpm": sd,
    }

def bootstrap_ci(y, pred, metric="mae_bpm", n_boot=BOOTSTRAP_N, seed=RNG):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 2:
        return {"low": None, "high": None, "confidence": 0.95, "n_bootstrap": 0}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if metric == "mae_bpm":
            vals[i] = mean_absolute_error(y[idx], pred[idx])
        elif metric == "rmse_bpm":
            vals[i] = np.sqrt(mean_squared_error(y[idx], pred[idx]))
        elif metric == "within_5_bpm_pct":
            vals[i] = np.mean(np.abs(y[idx] - pred[idx]) <= 5) * 100.0
        else:
            raise ValueError(metric)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "confidence": 0.95, "n_bootstrap": n_boot}

def scores(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = np.abs(y - pred)
    return {
        "n": int(len(y)),
        "mae_bpm": float(mean_absolute_error(y, pred)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "within_3_bpm_pct": float(np.mean(err <= 3) * 100),
        "within_5_bpm_pct": float(np.mean(err <= 5) * 100),
        "within_10_bpm_pct": float(np.mean(err <= 10) * 100),
        "bias_bpm": float(np.mean(pred - y)),
        "median_abs_error_bpm": float(np.median(err)),
        "p90_abs_error_bpm": float(np.percentile(err, 90)),
        "p95_abs_error_bpm": float(np.percentile(err, 95)),
        "max_abs_error_bpm": float(np.max(err)),
    }


def make_pipeline(kind):
    if kind == "ridge":
        model = Ridge(alpha=10.0)
    elif kind == "rf":
        model = RandomForestRegressor(
            n_estimators=700, min_samples_leaf=2, max_features=0.75,
            random_state=RNG, n_jobs=-1,
        )
    elif kind == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=700, min_samples_leaf=2, max_features=0.80,
            random_state=RNG, n_jobs=-1,
        )
    elif kind == "hgb":
        model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.035, max_leaf_nodes=31,
            l2_regularization=2.0, random_state=RNG,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def affine(y_true, pred):
    A = np.column_stack([pred, np.ones(len(pred))])
    coef, *_ = np.linalg.lstsq(A, y_true, rcond=None)
    return float(coef[0]), float(coef[1])


def temporal_by_record(pred_df, alpha=0.35, max_jump=20.0):
    """Causal bounded smoothing; state is reset independently for each record."""
    result = np.empty(len(pred_df), dtype=float)
    position = {idx: i for i, idx in enumerate(pred_df.index)}
    for _, group in pred_df.groupby("record", sort=False):
        previous = None
        for idx in group.index:
            raw = float(pred_df.at[idx, "prediction_raw"])
            if previous is None:
                stabilized = raw
            else:
                bounded = float(np.clip(raw, previous - max_jump, previous + max_jump))
                stabilized = float(alpha * bounded + (1.0 - alpha) * previous)
            result[position[idx]] = stabilized
            previous = stabilized
    return result


def feature_groups(features):
    motion = [
        c for c in features
        if c.startswith(("acc_", "gyro_", "motion_"))
        or "_accmag" in c or "_gyromag" in c
    ]
    ppg_only = [c for c in features if c not in motion]
    candidate_quality = [
        c for c in features
        if c in {
            "consensus_hr", "consensus_spread_bpm", "best_channel_sqi",
            "mean_channel_sqi", "max_channel_sqi", "channel_sqi_std",
            "best_channel_index",
        }
        or c.endswith(("_candidate_hr", "_candidate_spread", "_autocorr_hr",
                       "_autocorr_peak", "_dom_bpm", "_peak_hr", "_peak_hr_iqr",
                       "_rr_cv", "_peak_snr_db", "_harmonic_ratio", "_sqi"))
    ]
    return {
        "candidate_quality": [c for c in candidate_quality if c in features],
        "ppg_only": ppg_only,
        "ppg_plus_motion": list(features),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--report-out", required=True)
    ap.add_argument("--predictions-out", required=True)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--max-jump-bpm", type=float, default=20.0)
    ap.add_argument("--disable-calibration", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = META - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input table is empty.")

    features = [c for c in df.columns if c not in META]
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df["hr_bpm"], errors="coerce").to_numpy(float)
    if not np.isfinite(X_df.to_numpy(float)).all() or not np.isfinite(y).all():
        raise ValueError("Input contains NaN/Inf after numeric conversion.")

    ids = df["subject"].astype(str).to_numpy()
    train_subjects, val_subjects, test_subjects = split_subjects(ids)
    train = np.isin(ids, train_subjects)
    val = np.isin(ids, val_subjects)
    test = np.isin(ids, test_subjects)
    if not train.any() or not val.any() or not test.any():
        raise RuntimeError("Subject split produced an empty partition.")
    if set(train_subjects) & set(val_subjects) or set(train_subjects) & set(test_subjects) or set(val_subjects) & set(test_subjects):
        raise RuntimeError("Subject leakage detected.")

    X = X_df.to_numpy(float)
    groups = feature_groups(features)
    print(f"Rows={len(df)} features={len(features)}")
    print(f"Train={train_subjects} | Val={val_subjects} | Test={test_subjects}")

    train_mean = float(y[train].mean())
    baseline = {
        "validation": scores(y[val], np.full(val.sum(), train_mean)),
        "test": scores(y[test], np.full(test.sum(), train_mean)),
    }

    validation_results = {}
    for group_name, columns in groups.items():
        if not columns:
            continue
        idx = [features.index(c) for c in columns]
        for model_name in ("ridge", "rf", "extra_trees", "hgb"):
            model = make_pipeline(model_name)
            model.fit(X[train][:, idx], y[train])
            pred = model.predict(X[val][:, idx])
            validation_results[f"{group_name}_{model_name}"] = {
                "group": group_name,
                "model": model_name,
                "feature_count": len(columns),
                "metrics": scores(y[val], pred),
            }

    best_key = min(validation_results, key=lambda k: validation_results[k]["metrics"]["mae_bpm"])
    best = validation_results[best_key]
    selected_columns = groups[best["group"]]
    selected_idx = [features.index(c) for c in selected_columns]

    # Calibration is fitted only on validation data. It is used only if it
    # improves validation MAE, and can be disabled for an explicit ablation.
    val_model = make_pipeline(best["model"])
    val_model.fit(X[train][:, selected_idx], y[train])
    val_raw = val_model.predict(X[val][:, selected_idx])
    slope, intercept = affine(y[val], val_raw)
    val_calibrated = slope * val_raw + intercept
    raw_val_mae = mean_absolute_error(y[val], val_raw)
    cal_val_mae = mean_absolute_error(y[val], val_calibrated)
    calibration_enabled = (not args.disable_calibration) and cal_val_mae < raw_val_mae

    # Final model sees only train+validation. The held-out subjects are untouched.
    final_model = make_pipeline(best["model"])
    final_model.fit(X[train | val][:, selected_idx], y[train | val])
    raw_test = final_model.predict(X[test][:, selected_idx])

    pred = df.loc[test, ["subject", "activity", "record", "start_sample", "start_time", "hr_bpm"]].copy().reset_index(drop=True)
    pred["prediction_model"] = raw_test
    if calibration_enabled:
        pred["prediction_raw"] = slope * pred["prediction_model"] + intercept
    else:
        pred["prediction_raw"] = pred["prediction_model"]
    pred["prediction_temporal"] = temporal_by_record(pred, args.alpha, args.max_jump_bpm)
    pred["error_raw_bpm"] = pred["prediction_raw"] - pred["hr_bpm"]
    pred["error_temporal_bpm"] = pred["prediction_temporal"] - pred["hr_bpm"]

    test_raw = scores(pred["hr_bpm"], pred["prediction_raw"])
    test_temporal = scores(pred["hr_bpm"], pred["prediction_temporal"])
    raw_ba = bland_altman(pred["hr_bpm"], pred["prediction_raw"])
    temporal_ba = bland_altman(pred["hr_bpm"], pred["prediction_temporal"])
    raw_ci = {m: bootstrap_ci(pred["hr_bpm"], pred["prediction_raw"], m) for m in ("mae_bpm", "rmse_bpm", "within_5_bpm_pct")}
    temporal_ci = {m: bootstrap_ci(pred["hr_bpm"], pred["prediction_temporal"], m) for m in ("mae_bpm", "rmse_bpm", "within_5_bpm_pct")}

    def grouped(column, prediction_column):
        return {
            str(key): scores(group["hr_bpm"], group[prediction_column])
            for key, group in pred.groupby(column, sort=True)
        }

    report = {
        "version": "PPG V6 FINAL",
        "domain": "MAX30101-family PTT reference; MAX30102 validation pending",
        "objective": "Improve cross-subject HR robustness using cached signal processing, multi-channel consensus, explicit SQI features, motion contamination indicators, and conservative temporal stabilization.",
        "dataset": {
            "windows": len(df),
            "subjects": int(df["subject"].nunique()),
            "features": len(features),
            "ppg_features": len(groups["ppg_only"]),
            "motion_features": len(features) - len(groups["ppg_only"]),
            "activities": sorted(df["activity"].unique().tolist()),
        },
        "split": {
            "method": "numeric_subject_60_20_20",
            "train_subjects": train_subjects,
            "validation_subjects": val_subjects,
            "test_subjects": test_subjects,
            "subject_overlap": False,
        },
        "baselines": {"constant_train_mean_bpm": train_mean, "constant": baseline},
        "validation_model_search": validation_results,
        "selected": {
            "group": best["group"],
            "model": best["model"],
            "validation_mae_bpm": best["metrics"]["mae_bpm"],
            "feature_count": len(selected_columns),
            "feature_columns": selected_columns,
        },
        "validation_calibration": {
            "enabled": calibration_enabled,
            "slope": slope,
            "intercept": intercept,
            "raw_mae_bpm": float(raw_val_mae),
            "calibrated_mae_bpm": float(cal_val_mae),
        },
        "held_out_test_raw": test_raw,
        "held_out_test_temporal": test_temporal,
        "held_out_test_raw_bootstrap_95ci": raw_ci,
        "held_out_test_temporal_bootstrap_95ci": temporal_ci,
        "bland_altman_raw": raw_ba,
        "bland_altman_temporal": temporal_ba,
        "held_out_test_by_activity_raw": grouped("activity", "prediction_raw"),
        "held_out_test_by_activity_temporal": grouped("activity", "prediction_temporal"),
        "held_out_test_by_subject_temporal": grouped("subject", "prediction_temporal"),
        "temporal": {"alpha": args.alpha, "max_jump_bpm": args.max_jump_bpm, "reset_per_record": True},
        "artifact": str(args.model_out),
        "predictions": str(args.predictions_out),
        "notes": [
            "Regression task: classification accuracy is not the primary metric.",
            "ECG waveform is not a feature; ECG R-peaks provide the HR target.",
            "SpO2 is not used as a continuous target.",
            "Test subjects are never used for model selection or calibration.",
            "Temporal stabilization is post-processing and resets at each recording.",
            "Raw and temporally stabilized test metrics are both reported; smoothing is not allowed to hide raw-model degradation.",
            "This result must not be called MAX30102 production validation. Real MAX30102 raw RED/IR recordings remain required.",
        ],
    }

    for path in (args.model_out, args.report_out, args.predictions_out):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": final_model,
        "feature_names": selected_columns,
        "model_type": best["model"],
        "calibration": {"enabled": calibration_enabled, "slope": slope, "intercept": intercept},
        "temporal": {"alpha": args.alpha, "max_jump_bpm": args.max_jump_bpm, "reset_per_record": True},
        "preprocessing": {"bandpass_hz": [0.5, 5.0], "window_sec": 8.0, "shift_sec": 2.0},
        "version": "PPG V6 FINAL",
        "domain_warning": "MAX30101-family reference; validate on MAX30102 hardware.",
    }, args.model_out)
    pred.to_csv(args.predictions_out, index=False)
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== PPG V6.1 TEST ===")
    print(json.dumps({"selected": report["selected"], "raw": test_raw, "temporal": test_temporal}, indent=2))


if __name__ == "__main__":
    main()
