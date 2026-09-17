from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ai.ppg_signal import estimate_hr_bvp_only
from ai.ppg_motion_features import combined_quality_features


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    return {
        "n": int(y.size),
        "mae_bpm": float(mean_absolute_error(y, pred)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "within_5_bpm_pct": float(np.mean(np.abs(y - pred) <= 5.0) * 100.0),
        "within_10_bpm_pct": float(np.mean(np.abs(y - pred) <= 10.0) * 100.0),
    }


def subject_split(subjects):
    ids = np.array(sorted(set(subjects.tolist())))
    if len(ids) != 15:
        print(f"WARNING: expected 15 subjects, found {len(ids)}")
    if len(ids) < 3:
        raise ValueError("Need at least 3 subjects.")
    n = len(ids)
    n_train = min(max(round(n * 0.60), 1), n - 2)
    n_val = min(max(round(n * 0.20), 1), n - n_train - 1)
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


def make_features(bvp, acc, bvp_fs, acc_fs):
    base = sorted(combined_quality_features(
        bvp[0], acc[0], bvp_fs, acc_fs
    ).keys())
    names = base + ["fft_bpm", "peaks_bpm"]
    X = np.empty((len(bvp), len(names)), dtype=np.float32)

    for i in range(len(bvp)):
        f = combined_quality_features(bvp[i], acc[i], bvp_fs, acc_fs)
        f.update(estimate_hr_bvp_only(bvp[i], bvp_fs))
        X[i] = [f[k] for k in names]

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN/Inf.")
    return X, names


def main():
    ap = argparse.ArgumentParser(
        description="Train the PPG-DaLiA HR baseline model and save it."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    z = np.load(input_path, allow_pickle=True)
    required = {"bvp", "acc", "hr", "subject", "bvp_fs", "acc_fs"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"Missing arrays: {sorted(missing)}")

    bvp = z["bvp"]
    acc = z["acc"]
    y = z["hr"].astype(np.float32)
    subjects = z["subject"].astype(str)
    bvp_fs = float(z["bvp_fs"])
    acc_fs = float(z["acc_fs"])

    if bvp.ndim != 2 or bvp.shape[1] != int(round(8*bvp_fs)):
        raise ValueError(f"Unexpected BVP shape: {bvp.shape}")
    if acc.ndim != 3 or acc.shape[1:] != (int(round(8*acc_fs)), 3):
        raise ValueError(f"Unexpected ACC shape: {acc.shape}")
    if not (len(bvp) == len(acc) == len(y) == len(subjects)):
        raise ValueError("Window counts do not match.")
    if not (np.isfinite(bvp).all() and np.isfinite(acc).all() and np.isfinite(y).all()):
        raise ValueError("Input contains NaN/Inf.")

    train_s, val_s, test_s = subject_split(subjects)
    train_m = np.isin(subjects, train_s)
    val_m = np.isin(subjects, val_s)
    test_m = np.isin(subjects, test_s)

    if set(train_s) & set(val_s) or set(train_s) & set(test_s) or set(val_s) & set(test_s):
        raise RuntimeError("Subject leakage detected.")

    print(f"Dataset: {len(y)} windows, {len(set(subjects))} subjects")
    print(f"Train subjects: {train_s.tolist()} ({train_m.sum()} windows)")
    print(f"Val subjects:   {val_s.tolist()} ({val_m.sum()} windows)")
    print(f"Test subjects:  {test_s.tolist()} ({test_m.sum()} windows)")
    print("Building features ...")

    X, feature_names = make_features(bvp, acc, bvp_fs, acc_fs)

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X[train_m], y[train_m])
    val_pred = model.predict(X[val_m])
    val_metrics = metrics(y[val_m], val_pred)

    # The validation set is used only to confirm the frozen RF baseline.
    # Final deployment artifact is refit on TRAIN+VALIDATION.
    final_model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    refit_m = train_m | val_m
    final_model.fit(X[refit_m], y[refit_m])

    test_pred = final_model.predict(X[test_m])
    test_metrics = metrics(y[test_m], test_pred)

    # Signal-processing references are evaluated on the same held-out subjects.
    fft_pred = np.array([
        estimate_hr_bvp_only(x, bvp_fs)["fft_bpm"]
        for x in bvp[test_m]
    ], dtype=float)
    peak_pred = np.array([
        estimate_hr_bvp_only(x, bvp_fs)["peaks_bpm"]
        for x in bvp[test_m]
    ], dtype=float)

    baseline_metrics = {
        "bvp_fft": metrics(y[test_m], fft_pred),
        "bvp_peaks": metrics(y[test_m], peak_pred),
    }

    artifact = {
        "model": final_model,
        "feature_names": feature_names,
        "bvp_fs_hz": bvp_fs,
        "acc_fs_hz": acc_fs,
        "window_sec": 8.0,
        "shift_sec": 2.0,
        "source_sensor": "Empatica E4 wrist BVP + wrist ACC",
        "target": "ECG-derived HR BPM",
        "model_type": "RandomForestRegressor",
        "model_params": final_model.get_params(),
        "training_subjects": train_s.tolist(),
        "validation_subjects": val_s.tolist(),
        "test_subjects": test_s.tolist(),
        "domain_warning": "Not a MAX30102-trained model. Requires MAX30102 hardware validation.",
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)

    report = {
        "dataset": {
            "windows": int(len(y)),
            "subjects": int(len(set(subjects))),
            "bvp_shape": list(bvp.shape),
            "acc_shape": list(acc.shape),
            "bvp_fs_hz": bvp_fs,
            "acc_fs_hz": acc_fs,
        },
        "split": {
            "method": "deterministic_subject_level_60_20_20",
            "train_subjects": train_s.tolist(),
            "validation_subjects": val_s.tolist(),
            "test_subjects": test_s.tolist(),
            "train_windows": int(train_m.sum()),
            "validation_windows": int(val_m.sum()),
            "test_windows": int(test_m.sum()),
            "subject_overlap": False,
        },
        "validation_frozen_model": val_metrics,
        "held_out_test": {
            "bvp_fft": baseline_metrics["bvp_fft"],
            "bvp_peaks": baseline_metrics["bvp_peaks"],
            "random_forest": test_metrics,
        },
        "model_artifact": str(model_out),
        "notes": [
            "Random forest selected from the validated Stage-1 benchmark.",
            "Final artifact is refit on train+validation after test-blind validation.",
            "Metrics are HR regression metrics; classification accuracy is not appropriate.",
            "PPG-DaLiA uses Empatica E4 BVP, not MAX30102.",
            "This artifact is a research/reference model and must not be presented as MAX30102 validated.",
        ],
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nHR MODEL TRAINING COMPLETE")
    print(f"Validation RF: MAE={val_metrics['mae_bpm']:.3f} BPM")
    print("\nHeld-out TEST:")
    for name, m in {
        "BVP FFT": baseline_metrics["bvp_fft"],
        "BVP peaks": baseline_metrics["bvp_peaks"],
        "Random Forest": test_metrics,
    }.items():
        print(
            f"{name:14s} MAE={m['mae_bpm']:.3f} "
            f"RMSE={m['rmse_bpm']:.3f} "
            f"R2={m['r2']:.3f} "
            f"<=5BPM={m['within_5_bpm_pct']:.2f}% "
            f"<=10BPM={m['within_10_bpm_pct']:.2f}%"
        )
    print(f"\nModel:  {model_out}")
    print(f"Report: {report_out}")


if __name__ == "__main__":
    main()
