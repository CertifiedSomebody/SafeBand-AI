from __future__ import annotations

# Required when launched as:
# python tools\benchmark_ppgdalia_hr.py
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ai.ppg_signal import estimate_hr_bvp_only
from ai.ppg_motion_features import combined_quality_features


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if y.size == 0:
        return {"n": 0, "mae_bpm": None, "rmse_bpm": None, "r2": None}
    return {
        "n": int(y.size),
        "mae_bpm": float(mean_absolute_error(y, pred)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if y.size >= 2 else None,
    }


def subject_split(subjects):
    """Deterministic subject-level 60/20/20 split."""
    ids = np.array(sorted(set(subjects.tolist())))
    if ids.size < 3:
        raise ValueError(f"Need at least 3 subjects; found {ids.size}.")
    n = ids.size
    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.20))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    return ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]


def smoke_indices_by_subject(subjects, max_windows):
    """Return a deterministic, approximately equal-per-subject smoke subset.

    This is deliberately performed BEFORE train/val/test splitting. It avoids
    the invalid situation where a prefix of a subject-contiguous dataset
    contains nearly all windows from the first few subjects.
    """
    n_total = len(subjects)
    if max_windows <= 0 or max_windows >= n_total:
        return np.arange(n_total, dtype=int)

    ids = np.array(sorted(set(subjects.tolist())))
    if ids.size < 3:
        raise ValueError("Smoke test requires at least 3 subjects.")

    # Equal allocation, distributing remainder over the first subjects.
    base = max_windows // ids.size
    remainder = max_windows % ids.size
    if base < 1:
        raise ValueError(
            f"--max-windows={max_windows} is too small for {ids.size} subjects. "
            f"Use at least {ids.size}."
        )

    selected = []
    for rank, sid in enumerate(ids):
        positions = np.flatnonzero(subjects == sid)
        quota = base + (1 if rank < remainder else 0)
        if quota > len(positions):
            raise ValueError(
                f"Subject {sid} has only {len(positions)} windows, "
                f"but smoke sampler requested {quota}."
            )
        # Evenly spaced within the subject, rather than taking a contiguous prefix.
        local = np.linspace(0, len(positions) - 1, quota, dtype=int)
        selected.extend(positions[local].tolist())

    idx = np.asarray(selected, dtype=int)
    if len(idx) != max_windows or len(np.unique(idx)) != max_windows:
        raise RuntimeError("Smoke sampler produced an invalid number of unique windows.")
    return np.sort(idx)


def build_features(bvp, acc, bvp_fs, acc_fs):
    quality_names = sorted(
        combined_quality_features(bvp[0], acc[0], bvp_fs, acc_fs).keys()
    )
    names = quality_names + ["fft_bpm", "peaks_bpm"]
    X = np.empty((len(bvp), len(names)), dtype=np.float32)

    for i in range(len(bvp)):
        feat = combined_quality_features(bvp[i], acc[i], bvp_fs, acc_fs)
        feat.update(estimate_hr_bvp_only(bvp[i], bvp_fs))
        X[i] = [feat[k] for k in names]

    if not np.isfinite(X).all():
        raise ValueError("Generated feature matrix contains NaN/Inf.")
    return X, names


def main():
    ap = argparse.ArgumentParser(
        description="Subject-independent PPG-DaLiA HR benchmark."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--max-windows", type=int, default=0,
        help="Smoke-test cap distributed approximately equally across subjects; "
             "0 = full dataset."
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    z = np.load(input_path, allow_pickle=True)
    required = {"bvp", "acc", "hr", "subject", "bvp_fs", "acc_fs"}
    missing = required - set(z.files)
    if missing:
        raise ValueError(f"Missing required arrays: {sorted(missing)}")

    bvp = z["bvp"]
    acc = z["acc"]
    hr = z["hr"].astype(np.float32)
    subjects = z["subject"].astype(str)
    bvp_fs = float(z["bvp_fs"])
    acc_fs = float(z["acc_fs"])

    expected_bvp = int(round(8 * bvp_fs))
    expected_acc = int(round(8 * acc_fs))

    if bvp.ndim != 2 or bvp.shape[1] != expected_bvp:
        raise ValueError(
            f"Unexpected BVP shape {bvp.shape}; expected (*,{expected_bvp})."
        )
    if acc.ndim != 3 or acc.shape[1:] != (expected_acc, 3):
        raise ValueError(
            f"Unexpected ACC shape {acc.shape}; expected (*,{expected_acc},3)."
        )
    if not (len(bvp) == len(acc) == len(hr) == len(subjects)):
        raise ValueError("BVP/ACC/HR/subject lengths differ.")
    if not (
        np.isfinite(bvp).all()
        and np.isfinite(acc).all()
        and np.isfinite(hr).all()
    ):
        raise ValueError("Input contains NaN/Inf.")

    original_windows = len(hr)
    original_subjects = sorted(set(subjects.tolist()))
    print(f"Dataset: {original_windows} windows / {len(original_subjects)} subjects")

    if args.max_windows > 0:
        idx = smoke_indices_by_subject(subjects, args.max_windows)
        bvp, acc, hr, subjects = bvp[idx], acc[idx], hr[idx], subjects[idx]
        print(
            f"Smoke subset: {len(hr)} windows / "
            f"{len(set(subjects.tolist()))} subjects"
        )
        counts = {
            sid: int(np.sum(subjects == sid))
            for sid in sorted(set(subjects.tolist()))
        }
        print(f"Smoke windows per subject: {counts}")
    else:
        print("Full benchmark mode: all windows retained.")

    train_s, val_s, test_s = subject_split(subjects)
    train_m = np.isin(subjects, train_s)
    val_m = np.isin(subjects, val_s)
    test_m = np.isin(subjects, test_s)

    if np.intersect1d(train_s, val_s).size or np.intersect1d(train_s, test_s).size or np.intersect1d(val_s, test_s).size:
        raise RuntimeError("Subject leakage detected in split.")

    print(
        f"Subject split: train={train_s.tolist()} "
        f"val={val_s.tolist()} test={test_s.tolist()}"
    )
    print(
        f"Window split: train={int(train_m.sum())} "
        f"val={int(val_m.sum())} test={int(test_m.sum())}"
    )

    if min(train_m.sum(), val_m.sum(), test_m.sum()) < 2:
        raise RuntimeError(
            "Smoke subset produced too few windows in a split. "
            "Increase --max-windows."
        )

    print("Building compact PPG/ACC features ...")
    X, names = build_features(bvp, acc, bvp_fs, acc_fs)
    y = hr

    Xtr, ytr = X[train_m], y[train_m]
    Xv, yv = X[val_m], y[val_m]
    Xt, yt = X[test_m], y[test_m]

    models = {
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "rf": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
    }

    val_scores = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(Xtr, ytr)
        pred = model.predict(Xv)
        val_scores[name] = metrics(yv, pred)
        print(
            f"  validation MAE={val_scores[name]['mae_bpm']:.3f} "
            f"RMSE={val_scores[name]['rmse_bpm']:.3f}"
        )

    selected = min(
        val_scores,
        key=lambda k: (
            float("inf") if val_scores[k]["mae_bpm"] is None
            else val_scores[k]["mae_bpm"]
        ),
    )

    final_model = models[selected]
    final_model.fit(np.concatenate([Xtr, Xv]), np.concatenate([ytr, yv]))
    test_pred = final_model.predict(Xt)

    test_bvp = bvp[test_m]
    fft_pred = np.array(
        [estimate_hr_bvp_only(x, bvp_fs)["fft_bpm"] for x in test_bvp],
        dtype=float,
    )
    peak_pred = np.array(
        [estimate_hr_bvp_only(x, bvp_fs)["peaks_bpm"] for x in test_bvp],
        dtype=float,
    )

    report = {
        "schema": {
            "original_windows": int(original_windows),
            "evaluated_windows": int(len(hr)),
            "original_subjects": original_subjects,
            "evaluated_subjects": sorted(set(subjects.tolist())),
            "bvp_shape": list(bvp.shape),
            "acc_shape": list(acc.shape),
            "bvp_fs_hz": bvp_fs,
            "acc_fs_hz": acc_fs,
            "window_sec": float(z["window_sec"]) if "window_sec" in z.files else 8.0,
            "shift_sec": float(z["shift_sec"]) if "shift_sec" in z.files else 2.0,
            "target": "ECG-derived HR label",
        },
        "smoke_sampling": {
            "enabled": bool(args.max_windows > 0),
            "requested_windows": int(args.max_windows),
            "method": "approximately equal windows per subject, evenly spaced within subject",
        },
        "split": {
            "method": "subject_level_deterministic_60_20_20",
            "train_subjects": train_s.tolist(),
            "validation_subjects": val_s.tolist(),
            "test_subjects": test_s.tolist(),
            "train_windows": int(train_m.sum()),
            "validation_windows": int(val_m.sum()),
            "test_windows": int(test_m.sum()),
            "subject_overlap": False,
        },
        "validation_model_selection": val_scores,
        "selected_model": selected,
        "test": {
            "bvp_fft": metrics(yt, fft_pred),
            "bvp_peaks": metrics(yt, peak_pred),
            "ml_selected": metrics(yt, test_pred),
        },
        "feature_names": names,
        "notes": [
            "PPG-DaLiA wrist optical data is Empatica E4 BVP, not MAX30102 raw optical data.",
            "This is a methodology benchmark, not a MAX30102 performance claim.",
            "Smoke-test metrics must not be used as research results.",
            "The final full benchmark uses all available windows with subject-level splitting.",
            "The held-out test subjects are never used for model selection.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nPPG-DaLiA HR benchmark complete")
    print(f"Selected model: {selected}")
    for k, v in report["test"].items():
        print(
            f"  {k:12s} MAE={v['mae_bpm']:.3f} "
            f"RMSE={v['rmse_bpm']:.3f} R2={v['r2']}"
        )
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
