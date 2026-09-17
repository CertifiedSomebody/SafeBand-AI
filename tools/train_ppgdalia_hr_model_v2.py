from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ai.ppg_enhanced_features import extract_features


def split_subjects(subjects):
    ids = np.array(sorted(set(subjects.tolist())))
    if len(ids) < 3:
        raise ValueError("Need >=3 subjects.")
    n = len(ids)
    nt = min(max(round(n*0.60),1), n-2)
    nv = min(max(round(n*0.20),1), n-nt-1)
    return ids[:nt], ids[nt:nt+nv], ids[nt+nv:]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    e = np.abs(y-p)
    return {
        "n": int(len(y)),
        "mae_bpm": float(mean_absolute_error(y,p)),
        "rmse_bpm": float(np.sqrt(mean_squared_error(y,p))),
        "r2": float(r2_score(y,p)),
        "within_3_bpm_pct": float(np.mean(e<=3)*100),
        "within_5_bpm_pct": float(np.mean(e<=5)*100),
        "within_10_bpm_pct": float(np.mean(e<=10)*100),
        "bias_bpm": float(np.mean(p-y)),
    }


def build_matrix(bvp, acc, bfs, afs):
    first = extract_features(bvp[0], acc[0], bfs, afs)
    names = sorted(first)
    X = np.empty((len(bvp), len(names)), dtype=np.float32)
    for i in range(len(bvp)):
        f = extract_features(bvp[i], acc[i], bfs, afs)
        X[i] = [f[k] for k in names]
    # Feature NaNs are allowed only because a peak estimator can legitimately
    # fail on a low-quality window; median imputation is fitted on training only.
    return X, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--report-out", required=True)
    args = ap.parse_args()

    z = np.load(args.input, allow_pickle=True)
    req = {"bvp","acc","hr","subject","bvp_fs","acc_fs"}
    miss = req - set(z.files)
    if miss: raise ValueError(f"Missing arrays: {sorted(miss)}")

    bvp, acc = z["bvp"], z["acc"]
    y, subjects = z["hr"].astype(np.float32), z["subject"].astype(str)
    bfs, afs = float(z["bvp_fs"]), float(z["acc_fs"])

    if bvp.ndim != 2 or bvp.shape[1] != int(round(8*bfs)):
        raise ValueError(f"BVP shape {bvp.shape} incompatible with 8s @ {bfs}Hz")
    if acc.ndim != 3 or acc.shape[1:] != (int(round(8*afs)),3):
        raise ValueError(f"ACC shape {acc.shape} incompatible with 8s @ {afs}Hz")
    if not (len(bvp)==len(acc)==len(y)==len(subjects)):
        raise ValueError("Window arrays have different lengths.")
    if not (np.isfinite(bvp).all() and np.isfinite(acc).all() and np.isfinite(y).all()):
        raise ValueError("Raw dataset contains NaN/Inf.")

    tr_s, va_s, te_s = split_subjects(subjects)
    tr = np.isin(subjects,tr_s); va=np.isin(subjects,va_s); te=np.isin(subjects,te_s)
    if set(tr_s)&set(va_s) or set(tr_s)&set(te_s) or set(va_s)&set(te_s):
        raise RuntimeError("Subject leakage detected.")

    print(f"Dataset: {len(y)} windows / {len(set(subjects))} subjects")
    print(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")
    print("Extracting enhanced PPG + ACC features ...")
    X, names = build_matrix(bvp,acc,bfs,afs)

    # All models use training-only median imputation. HistGradientBoosting is
    # included because it can model nonlinear interactions without requiring
    # extremely large forests.
    models = {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "rf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=600, min_samples_leaf=2,
                max_features=0.75, random_state=42, n_jobs=-1
            )),
        ]),
        "extra_trees": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=600, min_samples_leaf=2,
                max_features=0.8, random_state=42, n_jobs=-1
            )),
        ]),
        "hgb": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42
            )),
        ]),
    }

    val = {}
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X[tr], y[tr])
        val[name] = metrics(y[va], model.predict(X[va]))
        print(f"  val MAE={val[name]['mae_bpm']:.3f}")

    selected = min(val, key=lambda k: val[k]["mae_bpm"])
    print(f"Selected by validation MAE: {selected}")

    final = models[selected]
    final.fit(X[tr|va], y[tr|va])
    pred = final.predict(X[te])
    test = metrics(y[te],pred)

    artifact = {
        "model": final,
        "feature_names": names,
        "bvp_fs_hz": bfs,
        "acc_fs_hz": afs,
        "window_sec": 8.0,
        "shift_sec": 2.0,
        "source_sensor": "Empatica E4 wrist BVP + wrist ACC",
        "target": "ECG-derived HR BPM",
        "model_type": selected,
        "training_subjects": tr_s.tolist(),
        "validation_subjects": va_s.tolist(),
        "test_subjects": te_s.tolist(),
        "domain_warning": "Reference model only; not MAX30102 validated.",
    }

    mo=Path(args.model_out); mo.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(artifact,mo)

    report = {
        "dataset": {"windows":len(y),"subjects":len(set(subjects)),
                    "bvp_shape":list(bvp.shape),"acc_shape":list(acc.shape),
                    "bvp_fs_hz":bfs,"acc_fs_hz":afs},
        "split": {"method":"deterministic_subject_level_60_20_20",
                  "train_subjects":tr_s.tolist(),"validation_subjects":va_s.tolist(),
                  "test_subjects":te_s.tolist(),"train_windows":int(tr.sum()),
                  "validation_windows":int(va.sum()),"test_windows":int(te.sum()),
                  "subject_overlap":False},
        "validation":val,
        "selected_model":selected,
        "held_out_test":test,
        "model_artifact":str(mo),
        "feature_names":names,
        "notes":[
            "Enhanced representation uses normalized PPG morphology, zero-padded spectral localization, autocorrelation, peak/IBI quality, and wrist-ACC motion features.",
            "Feature imputation is fit inside the training pipeline only.",
            "Final model is refit on train+validation after selection.",
            "Test subjects are never used for model selection.",
            "PPG-DaLiA is Empatica E4 BVP; this is not a MAX30102 deployment claim.",
        ],
    }
    ro=Path(args.report_out); ro.parent.mkdir(parents=True,exist_ok=True)
    ro.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("\nENHANCED HR MODEL COMPLETE")
    print(f"Test MAE={test['mae_bpm']:.3f} BPM | RMSE={test['rmse_bpm']:.3f} | R2={test['r2']:.3f}")
    print(f"Within ±5 BPM={test['within_5_bpm_pct']:.2f}% | ±10 BPM={test['within_10_bpm_pct']:.2f}%")
    print(f"Model: {mo}")
    print(f"Report: {ro}")


if __name__ == "__main__":
    main()
