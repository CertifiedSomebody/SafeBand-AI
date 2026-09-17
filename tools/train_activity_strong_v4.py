"""SafeBand Activity Strong V4.

Strengthen the deployment-relevant BITS2 activity model without changing the
sensor contract. The task is deliberately restricted to ordinary activities:
RESTING, SITTING, WALKING, RUNNING. FALL remains handled by the separate fall
pipeline.

Methodology
-----------
* Subject-independent outer StratifiedGroupKFold evaluation.
* Inner grouped CV selects the model/hyperparameter configuration.
* Selection target is macro-F1, with balanced accuracy as the tie-breaker.
* No outer-validation labels are used for model selection.
* A final deployment model is selected by grouped CV over the full activity
  dataset and then fitted on all available activity windows.

The saved joblib payload is compatible with SafeBand's MLActivityModel loader.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES = [
    "ax_mean", "ax_std", "ax_min", "ax_max", "ax_rms",
    "ay_mean", "ay_std", "ay_min", "ay_max", "ay_rms",
    "az_mean", "az_std", "az_min", "az_max", "az_rms",
    "acc_mag_mean", "acc_mag_std", "acc_mag_min", "acc_mag_max", "acc_mag_rms",
    "acc_mag_p10", "acc_mag_p25", "acc_mag_p50", "acc_mag_p75", "acc_mag_p90",
    "acc_sma", "acc_range", "acc_jerk_mean", "acc_jerk_std", "acc_jerk_max",
    "acc_diff_energy", "acc_zero_crossings",
    "acc_x_y_corr", "acc_x_z_corr", "acc_y_z_corr",
    "acc_fft_low", "acc_fft_mid", "acc_fft_high",
]
ACTIVITY_CLASSES = ["RESTING", "SITTING", "WALKING", "RUNNING"]
META = {"activity_label", "subject_id", "session_id", "source_file", "recording_type",
        "window_start_sample", "window_end_sample", "window_samples"}


def metric_dict(y, pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def weight_map(y: pd.Series, mode: str) -> Dict[str, float] | None:
    if mode == "none":
        return None
    counts = y.value_counts().to_dict()
    n = len(y)
    k = len(counts)
    if mode == "balanced":
        return {c: n / (k * counts[c]) for c in counts}
    if mode == "sqrt_balanced":
        return {c: float(np.sqrt(n / (k * counts[c]))) for c in counts}
    raise ValueError(mode)


def sample_weights(y: pd.Series, mode: str) -> np.ndarray | None:
    wm = weight_map(y, mode)
    if wm is None:
        return None
    return y.map(wm).to_numpy(dtype=float)


def build_candidates(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}
    for weight in ("none", "sqrt_balanced", "balanced"):
        cw = weight_map(pd.Series(ACTIVITY_CLASSES), weight)
        for leaf in (1, 2, 3):
            models[f"extra_trees__{weight}__leaf{leaf}"] = ExtraTreesClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=leaf,
                class_weight=cw,
                random_state=seed,
                n_jobs=-1,
            )
        models[f"random_forest__{weight}"] = RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight=cw,
            random_state=seed,
            n_jobs=-1,
        )
    # RBF-SVM is a useful nonlinear comparator for this feature space. Scaling
    # is inside the pipeline, so there is no leakage from the validation fold.
    for c in (1.0, 3.0, 10.0):
        models[f"rbf_svm__balanced__C{c:g}"] = Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed)),
        ])
    return models


def fit_predict(model, Xtr, ytr, Xte):
    # Tree models receive no internal class_weight object when the candidate is
    # 'none'; weighted candidates already carry their class weights.
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def grouped_cv_score(model, X, y, groups, splits) -> Tuple[float, float]:
    vals = []
    for tr, va in splits:
        pred = fit_predict(model, X.iloc[tr], y.iloc[tr], X.iloc[va])
        m = metric_dict(y.iloc[va], pred)
        vals.append((m["macro_f1"], m["balanced_accuracy"]))
    return float(np.mean([x[0] for x in vals])), float(np.mean([x[1] for x in vals]))


def select_inner(X, y, groups, seed: int):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    splits = list(splitter.split(X, y, groups))
    results = {}
    for name, model in build_candidates(seed).items():
        f1, bal = grouped_cv_score(model, X, y, groups, splits)
        results[name] = {"macro_f1": f1, "balanced_accuracy": bal}
    best_name = max(results, key=lambda n: (results[n]["macro_f1"], results[n]["balanced_accuracy"]))
    return best_name, results


def load_data(path: Path):
    df = pd.read_csv(path)
    required = FEATURES + ["activity_label", "subject_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[df["activity_label"].isin(ACTIVITY_CLASSES)].copy()
    before = len(df)
    df = df.dropna(subset=FEATURES + ["subject_id", "activity_label"]).copy()
    df["subject_id"] = df["subject_id"].astype(str)
    if df["subject_id"].nunique() < 5:
        raise ValueError("At least 5 subjects are required for grouped evaluation.")
    return df, before - len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/processed/bits2/activity_windows.csv")
    ap.add_argument("--out", default="models/activity_strong_v4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outer-folds", type=int, default=5)
    args = ap.parse_args()

    data_path = Path(args.input)
    df, dropped = load_data(data_path)
    X = df[FEATURES]
    y = df["activity_label"]
    groups = df["subject_id"]

    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    all_true, all_base = [], []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        gtr = groups.iloc[tr]
        best_name, inner_results = select_inner(Xtr, ytr, gtr, args.seed + fold)
        model = build_candidates(args.seed + fold)[best_name]
        pred = fit_predict(model, Xtr, ytr, Xte)
        m = metric_dict(yte, pred)
        fold_rows.append({"fold": fold, "selected_model": best_name, **m,
                          "test_subjects": sorted(groups.iloc[te].unique().tolist()),
                          "train_subject_count": int(gtr.nunique()),
                          "test_subject_count": int(groups.iloc[te].nunique()),
                          "inner_candidates": inner_results})
        all_true.extend(yte.tolist())
        all_base.extend(pred.tolist())
        print(f"fold {fold}: {best_name} | macro_f1={m['macro_f1']:.4f} | bal_acc={m['balanced_accuracy']:.4f} | acc={m['accuracy']:.4f}")

    outer_summary = {
        metric: {"mean": float(np.mean([r[metric] for r in fold_rows])),
                 "std": float(np.std([r[metric] for r in fold_rows], ddof=1))}
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }

    # Deployment selection uses only grouped CV on the complete activity set.
    deployment_name, deployment_candidates = select_inner(X, y, groups, args.seed + 100)
    deployment_model = build_candidates(args.seed + 100)[deployment_name]
    deployment_model.fit(X, y)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "activity_model.joblib"
    payload = {
        "model": deployment_model,
        "feature_columns": FEATURES,
        "model_name": "SafeBand BITS-2 Strong Activity Model",
        "model_version": "bits2_activity_strong_v4",
        "label_classes": ACTIVITY_CLASSES,
        "window_samples": int(df["window_samples"].iloc[0]) if "window_samples" in df else 80,
        "assumed_sample_rate_hz": 20,
        "task": "activity",
        "selection_metric": "grouped macro_f1, then balanced_accuracy",
        "deployment_selection": deployment_name,
    }
    joblib.dump(payload, model_path)

    labels = ACTIVITY_CLASSES
    report = {
        "experiment": "bits2_activity_strong_v4",
        "input": str(data_path),
        "task": "RESTING/SITTING/WALKING/RUNNING only; FALL remains separate",
        "method": "nested subject-grouped model selection",
        "outer_folds": args.outer_folds,
        "subject_count": int(groups.nunique()),
        "row_count": int(len(df)),
        "dropped_rows": int(dropped),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "outer_summary": outer_summary,
        "outer_fold_results": fold_rows,
        "pooled_outer_classification_report": classification_report(all_true, all_base, labels=labels, output_dict=True, zero_division=0),
        "pooled_outer_confusion_matrix": confusion_matrix(all_true, all_base, labels=labels).tolist(),
        "deployment_selection": deployment_name,
        "deployment_inner_candidates": deployment_candidates,
        "model_path": str(model_path),
        "reproducibility": {"seed": args.seed, "outer_folds": args.outer_folds},
    }
    report_path = out / "activity_strong_v4_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nOUTER SUMMARY")
    for k, v in outer_summary.items():
        print(f"{k}: {v['mean']:.6f} +/- {v['std']:.6f}")
    print(f"\nDeployment model: {deployment_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
