"""Train the first SafeBand motion models from BITS-2 windows.

Two models are trained deliberately:
1) activity_model.joblib: WALKING / RUNNING / SITTING / RESTING
2) fall_detector.joblib: FALL / NON_FALL

Keeping FALL separate prevents the many overlapping fall windows in BITS-2
from dominating ordinary activity classification. Splits are by SUBJECT, not
by window, so the reported metrics are leakage-resistant.
"""
from __future__ import annotations

import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from ai.feature_extraction import FEATURE_COLUMNS


def split_by_subject(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/validation/test with disjoint subjects."""
    labels = set(df["target"].unique())
    for attempt in range(100):
        rs = seed + attempt
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=rs)
        trv_i, te_i = next(g1.split(df, groups=df["subject_id"]))
        trv = df.iloc[trv_i]
        test = df.iloc[te_i]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=rs + 1000)
        tr_i, va_i = next(g2.split(trv, groups=trv["subject_id"]))
        train = trv.iloc[tr_i]
        val = trv.iloc[va_i]
        if all(set(part["target"].unique()) == labels for part in (train, val, test)):
            return train, val, test
    raise RuntimeError("Unable to find a subject-independent split containing every class.")


def metrics(y, pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def candidate_models(seed: int):
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed,
            n_jobs=-1, min_samples_leaf=2, max_features="sqrt"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed,
            n_jobs=-1, min_samples_leaf=2, max_features="sqrt"
        ),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ]),
    }


def train_task(df: pd.DataFrame, seed: int, fall_task: bool):
    train, val, test = split_by_subject(df, seed)
    results = {}
    models = candidate_models(seed)
    for name, model in models.items():
        model.fit(train[FEATURE_COLUMNS], train["target"])
        pred = model.predict(val[FEATURE_COLUMNS])
        m = metrics(val["target"], pred)
        if fall_task:
            m["fall_recall"] = float(recall_score(val["target"], pred, pos_label="FALL", zero_division=0))
        results[name] = m

    if fall_task:
        best_name = max(results, key=lambda n: (results[n]["fall_recall"], results[n]["macro_f1"], results[n]["balanced_accuracy"]))
    else:
        best_name = max(results, key=lambda n: (results[n]["macro_f1"], results[n]["balanced_accuracy"]))

    best = models[best_name]
    trainval = pd.concat([train, val], ignore_index=True)
    best.fit(trainval[FEATURE_COLUMNS], trainval["target"])
    pred = best.predict(test[FEATURE_COLUMNS])
    test_metrics = metrics(test["target"], pred)
    if fall_task:
        test_metrics["fall_recall"] = float(recall_score(test["target"], pred, pos_label="FALL", zero_division=0))
    labels = sorted(df["target"].unique())
    report = classification_report(test["target"], pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(test["target"], pred, labels=labels).tolist()
    split_info = {
        "train_subjects": sorted(train.subject_id.unique().tolist()),
        "validation_subjects": sorted(val.subject_id.unique().tolist()),
        "test_subjects": sorted(test.subject_id.unique().tolist()),
    }
    return best, best_name, results, test_metrics, report, cm, split_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="activity_windows.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report", default="models/bits2_training_report.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLUMNS + ["subject_id", "activity_label"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    df = df.dropna(subset=FEATURE_COLUMNS + ["subject_id", "activity_label"]).copy()
    df["subject_id"] = df["subject_id"].astype(str)

    activity = df[df["activity_label"].isin(["WALKING", "RUNNING", "SITTING", "RESTING"])].copy()
    activity["target"] = activity["activity_label"]
    fall = df.copy()
    fall["target"] = np.where(fall["activity_label"].eq("FALL"), "FALL", "NON_FALL")

    out = Path(args.model_dir); out.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": str(Path(args.input)),
        "feature_columns": FEATURE_COLUMNS,
        "window_samples": int(df["window_samples"].iloc[0]) if "window_samples" in df else 40,
        "assumed_sample_rate_hz": 20,
        "subject_count": int(df.subject_id.nunique()),
        "activity_window_counts": activity.activity_label.value_counts().sort_index().to_dict(),
        "fall_window_counts": fall.target.value_counts().sort_index().to_dict(),
    }

    for name, task_df, fall_task, filename in [
        ("activity", activity, False, "activity_model.joblib"),
        ("fall_detector", fall, True, "fall_detector.joblib"),
    ]:
        best, best_name, validation, test_metrics, class_report, cm, split_info = train_task(task_df, args.seed, fall_task)
        payload = {
            "model": best,
            "feature_columns": FEATURE_COLUMNS,
            "model_name": "SafeBand BITS-2 Activity Model" if name == "activity" else "SafeBand BITS-2 Fall Detector",
            "model_version": "bits2_motion_v1",
            "label_classes": sorted(task_df.target.unique()),
            "window_samples": int(df.window_samples.iloc[0]) if "window_samples" in df else 40,
            "assumed_sample_rate_hz": 20,
            "task": name,
            "selection_metric": "fall_recall, then macro_f1, then balanced_accuracy" if fall_task else "macro_f1, then balanced_accuracy",
        }
        joblib.dump(payload, out / filename)
        report[name] = {
            "selected_model": best_name,
            "validation": validation,
            "test": test_metrics,
            "classification_report": class_report,
            "confusion_matrix": cm,
            "split": split_info,
        }
        print(f"\n{name.upper()} -> {best_name}")
        print("validation:", validation[best_name])
        print("test:", test_metrics)
        print(classification_report(task_df.loc[task_df.subject_id.isin(split_info["test_subjects"]), "target"],
                                    best.predict(task_df.loc[task_df.subject_id.isin(split_info["test_subjects"]), FEATURE_COLUMNS]),
                                    zero_division=0))

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved models to: {out}")
    print(f"Saved report to: {args.report}")

if __name__ == "__main__":
    main()
