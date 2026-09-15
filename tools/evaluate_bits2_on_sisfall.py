"""Cross-dataset evaluation: BITS-2 V6 fall model -> SisFall.

This script performs NO SisFall training or threshold tuning.

Important fixes:
1. The FALL probability column is selected by class name. Do not assume
   predict_proba()[:, 1] is FALL; sklearn alphabetically orders classes, so
   ['FALL', 'NON_FALL'] makes column 0 the FALL probability.
2. Recording-level max probability is descriptive only.
"""

from __future__ import annotations
import argparse, json, os, time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

def fall_probability(model, X):
    classes = [str(c).upper() for c in model.classes_]
    try:
        idx = classes.index("FALL")
    except ValueError as exc:
        raise RuntimeError(
            f"Model classes do not contain FALL: {model.classes_!r}"
        ) from exc
    return model.predict_proba(X)[:, idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(args.input)
    payload = joblib.load(args.model)
    model = payload["model"]
    cols = list(payload["feature_columns"])

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing model features ({len(missing)}): {missing}"
        )

    threshold = float(
        args.threshold
        if args.threshold is not None
        else payload.get("fall_threshold", 0.5)
    )

    X = df[cols].astype("float32")
    y = (
        df["recording_type"].astype(str).str.upper().eq("FALL")
        .astype(int)
        .to_numpy()
    )

    start = time.time()
    p = fall_probability(model, X)
    elapsed = time.time() - start
    pred = (p >= threshold).astype(int)

    result = {
        "model": args.model,
        "input": args.input,
        "feature_count": len(cols),
        "model_classes": [str(c) for c in model.classes_],
        "fall_probability_class_index": [
            str(c).upper() for c in model.classes_
        ].index("FALL"),
        "threshold": threshold,
        "windows": int(len(df)),
        "recordings": int(df["source_file"].nunique()),
        "inference_seconds": elapsed,
        "window_metrics": {},
        "recording_metrics": {},
    }

    result["window_metrics"] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

    rec = df[
        ["source_file", "subject_id", "activity_code", "recording_type"]
    ].copy()
    rec["p_fall"] = p

    grouped = rec.groupby("source_file", sort=False)
    ry = grouped["recording_type"].first().eq("FALL").astype(int).to_numpy()
    rp = grouped["p_fall"].max().to_numpy()
    rpred = (rp >= threshold).astype(int)

    result["recording_metrics"] = {
        "accuracy": float(accuracy_score(ry, rpred)),
        "balanced_accuracy": float(balanced_accuracy_score(ry, rpred)),
        "precision": float(precision_score(ry, rpred, zero_division=0)),
        "recall": float(recall_score(ry, rpred, zero_division=0)),
        "f1": float(f1_score(ry, rpred, zero_division=0)),
        "roc_auc": float(roc_auc_score(ry, rp)),
        "confusion_matrix": confusion_matrix(ry, rpred).tolist(),
    }

    result["by_subject"] = {}
    for subject, sub in rec.groupby("subject_id"):
        sg = sub.groupby("source_file").agg(
            y=("recording_type", lambda x: int(x.iloc[0] == "FALL")),
            p=("p_fall", "max"),
        )
        auc = (
            None
            if sg.y.nunique() < 2
            else float(roc_auc_score(sg.y, sg.p))
        )
        result["by_subject"][str(subject)] = {
            "recordings": int(len(sg)),
            "falls": int(sg.y.sum()),
            "roc_auc": auc,
        }

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
