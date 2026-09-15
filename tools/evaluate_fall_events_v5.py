"""
SafeBand AI — Event-level fall evaluation v5.

Key methodological improvement over v4:
  1. Load the v3 model/window artifact.
  2. Refit the model on TRAIN subjects only.
  3. Select the temporal event rule on VALIDATION subjects only.
  4. Refit the same model on TRAIN + VALIDATION subjects.
  5. Evaluate the locked rule once on TEST subjects.

The evaluator sweeps a small family of temporal rules:
  - hit threshold: probability required for a candidate window
  - minimum hit count inside a temporal cluster
  - peak threshold: strongest window in the cluster
  - maximum gap between hit windows

This avoids using the test set to choose the operating point.
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Make project-root imports work when called as: python tools/<script>.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_inputs(model_path: str, window_csv: str, training_report: str):
    payload = joblib.load(model_path)
    df = pd.read_csv(window_csv)
    report = json.loads(Path(training_report).read_text(encoding="utf-8"))

    feature_columns = payload["feature_columns"]
    model = payload["model"]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Window CSV is missing model features: {missing}")

    train_subjects = {str(x) for x in report["subjects"]["train"]}
    val_subjects = {str(x) for x in report["subjects"]["validation"]}
    test_subjects = {str(x) for x in report["subjects"]["test"]}

    subjects = df["subject_id"].astype(str)
    split = {
        "train": df[subjects.isin(train_subjects)].copy(),
        "validation": df[subjects.isin(val_subjects)].copy(),
        "test": df[subjects.isin(test_subjects)].copy(),
    }
    if any(len(x) == 0 for x in split.values()):
        raise ValueError("One or more subject splits contain zero windows.")

    return payload, split


def fall_probabilities(model, df, feature_columns):
    p = model.predict_proba(df[feature_columns])
    classes = [str(c).upper() for c in model.classes_]
    if "FALL" not in classes:
        raise ValueError(f"Model classes do not contain FALL: {model.classes_}")
    return p[:, classes.index("FALL")]


def aggregate_recording(g: pd.DataFrame, hit_threshold: float,
                        min_hits: int, peak_threshold: float,
                        max_gap: int):
    g = g.sort_values("start_sample")
    starts = g["start_sample"].to_numpy(dtype=int)
    probs = g["fall_probability"].to_numpy(dtype=float)
    hit_idx = np.flatnonzero(probs >= hit_threshold)

    clusters = []
    current = []
    for idx in hit_idx:
        if not current or starts[idx] - starts[current[-1]] <= max_gap:
            current.append(int(idx))
        else:
            clusters.append(current)
            current = [int(idx)]
    if current:
        clusters.append(current)

    confirmed = []
    for cluster in clusters:
        peak = float(probs[cluster].max())
        if len(cluster) >= min_hits and peak >= peak_threshold:
            confirmed.append(cluster)

    peak_probability = max(
        (float(probs[c].max()) for c in confirmed), default=0.0
    )
    return bool(confirmed), peak_probability, len(confirmed)


def evaluate(df: pd.DataFrame, hit_threshold: float, min_hits: int,
             peak_threshold: float, max_gap: int):
    rows = []
    for key, g in df.groupby(["subject_id", "source_file"], sort=False):
        truth = "FALL" if str(g["recording_type"].iloc[0]).lower() == "fall" else "NON_FALL"
        pred, peak, clusters = aggregate_recording(
            g, hit_threshold, min_hits, peak_threshold, max_gap
        )
        rows.append({
            "subject_id": str(key[0]),
            "source_file": key[1],
            "truth": truth,
            "prediction": "FALL" if pred else "NON_FALL",
            "peak_probability": peak,
            "confirmed_clusters": clusters,
        })

    r = pd.DataFrame(rows)
    y, p = r["truth"], r["prediction"]
    tn, fp, fn, tp = confusion_matrix(
        y, p, labels=["NON_FALL", "FALL"]
    ).ravel()
    return {
        "recordings": int(len(r)),
        "fall_events": int((y == "FALL").sum()),
        "nonfall_recordings": int((y == "NON_FALL").sum()),
        "fall_recall": float(recall_score(y, p, pos_label="FALL", zero_division=0)),
        "fall_precision": float(precision_score(y, p, pos_label="FALL", zero_division=0)),
        "fall_f1": float(f1_score(y, p, pos_label="FALL", zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_alarms": int(fp),
        "missed_falls": int(fn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }, r


def parse_floats(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_ints(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def select_rule(results: pd.DataFrame, target_recall: float,
                target_fpr: float):
    # Prefer rules satisfying both engineering targets. If none exist,
    # prioritize recall >= target and then minimize FPR; this is transparent
    # and keeps safety recall ahead of cosmetic accuracy improvements.
    both = results[
        (results.fall_recall >= target_recall) &
        (results.false_positive_rate <= target_fpr)
    ]
    recall_eligible = results[results.fall_recall >= target_recall]

    if len(both):
        pool = both
        reason = "meets_recall_and_fpr_targets"
        ordered = pool.sort_values(
            ["false_positive_rate", "fall_f1", "fall_recall"],
            ascending=[True, False, False]
        )
    elif len(recall_eligible):
        pool = recall_eligible
        reason = "no_rule_meets_both_targets; recall_target_met"
        ordered = pool.sort_values(
            ["false_positive_rate", "fall_f1", "fall_recall"],
            ascending=[True, False, False]
        )
    else:
        pool = results
        reason = "no_rule_meets_recall_target; best_f1_fallback"
        ordered = pool.sort_values(
            ["fall_f1", "fall_recall", "false_positive_rate"],
            ascending=[False, False, True]
        )
    return ordered.iloc[0].to_dict(), reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--training-report", required=True,
                    help="v3 training report containing subject train/validation/test lists")
    ap.add_argument("--out", default="models/fall_event_report_v5.json")
    ap.add_argument("--hit-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--peak-thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--min-hits", default="2,3,4")
    ap.add_argument("--max-gaps", default="30,60")
    ap.add_argument("--target-recall", type=float, default=0.95)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    args = ap.parse_args()

    payload, split = load_inputs(args.model, args.input, args.training_report)
    feature_columns = payload["feature_columns"]
    base_model = payload["model"]

    # Validation probabilities must come from a model that did not train on
    # validation subjects. This is the key methodological difference from v4.
    validation_model = clone(base_model)
    validation_model.fit(split["train"][feature_columns], split["train"]["label"])
    split["validation"]["fall_probability"] = fall_probabilities(
        validation_model, split["validation"], feature_columns
    )

    hit_thresholds = parse_floats(args.hit_thresholds)
    peak_thresholds = parse_floats(args.peak_thresholds)
    min_hits_values = parse_ints(args.min_hits)
    max_gap_values = parse_ints(args.max_gaps)

    validation_results = []
    for ht in hit_thresholds:
        for pt in peak_thresholds:
            for mh in min_hits_values:
                for gap in max_gap_values:
                    metrics, _ = evaluate(
                        split["validation"], ht, mh, pt, gap
                    )
                    validation_results.append({
                        "hit_threshold": ht,
                        "peak_threshold": pt,
                        "min_hits": mh,
                        "max_gap_samples": gap,
                        **metrics,
                    })

    val_df = pd.DataFrame(validation_results)
    selected, selection_reason = select_rule(
        val_df, args.target_recall, args.target_fpr
    )

    # Lock the rule, then fit the model on train + validation only.
    final_model = clone(base_model)
    trainval = pd.concat([split["train"], split["validation"]], ignore_index=True)
    final_model.fit(trainval[feature_columns], trainval["label"])
    split["test"]["fall_probability"] = fall_probabilities(
        final_model, split["test"], feature_columns
    )

    ht = float(selected["hit_threshold"])
    pt = float(selected["peak_threshold"])
    mh = int(selected["min_hits"])
    gap = int(selected["max_gap_samples"])
    test_metrics, _ = evaluate(split["test"], ht, mh, pt, gap)

    report = {
        "version": "v5",
        "method": "validation_locked_temporal_event_rule",
        "model": args.model,
        "window_input": args.input,
        "training_report": args.training_report,
        "feature_count": len(feature_columns),
        "subjects": {
            "train": sorted(split["train"].subject_id.astype(str).unique(), key=int),
            "validation": sorted(split["validation"].subject_id.astype(str).unique(), key=int),
            "test": sorted(split["test"].subject_id.astype(str).unique(), key=int),
        },
        "search_space": {
            "hit_thresholds": hit_thresholds,
            "peak_thresholds": peak_thresholds,
            "min_hits": min_hits_values,
            "max_gaps": max_gap_values,
        },
        "targets": {
            "fall_recall": args.target_recall,
            "false_positive_rate": args.target_fpr,
        },
        "selection_reason": selection_reason,
        "selected_operating_point": selected,
        "test": test_metrics,
        "validation_operating_points": validation_results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection_reason": selection_reason,
        "selected_operating_point": selected,
        "test": test_metrics,
        "report": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
