from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.fall_event_confirmator_v7 import V7_FEATURES, extract_event_features

SEED = 42
TRAIN_SUBJECTS = ["1","2","3","4","7","9","10","12","13","14","15","16","18","20","23","24","27","30","31","32","34","35","37","38","40"]
VAL_SUBJECTS = ["11","21","22","25","26","28","29","36"]
TEST_SUBJECTS = ["5","6","8","17","19","33","39","41"]


def load_v6(model_path: str):
    artifact = joblib.load(model_path)
    if "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError("V6 artifact must contain model and feature_columns")
    return artifact


def fall_probability(model, X: np.ndarray) -> np.ndarray:
    classes = [str(c).upper() for c in model.classes_]
    return model.predict_proba(X)[:, classes.index("FALL")]


def recordings(df: pd.DataFrame, p: np.ndarray) -> list[dict]:
    cols = ["subject_id", "source_file", "recording_type", "label", "start_sample"]
    d = df[cols].copy()
    d["p"] = p.astype(np.float32)
    d["subject_id"] = d["subject_id"].astype(str)
    d = d.sort_values(["subject_id", "source_file", "recording_type", "start_sample"], kind="stable")
    out = []
    for key, g in d.groupby(["subject_id", "source_file", "recording_type"], sort=False):
        out.append({
            "subject_id": str(key[0]), "source_file": str(key[1]), "recording_type": str(key[2]),
            "label": str(g["label"].iloc[0]).upper(),
            "starts": g["start_sample"].to_numpy(np.int32, copy=True),
            "p": g["p"].to_numpy(np.float32, copy=True),
        })
    return out


def make_event_frame(recs: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    y = []
    for r in recs:
        row = extract_event_features(r["p"], r["starts"])
        row.update(subject_id=r["subject_id"], source_file=r["source_file"], recording_type=r["recording_type"])
        rows.append(row)
        y.append(1 if r["label"] == "FALL" else 0)
    return pd.DataFrame(rows), np.asarray(y, dtype=int)


def subject_sets(df: pd.DataFrame):
    s = df["subject_id"].astype(str)
    return {
        "train": df[s.isin(TRAIN_SUBJECTS)].copy(),
        "validation": df[s.isin(VAL_SUBJECTS)].copy(),
        "test": df[s.isin(TEST_SUBJECTS)].copy(),
    }


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "fall_recall": float(recall_score(y, pred, zero_division=0)),
        "fall_precision": float(precision_score(y, pred, zero_division=0)),
        "fall_f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_alarms": int(fp), "missed_falls": int(fn),
        "confusion_matrix": [[int(tp), int(fn)], [int(fp), int(tn)]],
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def train_event_models(Xtr, ytr):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "extra_trees": ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_leaf_nodes=7, l2_regularization=1.0, random_state=SEED),
    }


def threshold_search(y, score):
    # Validation-only threshold selection. Safety target has priority.
    rows = []
    for t in np.arange(0.20, 0.901, 0.01):
        pred = (score >= t).astype(int)
        m = metrics(y, pred, score)
        m["threshold"] = round(float(t), 2)
        rows.append(m)
    both = [r for r in rows if r["fall_recall"] >= 0.95 and r["false_positive_rate"] <= 0.05]
    if both:
        pool = both; reason = "validation_event_model_meets_both_targets"
    else:
        recall_ok = [r for r in rows if r["fall_recall"] >= 0.95]
        if recall_ok:
            pool = recall_ok; reason = "no_threshold_meets_both_targets; recall_target_met"
        else:
            pool = rows; reason = "no_threshold_meets_recall_target"
    best = max(pool, key=lambda r: (r["fall_recall"], -r["false_positive_rate"], r["fall_f1"], r["fall_precision"]))
    return best, reason, rows


def main():
    ap = argparse.ArgumentParser(description="SafeBand BITS-2 V7 event-level fall confirmation")
    ap.add_argument("--input", required=True)
    ap.add_argument("--v6-model", required=True)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report", default="models/bits2_fall_training_report_v7.json")
    args = ap.parse_args()
    started = time.perf_counter()

    print("[V7] Loading windows and V6 artifact...", flush=True)
    df = pd.read_csv(args.input, low_memory=False)
    artifact = load_v6(args.v6_model)
    base_model = artifact["model"]
    features = artifact["feature_columns"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing V6 feature columns: {missing}")

    sets = subject_sets(df)
    print(f"[V7] Windows: total={len(df):,}, train={len(sets['train']):,}, validation={len(sets['validation']):,}, test={len(sets['test']):,}", flush=True)

    # V6 final artifact was fit on TRAIN+VALIDATION. That artifact is intentionally
    # not used to generate validation data for event-model selection. Instead we
    # retrain the V6 base classifier on TRAIN only, then generate validation/test
    # probabilities from that TRAIN-only model. This prevents validation leakage.
    from sklearn.base import clone
    train_model = clone(base_model)
    Xtr = sets["train"][features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    ytr = sets["train"]["label"].astype(str).str.upper().to_numpy()
    print("[V7] Fitting TRAIN-only base classifier...", flush=True)
    train_model.fit(Xtr, ytr)

    recs = {}
    for name in ("train", "validation", "test"):
        X = sets[name][features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        p = fall_probability(train_model, X)
        recs[name] = recordings(sets[name], p)
        print(f"[V7] {name}: cached {len(recs[name])} recordings", flush=True)

    # IMPORTANT: training event features use predictions from the TRAIN-only base
    # model. To reduce optimistic bias, event-model training uses grouped 5-fold
    # OOF base probabilities over training subjects.
    from sklearn.model_selection import GroupKFold
    groups = sets["train"]["subject_id"].astype(str).to_numpy()
    oof = np.zeros(len(sets["train"]), dtype=np.float32)
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    print(f"[V7] Generating {n_splits}-fold subject-grouped OOF base probabilities...", flush=True)
    for fold, (a, b) in enumerate(gkf.split(Xtr, ytr, groups), 1):
        m = clone(base_model)
        m.fit(Xtr[a], ytr[a])
        oof[b] = fall_probability(m, Xtr[b])
        print(f"  OOF fold {fold}/{n_splits} complete", flush=True)
    oof_recs = recordings(sets["train"], oof)

    X_event_train_df, y_event_train = make_event_frame(oof_recs)
    X_event_val_df, y_event_val = make_event_frame(recs["validation"])
    X_event_test_df, y_event_test = make_event_frame(recs["test"])
    X_event_train = X_event_train_df[V7_FEATURES].to_numpy(np.float32)
    X_event_val = X_event_val_df[V7_FEATURES].to_numpy(np.float32)
    X_event_test = X_event_test_df[V7_FEATURES].to_numpy(np.float32)

    summaries = {}
    selected_name = None
    selected_threshold = None
    selected_rows = None
    for name, model in train_event_models(X_event_train, y_event_train).items():
        print(f"[V7] Training event model: {name}...", flush=True)
        model.fit(X_event_train, y_event_train)
        pv = model.predict_proba(X_event_val)[:, 1]
        best, reason, rows = threshold_search(y_event_val, pv)
        summaries[name] = {"best_validation": best, "selection_reason": reason}
        print(f"  {name}: val recall={best['fall_recall']:.3f}, FPR={best['false_positive_rate']:.3f}, F1={best['fall_f1']:.3f}, threshold={best['threshold']:.2f}", flush=True)
        if selected_name is None or (best["fall_recall"], -best["false_positive_rate"], best["fall_f1"], best["fall_precision"]) > (summaries[selected_name]["best_validation"]["fall_recall"], -summaries[selected_name]["best_validation"]["false_positive_rate"], summaries[selected_name]["best_validation"]["fall_f1"], summaries[selected_name]["best_validation"]["fall_precision"]):
            selected_name = name
            selected_threshold = best["threshold"]
            selected_rows = rows

    # Lock event model/threshold using validation only, then refit event model on
    # all available non-test recordings. The threshold remains locked.
    print(f"[V7] Selected event model: {selected_name}, locked threshold={selected_threshold:.2f}", flush=True)
    all_train_val_df = pd.concat([X_event_train_df, X_event_val_df], ignore_index=True)
    all_train_val_y = np.concatenate([y_event_train, y_event_val])
    final_event_model = train_event_models(X_event_train, y_event_train)[selected_name]
    final_event_model.fit(all_train_val_df[V7_FEATURES].to_numpy(np.float32), all_train_val_y)
    pt = final_event_model.predict_proba(X_event_test)[:, 1]
    pred = (pt >= selected_threshold).astype(int)
    test_metrics = metrics(y_event_test, pred, pt)

    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "fall_event_confirmator_v7.joblib"
    joblib.dump({
        "model": final_event_model,
        "feature_columns": V7_FEATURES,
        "model_version": "bits2-v7-event-confirmator",
        "threshold": float(selected_threshold),
        "base_model_path": str(args.v6_model),
        "base_model_version": artifact.get("model_version", "unknown"),
        "subject_split": {"train": TRAIN_SUBJECTS, "validation": VAL_SUBJECTS, "test": TEST_SUBJECTS},
    }, out_model)

    report = {
        "version": "v7",
        "method": "validation_locked_event_level_model_on_v6_probability_trajectories",
        "seed": SEED,
        "input": args.input,
        "base_model": args.v6_model,
        "feature_count": len(V7_FEATURES),
        "features": V7_FEATURES,
        "subjects": {"train": TRAIN_SUBJECTS, "validation": VAL_SUBJECTS, "test": TEST_SUBJECTS},
        "recordings": {k: len(v) for k, v in recs.items()},
        "fall_recordings": {k: int(sum(r["label"] == "FALL" for r in v)) for k, v in recs.items()},
        "models_compared": list(train_event_models(X_event_train, y_event_train).keys()),
        "validation_model_scores": summaries,
        "selected_model": selected_name,
        "locked_threshold": float(selected_threshold),
        "selection_reason": summaries[selected_name]["selection_reason"],
        "targets": {"fall_recall": 0.95, "false_positive_rate": 0.05},
        "test": test_metrics,
        "training_seconds": round(time.perf_counter() - started, 2),
        "leakage_controls": [
            "Fixed 25/8/8 subject split from prior stages.",
            "Base classifier probabilities for validation/test come from a TRAIN-only base model.",
            "Event-model training uses subject-grouped 5-fold OOF base probabilities.",
            "Event threshold is selected on validation only and locked before test evaluation.",
            "Test subjects are never used for model or threshold selection.",
        ],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n[V7] COMPLETE", flush=True)
    print(json.dumps({"selected_model": selected_name, "threshold": selected_threshold, "test": test_metrics, "model": str(out_model), "report": args.report}, indent=2), flush=True)

if __name__ == "__main__":
    main()
