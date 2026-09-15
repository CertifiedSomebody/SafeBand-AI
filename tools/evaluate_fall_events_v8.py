from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.fall_event_confirmator_v7 import V7_FEATURES, extract_event_features

SEED = 42
TRAIN_SUBJECTS = ["1","2","3","4","7","9","10","12","13","14","15","16","18","20","23","24","27","30","31","32","34","35","37","38","40"]
VAL_SUBJECTS = ["11","21","22","25","26","28","29","36"]
TEST_SUBJECTS = ["5","6","8","17","19","33","39","41"]

TARGET_RECALL = 0.95
TARGET_FPR = 0.05
CONTROLLED_FPR = 0.10


def load_v6(path: str):
    a = joblib.load(path)
    if "model" not in a or "feature_columns" not in a:
        raise ValueError("V6 artifact must contain model and feature_columns")
    return a


def fall_probability(model, X):
    classes = [str(c).upper() for c in model.classes_]
    return model.predict_proba(X)[:, classes.index("FALL")]


def make_recordings(df, p):
    cols = ["subject_id", "source_file", "recording_type", "label", "start_sample"]
    d = df[cols].copy()
    d["p"] = np.asarray(p, dtype=np.float32)
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


def event_frame(recs):
    rows, y = [], []
    for r in recs:
        row = extract_event_features(r["p"], r["starts"])
        row.update(subject_id=r["subject_id"], source_file=r["source_file"], recording_type=r["recording_type"])
        rows.append(row)
        y.append(1 if r["label"] == "FALL" else 0)
    return pd.DataFrame(rows), np.asarray(y, dtype=int)


def split_subjects(df):
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
        "false_alarms": int(fp),
        "missed_falls": int(fn),
        "confusion_matrix": [[int(tp), int(fn)], [int(fp), int(tn)]],
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
    }


def candidate_models():
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    models = {
        "logistic_regression": (LogisticRegression(max_iter=1200, class_weight="balanced", random_state=SEED), None),
        "extra_trees": (ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1), None),
        "random_forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1), None),
    }
    for w in (1.0, 1.5, 2.0, 3.0):
        models[f"hist_gradient_boosting_fn_weight_{w:g}"] = (
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.5, random_state=SEED),
            w,
        )
    return models


def score_model(model, X):
    return model.predict_proba(X)[:, 1]


def evaluate_rule(y, score, pmax, threshold, pmax_floor):
    pred = ((score >= threshold) & (pmax >= pmax_floor)).astype(int)
    return metrics(y, pred, score), pred


def search_operating_point(y, score, pmax):
    """Validation-only search. A pmax floor is an optional second evidence gate.

    Priority:
      1) Meet both safety targets if possible, maximize F1.
      2) Otherwise meet recall target, then maximize recall and minimize FPR.
      3) Otherwise maximize recall subject to FPR <= 10%.
      4) Otherwise maximize F1.
    """
    rows = []
    thresholds = np.arange(0.10, 0.901, 0.01)
    floors = np.concatenate(([0.0], np.arange(0.50, 0.951, 0.025)))
    for t in thresholds:
        for floor in floors:
            m, _ = evaluate_rule(y, score, pmax, float(t), float(floor))
            m.update(threshold=round(float(t), 3), pmax_floor=round(float(floor), 3))
            rows.append(m)

    both = [r for r in rows if r["fall_recall"] >= TARGET_RECALL and r["false_positive_rate"] <= TARGET_FPR]
    if both:
        pool, reason = both, "validation_meets_both_targets"
        best = max(pool, key=lambda r: (r["fall_f1"], r["fall_precision"], r["fall_recall"], -r["false_positive_rate"]))
    else:
        recall_ok = [r for r in rows if r["fall_recall"] >= TARGET_RECALL]
        if recall_ok:
            pool, reason = recall_ok, "no_rule_meets_both_targets; recall_target_met"
            best = max(pool, key=lambda r: (r["fall_recall"], -r["false_positive_rate"], r["fall_f1"], r["fall_precision"]))
        else:
            controlled = [r for r in rows if r["false_positive_rate"] <= CONTROLLED_FPR]
            if controlled:
                pool, reason = controlled, "no_rule_meets_recall_target; selected_best_recall_at_controlled_fpr"
                best = max(pool, key=lambda r: (r["fall_recall"], r["fall_f1"], r["fall_precision"], -r["false_positive_rate"]))
            else:
                pool, reason = rows, "no_rule_meets_recall_or_controlled_fpr; selected_best_f1"
                best = max(pool, key=lambda r: (r["fall_f1"], r["fall_recall"], r["fall_precision"], -r["false_positive_rate"]))
    return best, reason, rows


def main():
    ap = argparse.ArgumentParser(description="SafeBand BITS-2 V8 final fall feasibility benchmark")
    ap.add_argument("--input", required=True)
    ap.add_argument("--v6-model", required=True)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report", default="models/bits2_fall_training_report_v8.json")
    args = ap.parse_args()
    started = time.perf_counter()

    print("[V8] Loading windows and V6 artifact...", flush=True)
    df = pd.read_csv(args.input, low_memory=False)
    artifact = load_v6(args.v6_model)
    base = artifact["model"]
    features = artifact["feature_columns"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing V6 feature columns: {missing}")
    sets = split_subjects(df)
    print(f"[V8] Windows: total={len(df):,}, train={len(sets['train']):,}, validation={len(sets['validation']):,}, test={len(sets['test']):,}", flush=True)

    Xtr = sets["train"][features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
    ytr = sets["train"]["label"].astype(str).str.upper().to_numpy()
    print("[V8] Fitting TRAIN-only V6 base model...", flush=True)
    train_base = clone(base)
    train_base.fit(Xtr, ytr)

    recs = {}
    for name in ("train", "validation", "test"):
        X = sets[name][features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(np.float32)
        p = fall_probability(train_base, X)
        recs[name] = make_recordings(sets[name], p)
        print(f"[V8] {name}: {len(recs[name])} recordings cached", flush=True)

    # Subject-grouped OOF probabilities for event-model training.
    groups = sets["train"]["subject_id"].astype(str).to_numpy()
    oof = np.zeros(len(sets["train"]), dtype=np.float32)
    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    print(f"[V8] Generating {n_splits}-fold subject-grouped OOF base probabilities...", flush=True)
    for fold, (a, b) in enumerate(gkf.split(Xtr, ytr, groups), 1):
        m = clone(base)
        m.fit(Xtr[a], ytr[a])
        oof[b] = fall_probability(m, Xtr[b])
        print(f"  OOF fold {fold}/{n_splits} complete", flush=True)

    train_event_df, y_event_train = event_frame(make_recordings(sets["train"], oof))
    val_event_df, y_event_val = event_frame(recs["validation"])
    test_event_df, y_event_test = event_frame(recs["test"])
    Xe_tr = train_event_df[V7_FEATURES].to_numpy(np.float32)
    Xe_val = val_event_df[V7_FEATURES].to_numpy(np.float32)
    Xe_test = test_event_df[V7_FEATURES].to_numpy(np.float32)
    pmax_val = val_event_df["p_max"].to_numpy(np.float32)
    pmax_test = test_event_df["p_max"].to_numpy(np.float32)

    summaries = {}
    selected = None
    selected_best = None
    selected_model_obj = None

    for name, (model, fn_weight) in candidate_models().items():
        print(f"[V8] Training event model: {name}...", flush=True)
        if fn_weight is None:
            model.fit(Xe_tr, y_event_train)
        else:
            sw = np.where(y_event_train == 1, float(fn_weight), 1.0)
            model.fit(Xe_tr, y_event_train, sample_weight=sw)
        pv = score_model(model, Xe_val)
        best, reason, rows = search_operating_point(y_event_val, pv, pmax_val)
        summaries[name] = {
            "best_validation": best,
            "selection_reason": reason,
            "fn_weight": fn_weight,
            "roc_auc": float(roc_auc_score(y_event_val, pv)),
        }
        print(f"  {name}: recall={best['fall_recall']:.3f}, FPR={best['false_positive_rate']:.3f}, F1={best['fall_f1']:.3f}, t={best['threshold']:.2f}, pmax_floor={best['pmax_floor']:.3f}", flush=True)

        # Global candidate selection uses the same safety hierarchy as the rule search.
        rank = (
            int(best["fall_recall"] >= TARGET_RECALL and best["false_positive_rate"] <= TARGET_FPR),
            int(best["fall_recall"] >= TARGET_RECALL),
            int(best["false_positive_rate"] <= CONTROLLED_FPR),
            best["fall_recall"] if best["false_positive_rate"] <= CONTROLLED_FPR else 0.0,
            best["fall_f1"],
            best["fall_precision"],
            -best["false_positive_rate"],
        )
        if selected is None or rank > selected["rank"]:
            selected = {"name": name, "rank": rank}
            selected_best = best
            selected_model_obj = model

    selected_name = selected["name"]
    print(f"[V8] Selected: {selected_name}, threshold={selected_best['threshold']:.2f}, pmax_floor={selected_best['pmax_floor']:.3f}", flush=True)

    # Lock model/rule from validation. Refit the chosen event model on OOF-train + validation.
    final_model, fn_weight = candidate_models()[selected_name]
    all_X = np.vstack([Xe_tr, Xe_val])
    all_y = np.concatenate([y_event_train, y_event_val])
    if fn_weight is None:
        final_model.fit(all_X, all_y)
    else:
        sw = np.where(all_y == 1, float(fn_weight), 1.0)
        final_model.fit(all_X, all_y, sample_weight=sw)

    pt = score_model(final_model, Xe_test)
    test_metrics, test_pred = evaluate_rule(
        y_event_test, pt, pmax_test, selected_best["threshold"], selected_best["pmax_floor"]
    )

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_model = model_dir / "fall_event_confirmator_v8.joblib"
    joblib.dump({
        "model": final_model,
        "feature_columns": V7_FEATURES,
        "model_version": "bits2-v8-final-event-confirmator",
        "threshold": float(selected_best["threshold"]),
        "pmax_floor": float(selected_best["pmax_floor"]),
        "base_model_path": str(args.v6_model),
        "base_model_version": artifact.get("model_version", "unknown"),
        "subject_split": {"train": TRAIN_SUBJECTS, "validation": VAL_SUBJECTS, "test": TEST_SUBJECTS},
        "method": "validation_locked_final_event_benchmark",
    }, out_model)

    report = {
        "version": "v8",
        "method": "validation_locked_final_event_benchmark_on_v6_probability_trajectories",
        "seed": SEED,
        "input": args.input,
        "base_model": args.v6_model,
        "feature_count": len(V7_FEATURES),
        "features": V7_FEATURES,
        "subjects": {"train": TRAIN_SUBJECTS, "validation": VAL_SUBJECTS, "test": TEST_SUBJECTS},
        "recordings": {k: len(v) for k, v in recs.items()},
        "fall_recordings": {k: int(sum(r["label"] == "FALL" for r in v)) for k, v in recs.items()},
        "models_compared": list(candidate_models().keys()),
        "validation_model_scores": summaries,
        "selected_model": selected_name,
        "selected_operating_point": selected_best,
        "selection_reason": summaries[selected_name]["selection_reason"],
        "targets": {"fall_recall": TARGET_RECALL, "false_positive_rate": TARGET_FPR},
        "controlled_fpr_reference": CONTROLLED_FPR,
        "test": test_metrics,
        "test_predictions": {
            "positive_events": int(test_pred.sum()),
            "total_events": int(len(test_pred)),
        },
        "training_seconds": round(time.perf_counter() - started, 2),
        "leakage_controls": [
            "Fixed 25/8/8 subject split retained from prior stages.",
            "Validation/test base probabilities come from a TRAIN-only V6 base model.",
            "Event-model training uses subject-grouped 5-fold OOF base probabilities from training subjects.",
            "Operating point (event threshold and optional pmax evidence floor) is selected on validation only.",
            "Selected event model is refit on non-test training + validation event features after the rule is locked.",
            "Test subjects are never used for model or operating-point selection.",
        ],
        "final_stage_note": "This is the final BITS-2-only algorithmic benchmark. If the safety targets remain unmet, move to SafeBand hardware data rather than continuing dataset-specific tuning.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n[V8] COMPLETE", flush=True)
    print(json.dumps({"selected_model": selected_name, "operating_point": selected_best, "test": test_metrics, "model": str(out_model), "report": args.report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
