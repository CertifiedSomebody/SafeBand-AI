from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Allow `python tools/train_fall_v6.py ...` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai.bits2_features import FEATURE_COLUMNS
from ai.fall_event_features import FALL_EVENT_FEATURES
from ai.fall_v6_features import V6_FEATURES

SEED = 42


def make_models() -> dict[str, object]:
    """Return the V6 candidate classifiers.

    The tree counts are deliberately moderate: V6 is a model-comparison
    experiment, not a brute-force hyperparameter search. All models remain
    deterministic through the fixed random seed.
    """
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        ),
    }


def subject_split(subjects: pd.Series) -> tuple[list[str], list[str], list[str]]:
    """Return train/validation/test subject IDs using the established 25/8/8 split."""
    unique = sorted(set(subjects.astype(str)), key=int)
    rng = np.random.default_rng(SEED)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * 0.20))
    n_val = max(1, round(len(shuffled) * 0.20))
    test = sorted(shuffled[:n_test], key=int)
    validation = sorted(shuffled[n_test : n_test + n_val], key=int)
    train = sorted(shuffled[n_test + n_val :], key=int)
    return train, validation, test


def fall_probability(model, X: np.ndarray) -> np.ndarray:
    classes = [str(c).upper() for c in model.classes_]
    if "FALL" not in classes:
        raise RuntimeError(f"Model classes do not contain FALL: {model.classes_}")
    return model.predict_proba(X)[:, classes.index("FALL")]


def _recordings(df: pd.DataFrame, probabilities: np.ndarray) -> list[dict]:
    """Cache each recording's ordered probabilities once.

    This is the key V6 performance fix. The old implementation rebuilt a
    DataFrame and regrouped every recording for every rule combination.
    """
    required = ["subject_id", "source_file", "recording_type", "label", "start_sample"]
    d = df[required].copy()
    d["p"] = probabilities
    d["subject_id"] = d["subject_id"].astype(str)
    d = d.sort_values(
        ["subject_id", "source_file", "recording_type", "start_sample"],
        kind="stable",
    )

    out: list[dict] = []
    for (subject, source, recording_type), g in d.groupby(
        ["subject_id", "source_file", "recording_type"], sort=False
    ):
        out.append(
            {
                "subject_id": subject,
                "source_file": source,
                "recording_type": recording_type,
                "label": str(g["label"].iloc[0]).upper(),
                "starts": g["start_sample"].to_numpy(dtype=np.int32, copy=True),
                "p": g["p"].to_numpy(dtype=np.float32, copy=True),
            }
        )
    return out


def _confirmed(recording: dict, hit_threshold: float, peak_threshold: float, min_hits: int, max_gap: int) -> bool:
    """Apply one temporal event rule to one recording."""
    p = recording["p"]
    starts = recording["starts"]
    hit_indices = np.flatnonzero(p >= hit_threshold)
    if hit_indices.size < min_hits:
        return False

    run = 0
    last_start: int | None = None
    cluster_peak = 0.0

    for idx in hit_indices:
        start = int(starts[idx])
        if last_start is None or start - last_start <= max_gap:
            run += 1
            cluster_peak = max(cluster_peak, float(p[idx]))
        else:
            run = 1
            cluster_peak = float(p[idx])
        last_start = start

        if run >= min_hits and cluster_peak >= peak_threshold:
            return True

    return False


def evaluate_rule(recordings: list[dict], hit_threshold: float, peak_threshold: float, min_hits: int, max_gap: int) -> dict:
    y_true = np.fromiter((r["label"] == "FALL" for r in recordings), dtype=bool, count=len(recordings))
    y_pred = np.fromiter(
        (
            _confirmed(r, hit_threshold, peak_threshold, min_hits, max_gap)
            for r in recordings
        ),
        dtype=bool,
        count=len(recordings),
    )

    tp = int(np.sum(y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fall_count = max(1, int(np.sum(y_true)))
    nonfall_count = max(1, int(np.sum(~y_true)))

    recall = tp / fall_count
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    fpr = fp / nonfall_count

    return {
        "hit_threshold": round(float(hit_threshold), 3),
        "peak_threshold": round(float(peak_threshold), 3),
        "min_hits": int(min_hits),
        "max_gap_samples": int(max_gap),
        "fall_recall": float(recall),
        "fall_precision": float(precision),
        "fall_f1": float(f1),
        "false_positive_rate": float(fpr),
        "false_alarms": fp,
        "missed_falls": fn,
        "confusion_matrix": [[tp, fn], [fp, tn]],
    }


def search_rules(
    recordings: list[dict],
    hit_thresholds: np.ndarray,
    peak_thresholds: np.ndarray,
    min_hits_values: tuple[int, ...],
    max_gaps: tuple[int, ...],
) -> list[dict]:
    """Search the temporal rule on cached recording arrays.

    Complexity is now proportional to the number of candidate rules times the
    number of recordings, rather than repeatedly constructing/grouping full
    validation DataFrames. With the V6 grid this is ~3k lightweight checks.
    """
    total = len(hit_thresholds) * len(peak_thresholds) * len(min_hits_values) * len(max_gaps)
    rows: list[dict] = []
    done = 0
    last_report = time.perf_counter()

    for hit_threshold in hit_thresholds:
        for peak_threshold in peak_thresholds:
            for min_hits in min_hits_values:
                for max_gap in max_gaps:
                    rows.append(
                        evaluate_rule(
                            recordings,
                            float(hit_threshold),
                            float(peak_threshold),
                            int(min_hits),
                            int(max_gap),
                        )
                    )
                    done += 1
                    now = time.perf_counter()
                    if now - last_report >= 5.0 or done == total:
                        print(f"  Rule search: {done}/{total} ({done / total:.0%})", flush=True)
                        last_report = now
    return rows


def select_rule(rows: list[dict]) -> tuple[dict, str]:
    target_recall = 0.95
    target_fpr = 0.05

    both = [r for r in rows if r["fall_recall"] >= target_recall and r["false_positive_rate"] <= target_fpr]
    if both:
        pool = both
        reason = "validation_rule_meets_both_targets"
    else:
        recall_ok = [r for r in rows if r["fall_recall"] >= target_recall]
        if recall_ok:
            pool = recall_ok
            reason = "no_rule_meets_both_targets; recall_target_met"
        else:
            pool = rows
            reason = "no_rule_meets_recall_target"

    # Safety-oriented tie breaking: first maximize recall, then minimize FPR,
    # then maximize F1. We never use test results to choose this point.
    best = max(
        pool,
        key=lambda r: (
            r["fall_recall"],
            -r["false_positive_rate"],
            r["fall_f1"],
            r["fall_precision"],
        ),
    )
    return best, reason


def model_score(rows: list[dict]) -> dict:
    best = max(rows, key=lambda r: (r["fall_f1"], r["fall_recall"], -r["false_positive_rate"]))
    return {
        "best_validation_f1": best["fall_f1"],
        "best_validation_recall": best["fall_recall"],
        "best_validation_fpr": best["false_positive_rate"],
    }


def final_test_metrics(recordings: list[dict], rule: dict) -> dict:
    base = evaluate_rule(
        recordings,
        rule["hit_threshold"],
        rule["peak_threshold"],
        rule["min_hits"],
        rule["max_gap_samples"],
    )
    y_true = np.array([r["label"] == "FALL" for r in recordings], dtype=int)
    # Recording-level maximum probability is an evaluation diagnostic only;
    # it is not used to choose the temporal rule.
    y_score = np.array([float(np.max(r["p"])) for r in recordings], dtype=float)
    base["roc_auc_max_window_probability"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    base["recordings"] = len(recordings)
    base["fall_events"] = int(np.sum(y_true))
    base["nonfall_recordings"] = int(len(y_true) - np.sum(y_true))
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="SafeBand BITS-2 V6 subject-independent fall detector")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--report", default="models/bits2_fall_training_report_v6.json")
    parser.add_argument("--max-gap", type=int, default=30, help="Retained for compatibility; the rule search includes its own gap grid.")
    args = parser.parse_args()

    started = time.perf_counter()
    print("[V6] Loading window dataset...", flush=True)
    df = pd.read_csv(args.input, low_memory=False)

    required = {"subject_id", "source_file", "recording_type", "label", "start_sample"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = FEATURE_COLUMNS + FALL_EVENT_FEATURES + V6_FEATURES
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns ({len(missing_features)}): {missing_features}")

    train_subjects, validation_subjects, test_subjects = subject_split(df["subject_id"])
    subject_strings = df["subject_id"].astype(str)
    sets = {
        "train": df[subject_strings.isin(train_subjects)].copy(),
        "validation": df[subject_strings.isin(validation_subjects)].copy(),
        "test": df[subject_strings.isin(test_subjects)].copy(),
    }

    print(
        f"[V6] Windows: {len(df):,} | train {len(sets['train']):,} | "
        f"validation {len(sets['validation']):,} | test {len(sets['test']):,}",
        flush=True,
    )
    print(
        f"[V6] Subjects: train={len(train_subjects)}, validation={len(validation_subjects)}, test={len(test_subjects)}",
        flush=True,
    )

    # Use float32 matrices to keep memory pressure under control.
    X_train = sets["train"][features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    X_val = sets["validation"][features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    X_test = sets["test"][features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_train = sets["train"]["label"].astype(str).str.upper().to_numpy()
    y_val = sets["validation"]["label"].astype(str).str.upper().to_numpy()

    # Moderate, reproducible search grid. The implementation caches recording
    # probabilities, so this is no longer a nested pandas/groupby bottleneck.
    hit_thresholds = np.arange(0.40, 0.86, 0.025)
    peak_thresholds = np.arange(0.55, 0.91, 0.05)
    min_hits_values = (2, 3, 4)
    max_gaps = (15, 30, 45, 60)

    candidates = make_models()
    validation_rules: dict[str, list[dict]] = {}
    model_summaries: dict[str, dict] = {}

    for name, model in candidates.items():
        print(f"\n[V6] Training {name}...", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        pv = fall_probability(model, X_val)
        print(f"[V6] {name} fit complete in {time.perf_counter() - t0:.1f}s", flush=True)

        recordings = _recordings(sets["validation"], pv)
        print(f"[V6] Cached {len(recordings)} validation recordings. Searching event rules...", flush=True)
        rows = search_rules(recordings, hit_thresholds, peak_thresholds, min_hits_values, max_gaps)
        validation_rules[name] = rows
        model_summaries[name] = model_score(rows)
        print(f"[V6] {name}: best validation F1={model_summaries[name]['best_validation_f1']:.4f}", flush=True)

    # Select classifier using validation only. Test is still untouched.
    selected_model_name = max(
        model_summaries,
        key=lambda n: (
            model_summaries[n]["best_validation_f1"],
            model_summaries[n]["best_validation_recall"],
            -model_summaries[n]["best_validation_fpr"],
        ),
    )
    selected_rule, selection_reason = select_rule(validation_rules[selected_model_name])

    print(
        f"\n[V6] Selected model: {selected_model_name} | "
        f"rule: hit>={selected_rule['hit_threshold']}, peak>={selected_rule['peak_threshold']}, "
        f"hits={selected_rule['min_hits']}, gap={selected_rule['max_gap_samples']}",
        flush=True,
    )

    # Refit selected classifier on TRAIN + VALIDATION, then evaluate the locked
    # rule once on TEST. No test metric participates in selection.
    print("[V6] Refitting selected model on train + validation...", flush=True)
    fit_df = pd.concat([sets["train"], sets["validation"]], ignore_index=True)
    X_fit = fit_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y_fit = fit_df["label"].astype(str).str.upper().to_numpy()
    final_model = make_models()[selected_model_name]
    t0 = time.perf_counter()
    final_model.fit(X_fit, y_fit)
    print(f"[V6] Final fit complete in {time.perf_counter() - t0:.1f}s", flush=True)

    pt = fall_probability(final_model, X_test)
    test_recordings = _recordings(sets["test"], pt)
    test_metrics = final_test_metrics(test_recordings, selected_rule)

    report = {
        "version": "v6",
        "method": "validation_locked_temporal_event_rule_with_v6_features",
        "seed": SEED,
        "input": args.input,
        "feature_count": len(features),
        "features": features,
        "subjects": {
            "train": train_subjects,
            "validation": validation_subjects,
            "test": test_subjects,
        },
        "dataset_windows": {
            "total": int(len(df)),
            "train": int(len(sets["train"])),
            "validation": int(len(sets["validation"])),
            "test": int(len(sets["test"])),
        },
        "targets": {"fall_recall": 0.95, "false_positive_rate": 0.05},
        "models_compared": list(candidates.keys()),
        "validation_model_scores": model_summaries,
        "selected_model": selected_model_name,
        "selection_reason": selection_reason,
        "selected_operating_point": selected_rule,
        "validation_selected_rule_performance": selected_rule,
        "test": test_metrics,
        "validation_search_space": {
            "hit_thresholds": "0.40..0.85 step 0.025",
            "peak_thresholds": "0.55..0.90 step 0.05",
            "min_hits": list(min_hits_values),
            "max_gaps_samples": list(max_gaps),
        },
        "training_seconds": round(time.perf_counter() - started, 2),
    }

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "feature_columns": features,
        "model_version": "bits2-v6-fall",
        "event_rule": {
            "hit_threshold": selected_rule["hit_threshold"],
            "peak_threshold": selected_rule["peak_threshold"],
            "min_hits": selected_rule["min_hits"],
            "max_gap_samples": selected_rule["max_gap_samples"],
        },
        "subject_split": {
            "train": train_subjects,
            "validation": validation_subjects,
            "test": test_subjects,
        },
    }
    artifact_path = model_dir / "fall_detector_v6.joblib"
    joblib.dump(artifact, artifact_path)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[V6] COMPLETE", flush=True)
    print(json.dumps({"selected_model": selected_model_name, "selected_operating_point": selected_rule, "test": test_metrics, "model": str(artifact_path), "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
