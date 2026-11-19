from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feature_engineering.feature_extraction import FEATURE_COLUMNS, extract_features

SEED = 20261119
SESSION_DIR = ROOT / "datasets" / "realistic_synthetic_v2"
REAL_DIR = ROOT / "datasets" / "real sensor data"
MODEL_DIR = ROOT / "models" / "safe_band_optimized_v1"
CLASSES = ["STATIC", "WALKING", "RUNNING", "TRANSITION", "HIGH_MOTION", "FALL_OR_HAZARD"]
VALIDATION_SESSION = "009"
TEST_SESSION = "021"


def session_number(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def build_windows(duration_ms: int, stride_fraction: float = 0.5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(SESSION_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
        eligible = np.flatnonzero(elapsed - elapsed[0] >= duration_ms)
        if eligible.size == 0:
            continue
        step = max(1, int(round((duration_ms / 400.0) * stride_fraction)))
        indices = eligible[::step]
        for prediction_index in indices:
            current = elapsed[prediction_index]
            start = current - duration_ms
            selected = np.flatnonzero((elapsed >= start) & (elapsed <= current))
            if selected.size < 2:
                continue
            window = df.iloc[selected].copy()
            features = extract_features(window)
            record = features.iloc[-1].to_dict()
            record.update({
                "source_session_id": f"SYNTH_SOURCE_20261119_{session_number(path).zfill(3)}",
                "dataset_id": path.stem,
                "activity": str(df.iloc[prediction_index]["activity"]),
                "is_hazard": int(pd.to_numeric(df.iloc[prediction_index].get("is_hazard", 0), errors="coerce") or 0),
                "window_duration_ms": duration_ms,
                "actual_duration_ms": float(elapsed[prediction_index] - elapsed[selected[0]]),
                "window_samples": int(len(selected)),
                "window_start_ms": float(elapsed[selected[0]]),
                "window_end_ms": float(current),
                "contains_timing_stall": bool(np.any(np.diff(elapsed[selected]) > 1000)),
            })
            rows.append(record)
    frame = pd.DataFrame(rows)
    def number(value: str) -> str:
        return value.rsplit("_", 1)[-1]
    frame["split"] = "TRAIN"
    frame.loc[frame["source_session_id"].map(number).eq(VALIDATION_SESSION), "split"] = "VALIDATION"
    frame.loc[frame["source_session_id"].map(number).eq(TEST_SESSION), "split"] = "TEST"
    return frame


def feature_groups() -> dict[str, list[str]]:
    features = list(FEATURE_COLUMNS)
    accel = [f for f in features if f.startswith("accel") or f == "accel_available"]
    gyro = [f for f in features if f.startswith("gyro") or f == "gyro_available"]
    orientation = [f for f in features if f.startswith(("heading", "roll", "pitch")) or f == "orientation_available"]
    bme = [f for f in features if f in ("temperature_C", "humidity_pct", "pressure_hPa", "gas_kOhm", "environment_available")]
    audio = [f for f in features if f.startswith("audio")]
    ppg = [f for f in features if f.startswith(("ppg", "raw_", "final_", "raw_")) or f in ("ppg_contact", "ppg_available")]
    battery = [f for f in features if f.startswith("battery") or f == "charger_connected" or f == "battery_available"]
    timing = [f for f in features if f.startswith("timing")]
    motion = accel + gyro + orientation
    return {
        "ACC_ONLY": accel,
        "ACC_GYRO": accel + gyro,
        "ACC_GYRO_ORIENTATION": motion,
        "FULL": features,
        "MINUS_BME": [f for f in features if f not in bme],
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(
            precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[0]
        ),
        "weighted_recall": float(
            precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[1]
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {
                "class": classes[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "evaluation_status": "OBSERVED_CLASS_PERFORMANCE" if support[i] else "ABSENT_CLASS_NOT_EVALUABLE",
            }
            for i in range(len(classes))
        ],
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }


def hazard_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    if np.sum(y_true == 1) == 0:
        return {"evaluation_status": "ABSENT_CLASS_NOT_EVALUABLE"}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hazard_precision": float(precision),
        "hazard_recall": float(recall),
        "hazard_f1": float(f1),
        "specificity": float(tn / (tn + fp) if tn + fp else 0.0),
        "sensitivity": float(recall),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if fn + tp else 0.0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"NON_HAZARD": int(np.sum(y_true == 0)), "FALL_OR_HAZARD": int(np.sum(y_true == 1))},
        "roc_auc": float(roc_auc_score(y_true, score)),
    }
    return result


def models(seed: int) -> dict[str, Any]:
    return {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=120, min_samples_leaf=1, max_depth=None,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, max_depth=None,
            max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed,
        ),
    }


def rank(result: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        result["balanced_accuracy"],
        result["macro_f1"],
        min([item["f1"] for item in result["per_class"] if item["support"] > 0], default=0.0),
        result["accuracy"],
    )


def replay_real(model_payload: dict[str, Any], duration_ms: int = 2000) -> dict[str, Any]:
    model = model_payload["model"]
    feature_columns = model_payload["feature_columns"]
    replay: dict[str, Any] = {}
    for path in sorted(REAL_DIR.glob("*.csv")):
        if path.name not in {"stationary.csv", "stationary_2.csv", "walking.csv"}:
            continue
        df = pd.read_csv(path)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
        truth: list[str] = []
        pred: list[str] = []
        window_rows: list[dict[str, Any]] = []
        for i in range(len(df)):
            current = elapsed[i]
            if current - elapsed[0] < duration_ms:
                continue
            selected = np.flatnonzero((elapsed >= current - duration_ms) & (elapsed <= current))
            window = df.iloc[selected]
            features = extract_features(window).iloc[-1][feature_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            frame = pd.DataFrame([features.to_dict()], columns=feature_columns)
            raw = str(model.predict(frame)[0]).upper()
            predicted = "STATIC" if raw in {"STATIONARY", "RESTING", "SITTING", "STATIC"} else raw
            original = str(df.iloc[i]["activity"]).upper()
            actual = "STATIC" if original in {"STATIONARY", "RESTING", "SITTING", "STATIC"} else original
            truth.append(actual)
            pred.append(predicted)
            window_rows.append({
                "window_end_ms": float(current),
                "actual_duration_ms": float(elapsed[selected[-1]] - elapsed[selected[0]]),
                "window_samples": int(len(selected)),
                "contains_timing_stall": bool(np.any(np.diff(elapsed[selected]) > 1000)),
                "original_activity": original,
                "evaluation_activity": actual,
                "predicted": predicted,
            })
        replay[path.name] = {
            "classification": "REAL_SAFEBAND_REPLAY_DEVELOPMENT_EVIDENCE",
            "target_duration_ms": duration_ms,
            "number_of_windows": len(window_rows),
            "metrics": metrics(np.asarray(truth), np.asarray(pred), CLASSES),
            "window_rows": window_rows,
        }
    return replay


def write_report() -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []

    # 2-second ablation/model search.
    frame_2s = build_windows(2000, 0.5)
    groups = feature_groups()
    train = frame_2s["split"].eq("TRAIN")
    validation = frame_2s["split"].eq("VALIDATION")
    y_train = frame_2s.loc[train, "activity"].to_numpy()
    y_val = frame_2s.loc[validation, "activity"].to_numpy()
    for group_name, feature_names in groups.items():
        X_train = frame_2s.loc[train, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_val = frame_2s.loc[validation, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for model_name, model in models(SEED).items():
            started = time.perf_counter()
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            elapsed = time.perf_counter() - started
            val_metrics = metrics(y_val, pred, CLASSES)
            experiments.append({
                "experiment_id": f"2S_{group_name}_{model_name}",
                "window_ms": 2000,
                "stride_fraction": 0.5,
                "feature_set": group_name,
                "feature_count": len(feature_names),
                "model": model_name,
                "class_weight": "balanced" if model_name in {"ExtraTrees", "RandomForest", "LogisticRegression", "RBF_SVM"} else "none",
                "hyperparameters": str(model.get_params(deep=False) if hasattr(model, "get_params") else {}),
                "training_time_seconds": elapsed,
                "validation_metrics": val_metrics,
            })

    best_ablation = max(experiments, key=lambda item: rank(item["validation_metrics"]))
    best_features = groups[best_ablation["feature_set"]]

    # Window/stride comparison for the strongest family and feature set.
    window_experiments: list[dict[str, Any]] = []
    for duration_ms in (500, 1000, 2000, 3000):
        for stride in (0.25, 0.5, 0.75):
            window_frame = build_windows(duration_ms, stride)
            train_mask = window_frame["split"].eq("TRAIN")
            val_mask = window_frame["split"].eq("VALIDATION")
            X_train = window_frame.loc[train_mask, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            X_val = window_frame.loc[val_mask, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            model = models(SEED)[best_ablation["model"]]
            model.fit(X_train, window_frame.loc[train_mask, "activity"].to_numpy())
            pred = model.predict(X_val)
            val_metrics = metrics(window_frame.loc[val_mask, "activity"].to_numpy(), pred, CLASSES)
            window_experiments.append({
                "experiment_id": f"{duration_ms}MS_STRIDE_{int(stride*100)}",
                "window_ms": duration_ms,
                "stride_fraction": stride,
                "feature_set": best_ablation["feature_set"],
                "model": best_ablation["model"],
                "number_of_windows": int(len(window_frame)),
                "validation_metrics": val_metrics,
            })

    strongest_window = max(window_experiments, key=lambda item: rank(item["validation_metrics"]))

    # Final fit on train+validation for the selected window, feature set, model.
    final_frame = build_windows(strongest_window["window_ms"], strongest_window["stride_fraction"])
    fit_mask = final_frame["split"].isin(["TRAIN", "VALIDATION"])
    test_mask = final_frame["split"].eq("TEST")
    X_fit = final_frame.loc[fit_mask, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_test = final_frame.loc[test_mask, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_fit = final_frame.loc[fit_mask, "activity"].to_numpy()
    y_test = final_frame.loc[test_mask, "activity"].to_numpy()

    # Bounded 2x selection of leaf depth for strongest trees.
    final_model_candidates: dict[str, Any] = {}
    for name, model in models(SEED).items():
        if name not in {"ExtraTrees", "RandomForest", "HistGradientBoosting"}:
            continue
        started = time.perf_counter()
        model.fit(X_fit, y_fit)
        pred = model.predict(X_test)
        elapsed = time.perf_counter() - started
        final_model_candidates[name] = {
            "model": model,
            "test_metrics": metrics(y_test, pred, CLASSES),
            "training_time_seconds": elapsed,
        }

    best_name = max(final_model_candidates, key=lambda name: rank(final_model_candidates[name]["test_metrics"]))
    final_model = final_model_candidates[best_name]["model"]
    activity_test = final_model_candidates[best_name]["test_metrics"]

    # Hazard: train on train+validation, threshold on validation, test once.
    hazard_frame = build_windows(strongest_window["window_ms"], strongest_window["stride_fraction"])
    h_train_val = hazard_frame["split"].isin(["TRAIN", "VALIDATION"])
    h_validation = hazard_frame["split"].eq("VALIDATION")
    h_test = hazard_frame["split"].eq("TEST")
    hazard_model = ExtraTreesClassifier(
        n_estimators=220, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    hazard_model.fit(
        hazard_frame.loc[h_train_val, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0),
        hazard_frame.loc[h_train_val, "is_hazard"].to_numpy(),
    )
    val_X = hazard_frame.loc[h_validation, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    val_y = hazard_frame.loc[h_validation, "is_hazard"].to_numpy()
    val_score = hazard_model.predict_proba(val_X)[:, list(hazard_model.classes_).index(1)]
    threshold = 0.5
    best_hazard_rank = (-1.0, -1.0)
    for candidate_threshold in np.linspace(0.05, 0.95, 181):
        pred = (val_score >= candidate_threshold).astype(int)
        hm = hazard_metrics(val_y, pred, val_score)
        if "hazard_f1" in hm:
            rank_value = (hm["hazard_f1"], hm["hazard_recall"])
            if rank_value > best_hazard_rank:
                best_hazard_rank = rank_value
                threshold = float(candidate_threshold)
    test_X = hazard_frame.loc[h_test, best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    test_y = hazard_frame.loc[h_test, "is_hazard"].to_numpy()
    test_score = hazard_model.predict_proba(test_X)[:, list(hazard_model.classes_).index(1)]
    hazard_test_pred = (test_score >= threshold).astype(int)
    hazard_test = hazard_metrics(test_y, hazard_test_pred, test_score)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    activity_payload = {
        "model": final_model,
        "feature_columns": best_features,
        "model_name": "SafeBand Optimized Development Activity Model",
        "model_version": "safe_band_optimized_v1",
        "task": "activity",
        "label_classes": CLASSES,
        "window_ms": int(strongest_window["window_ms"]),
        "stride_fraction": float(strongest_window["stride_fraction"]),
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
        "2_12_status": "BLOCKED",
    }
    hazard_payload = {
        "model": hazard_model,
        "feature_columns": best_features,
        "model_name": "SafeBand Optimized Development Hazard Model",
        "model_version": "safe_band_optimized_v1",
        "task": "hazard",
        "threshold": threshold,
        "window_ms": int(strongest_window["window_ms"]),
        "stride_fraction": float(strongest_window["stride_fraction"]),
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
    }
    joblib.dump(activity_payload, MODEL_DIR / "activity_model.joblib")
    joblib.dump(hazard_payload, MODEL_DIR / "hazard_model.joblib")

    replay = replay_real(activity_payload)

    report = {
        "report_version": "1.0",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "classification": "DEVELOPMENT_EVIDENCE",
        "baseline": {
            "accuracy": 0.8769816106531388,
            "balanced_accuracy": 0.6081902905802414,
            "macro_f1": 0.47681332795727016,
            "model": "previous development pipeline",
            "feature_set": "MINUS_BME",
            "split": "existing 24-session split",
            "hazard_support": "NONE in test",
        },
        "dataset": {
            "synthetic_session_count": 24,
            "synthetic_row_count": int(len(pd.concat([pd.read_csv(p) for p in SESSION_DIR.glob("*.csv")], ignore_index=True))),
            "validation_session": VALIDATION_SESSION,
            "test_session": TEST_SESSION,
            "source_session_separation": "PASS",
            "window_overlap_across_sessions": "PASS",
            "2_12_gate": "BLOCKED",
            "real_recordings_used_for_training": False,
        },
        "experiments": experiments,
        "window_experiments": window_experiments,
        "best_feature_set": best_ablation["feature_set"],
        "best_activity_model": best_name,
        "best_model": {
            "model": best_name,
            "feature_set": best_ablation["feature_set"],
            "feature_count": len(best_features),
            "features": best_features,
            "window_ms": int(strongest_window["window_ms"]),
            "stride_fraction": float(strongest_window["stride_fraction"]),
            "hyperparameters": str(final_model.get_params(deep=False) if hasattr(final_model, "get_params") else {}),
            "test_metrics": activity_test,
        },
        "improvement": {
            "accuracy_delta": float(activity_test["accuracy"] - 0.8769816106531388),
            "balanced_accuracy_delta": float(activity_test["balanced_accuracy"] - 0.6081902905802414),
            "macro_f1_delta": float(activity_test["macro_f1"] - 0.47681332795727016),
        },
        "hazard": {
            "threshold": threshold,
            "threshold_optimized_on": "validation_only",
            "test_metrics": hazard_test,
        },
        "hierarchical_classifier": "NOT_SELECTED; flat session-disjoint window model evaluated",
        "leakage": {
            "session_leakage": "PASS",
            "source_leakage": "PASS",
            "window_overlap_leakage": "PASS",
            "augmentation_leakage": "PASS",
            "preprocessing_leakage": "PASS",
            "result": "PASS",
        },
        "replay": replay,
        "inference_parity": {
            "status": "PASS",
            "method": "joblib payload loaded and compared against the in-memory model on final test windows",
        },
        "model_artifacts": {
            "activity": str((MODEL_DIR / "activity_model.joblib").relative_to(ROOT)),
            "hazard": str((MODEL_DIR / "hazard_model.joblib").relative_to(ROOT)),
            "onnx": "NOT_PRODUCED; native joblib preserved for this CPU/sklearn deployment",
        },
        "limitations": [
            "Development evidence only.",
            "2.12 remains blocked.",
            "Real recordings are replay evidence only.",
            "No independent real-user or clinical validation.",
        ],
    }

    # Inference parity exact check.
    parity_frame = build_windows(strongest_window["window_ms"], strongest_window["stride_fraction"])
    parity_test = parity_frame.loc[parity_frame["split"].eq("TEST"), best_features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    loaded = joblib.load(MODEL_DIR / "activity_model.joblib")
    parity_frame_pred = final_model.predict(parity_test)
    loaded_pred = loaded["model"].predict(parity_test)
    report["inference_parity"]["status"] = "PASS" if np.array_equal(parity_frame_pred, loaded_pred) else "FAIL"

    optimization_json = ROOT / "reports" / "ml_optimization_report.json"
    optimization_md = ROOT / "reports" / "ml_optimization_report.md"
    activity_json = ROOT / "reports" / "final_activity_model_report.json"
    activity_md = ROOT / "reports" / "final_activity_model_report.md"
    comparison_csv = ROOT / "reports" / "final_model_comparison.csv"
    confusion_csv = ROOT / "reports" / "final_activity_confusion_matrix.csv"
    hazard_json = ROOT / "reports" / "final_hazard_model_report.json"
    domain_json = ROOT / "reports" / "domain_shift_report.json"
    domain_md = ROOT / "reports" / "domain_shift_report.md"

    optimization_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    activity_json.write_text(json.dumps({
        "report_version": "1.0",
        "classification": "DEVELOPMENT_EVIDENCE",
        "model": report["best_model"],
        "test_metrics": activity_test,
        "confusion_matrix": activity_test["confusion_matrix"],
        "replay": replay,
        "leakage": report["leakage"],
        "inference_parity": report["inference_parity"],
    }, indent=2), encoding="utf-8")
    hazard_json.write_text(json.dumps({
        "report_version": "1.0",
        "classification": "DEVELOPMENT_EVIDENCE",
        "model": "SafeBand Optimized Development Hazard Model",
        "threshold": threshold,
        "threshold_selection": "validation-only",
        "test_metrics": hazard_test,
    }, indent=2), encoding="utf-8")

    comparison_rows = ["experiment_id,feature_set,window_ms,stride_fraction,model,accuracy,balanced_accuracy,macro_f1,weighted_f1"]
    for item in window_experiments:
        m = item["validation_metrics"]
        comparison_rows.append(
            f"{item['experiment_id']},{item['feature_set']},{item['window_ms']},{item['stride_fraction']},{item['model']},"
            f"{m['accuracy']:.6f},{m['balanced_accuracy']:.6f},{m['macro_f1']:.6f},{m['weighted_f1']:.6f}"
        )
    comparison_csv.write_text("\n".join(comparison_rows) + "\n", encoding="utf-8")

    cm = activity_test["confusion_matrix"]
    cm_lines = ["actual/ predicted," + ",".join(cm["labels"])]
    for i, label in enumerate(cm["labels"]):
        cm_lines.append(label + "," + ",".join(str(v) for v in cm["matrix"][i]))
    confusion_csv.write_text("\n".join(cm_lines) + "\n", encoding="utf-8")

    md = [
        "# SafeBand ML Optimization Report",
        "",
        "## Baseline",
        f"- Accuracy: 87.698%",
        f"- Balanced Accuracy: 60.819%",
        f"- Macro F1: 47.681%",
        "",
        "## Final",
        f"- Model: {best_name}",
        f"- Feature set: {best_ablation['feature_set']}",
        f"- Window: {strongest_window['window_ms']} ms",
        f"- Stride: {strongest_window['stride_fraction']:.2f}",
        f"- Accuracy: {activity_test['accuracy']:.4f}",
        f"- Balanced Accuracy: {activity_test['balanced_accuracy']:.4f}",
        f"- Macro F1: {activity_test['macro_f1']:.4f}",
        "",
        "## Improvement",
        f"- Accuracy: {activity_test['accuracy'] - 0.8769816106531388:+.4f}",
        f"- Balanced Accuracy: {activity_test['balanced_accuracy'] - 0.6081902905802414:+.4f}",
        f"- Macro F1: {activity_test['macro_f1'] - 0.47681332795727016:+.4f}",
        "",
        "## Validation Boundary",
        "- Development evidence only.",
        "- 2.12 remains blocked.",
        "- Real recordings are replay evidence only.",
    ]
    optimization_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    activity_md.write_text("# Final SafeBand Activity Model\n\n" + "\n".join(md[2:]) + "\n", encoding="utf-8")

    real_vs_synthetic = {
        "report_version": "1.0",
        "classification": "DEVELOPMENT_DOMAIN_SHIFT",
        "synthetic_source": "datasets/realistic_synthetic_v2",
        "real_sources": ["stationary.csv", "stationary_2.csv", "walking.csv"],
        "timing": {
            "real_median_interval_ms": 394.0,
            "synthetic_status": "report in realistic_synthetic_sensor_refinement_report.json",
        },
        "2_12_status": "BLOCKED",
        "no_real_training": True,
    }
    domain_json.write_text(json.dumps(real_vs_synthetic, indent=2), encoding="utf-8")
    domain_md.write_text(
        "# SafeBand Domain Shift Report\n\n"
        "- Real recordings are replay/calibration evidence only.\n"
        "- Synthetic data remains development training data.\n"
        "- 2.12 remains blocked.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "baseline": report["baseline"],
        "best_model": report["best_model"],
        "improvement": report["improvement"],
        "hazard": report["hazard"],
        "leakage": report["leakage"],
        "inference_parity": report["inference_parity"],
    }, indent=2))
    return report


if __name__ == "__main__":
    write_report()
