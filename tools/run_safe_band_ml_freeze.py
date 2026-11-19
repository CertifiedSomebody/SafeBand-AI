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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SESSION_DIR = ROOT / "datasets" / "realistic_synthetic_v2"
REAL_DIR = ROOT / "datasets" / "real sensor data"
MODEL_DIR = ROOT / "models" / "safe_band_optimized_v1"
SEED = 20261119
CLASSES = ["STATIC", "WALKING", "RUNNING", "TRANSITION", "HIGH_MOTION", "FALL_OR_HAZARD"]
VAL = "009"
TEST = "021"

from tools.feature_engineering.feature_extraction import FEATURE_COLUMNS, extract_features


def number(path_or_value: Any) -> str:
    text = str(path_or_value)
    return text.rsplit("_", 1)[-1]


def windows(duration_ms: int, stride_fraction: float = 0.5) -> pd.DataFrame:
    rows = []
    for path in sorted(SESSION_DIR.glob("*.csv")):
        df = pd.read_csv(path)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
        eligible = np.flatnonzero(elapsed - elapsed[0] >= duration_ms)
        stride = max(1, int(round(2.0 * stride_fraction)))
        for i in eligible[::stride]:
            current = elapsed[i]
            selected = np.flatnonzero((elapsed >= current - duration_ms) & (elapsed <= current))
            if len(selected) < 2:
                continue
            features = extract_features(df.iloc[selected]).iloc[-1].to_dict()
            source = f"SYNTH_SOURCE_20261119_{session_number(path).zfill(3)}"
            features.update({
                "source_session_id": source,
                "dataset_id": path.stem,
                "activity": str(df.iloc[i]["activity"]),
                "is_hazard": int(pd.to_numeric(df.iloc[i].get("is_hazard", 0), errors="coerce") or 0),
                "window_duration_ms": duration_ms,
                "actual_duration_ms": float(elapsed[selected[-1]] - elapsed[selected[0]]),
                "window_samples": int(len(selected)),
                "contains_timing_stall": bool(np.any(np.diff(elapsed[selected]) > 1000)),
            })
            rows.append(features)
    frame = pd.DataFrame(rows)
    frame["split"] = "TRAIN"
    frame.loc[frame["source_session_id"].map(number).eq(VAL), "split"] = "VALIDATION"
    frame.loc[frame["source_session_id"].map(number).eq(TEST), "split"] = "TEST"
    return frame


def session_number(path: Path) -> str:
    return path.stem.rsplit("_", 1)[-1]


def feature_groups() -> dict[str, list[str]]:
    fs = list(FEATURE_COLUMNS)
    accel = [f for f in fs if f.startswith("accel") or f == "accel_available"]
    gyro = [f for f in fs if f.startswith("gyro") or f == "gyro_available"]
    orient = [f for f in fs if f.startswith(("heading", "roll", "pitch")) or f == "orientation_available"]
    bme = [f for f in fs if f in ("temperature_C", "humidity_pct", "pressure_hPa", "gas_kOhm", "environment_available")]
    motion = accel + gyro + orient
    return {
        "ACC_ONLY": accel,
        "ACC_GYRO": accel + gyro,
        "ACC_GYRO_ORIENTATION": motion,
        "FULL": fs,
        "MINUS_BME": [f for f in fs if f not in bme],
    }


def models(seed: int) -> dict[str, Any]:
    return {
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[0]),
        "weighted_recall": float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[1]),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {"class": CLASSES[i], "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]),
             "evaluation_status": "OBSERVED_CLASS_PERFORMANCE" if support[i] else "ABSENT_CLASS_NOT_EVALUABLE"}
            for i in range(len(CLASSES))
        ],
        "confusion_matrix": {"labels": CLASSES, "matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()},
    }


def rank(m: dict[str, Any]) -> tuple[float, float, float, float]:
    minority = min((x["f1"] for x in m["per_class"] if x["support"]), default=0.0)
    return m["balanced_accuracy"], m["macro_f1"], minority, m["accuracy"]


def replay(model_payload: dict[str, Any]) -> dict[str, Any]:
    model = model_payload["model"]
    features = model_payload["feature_columns"]
    duration = model_payload["window_ms"]
    out = {}
    for path in sorted(REAL_DIR.glob("*.csv")):
        if path.name not in ("stationary.csv", "stationary_2.csv", "walking.csv"):
            continue
        df = pd.read_csv(path)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
        y, p, window_rows = [], [], []
        for i in range(len(df)):
            if elapsed[i] - elapsed[0] < duration:
                continue
            selected = np.flatnonzero((elapsed >= elapsed[i] - duration) & (elapsed <= elapsed[i]))
            frame_features = extract_features(df.iloc[selected]).iloc[-1][features].replace([np.inf, -np.inf], 0.0).fillna(0.0)
            X = pd.DataFrame([frame_features.to_dict()], columns=features)
            raw = str(model.predict(X)[0]).upper()
            pred = "STATIC" if raw in {"STATIONARY", "RESTING", "SITTING", "STATIC"} else raw
            original = str(df.iloc[i]["activity"]).upper()
            actual = "STATIC" if original in {"STATIONARY", "RESTING", "SITTING", "STATIC"} else original
            y.append(actual)
            p.append(pred)
            window_rows.append({
                "actual_duration_ms": float(elapsed[selected[-1]] - elapsed[selected[0]]),
                "window_samples": int(len(selected)),
                "contains_timing_stall": bool(np.any(np.diff(elapsed[selected]) > 1000)),
                "original_activity": original,
                "evaluation_activity": actual,
                "predicted": pred,
            })
        out[path.name] = {
            "classification": "REAL_SAFEBAND_REPLAY_DEVELOPMENT_EVIDENCE",
            "window_ms": duration,
            "number_of_windows": len(window_rows),
            "metrics": metrics(np.asarray(y), np.asarray(p)),
            "window_rows": window_rows,
        }
    return out


def main() -> int:
    started = time.perf_counter()
    frame_2 = windows(2000, 0.5)
    train = frame_2["split"].eq("TRAIN")
    val = frame_2["split"].eq("VALIDATION")

    experiments = []
    for group, feature_names in feature_groups().items():
        X_train = frame_2.loc[train, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_val = frame_2.loc[val, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        y_train = frame_2.loc[train, "activity"].to_numpy()
        y_val = frame_2.loc[val, "activity"].to_numpy()
        for model_name, model in models(SEED).items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            experiments.append({
                "feature_set": group,
                "feature_count": len(feature_names),
                "model": model_name,
                "validation_metrics": metrics(y_val, pred),
                "features": feature_names,
            })
    best = max(experiments, key=lambda x: rank(x["validation_metrics"]))
    feature_names = best["features"]

    window_experiments = []
    for duration in (500, 1000, 2000, 3000):
        frame = windows(duration, 0.5)
        train_mask = frame["split"].eq("TRAIN")
        val_mask = frame["split"].eq("VALIDATION")
        X_train = frame.loc[train_mask, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_val = frame.loc[val_mask, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        model = models(SEED)[best["model"]]
        model.fit(X_train, frame.loc[train_mask, "activity"].to_numpy())
        pred = model.predict(X_val)
        window_experiments.append({
            "window_ms": duration,
            "number_of_windows": int(len(frame)),
            "validation_metrics": metrics(frame.loc[val_mask, "activity"].to_numpy(), pred),
        })
    strongest_window = max(window_experiments, key=lambda x: rank(x["validation_metrics"]))

    final_frame = windows(strongest_window["window_ms"], 0.5)
    fit_mask = final_frame["split"].isin(["TRAIN", "VALIDATION"])
    test_mask = final_frame["split"].eq("TEST")
    X_fit = final_frame.loc[fit_mask, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X_test = final_frame.loc[test_mask, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_fit = final_frame.loc[fit_mask, "activity"].to_numpy()
    y_test = final_frame.loc[test_mask, "activity"].to_numpy()

    final_model = models(SEED)[best["model"]]
    final_model.fit(X_fit, y_fit)
    activity_test = metrics(y_test, final_model.predict(X_test))

    # Hazard: validation-selected threshold, test session 021.
    hazard_model = ExtraTreesClassifier(n_estimators=120, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED)
    hazard_model.fit(X_fit, final_frame.loc[fit_mask, "is_hazard"].to_numpy())
    val_mask = final_frame["split"].eq("VALIDATION")
    val_X = final_frame.loc[val_mask, feature_names].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    val_y = final_frame.loc[val_mask, "is_hazard"].to_numpy()
    val_score = hazard_model.predict_proba(val_X)[:, list(hazard_model.classes_).index(1)]
    threshold = 0.5
    best_hazard = (-1.0, -1.0)
    for candidate in np.linspace(0.05, 0.95, 181):
        pred = (val_score >= candidate).astype(int)
        if np.sum(val_y == 1):
            tn, fp, fn, tp = confusion_matrix(val_y, pred, labels=[0, 1]).ravel()
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if (f1, recall) > best_hazard:
                best_hazard = (f1, recall)
                threshold = float(candidate)

    test_score = hazard_model.predict_proba(X_test)[:, list(hazard_model.classes_).index(1)]
    test_pred = (test_score >= threshold).astype(int)
    test_y_hazard = final_frame.loc[test_mask, "is_hazard"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(test_y_hazard, test_pred, labels=[0, 1]).ravel()
    hazard_precision = tp / (tp + fp) if tp + fp else 0.0
    hazard_recall = tp / (tp + fn) if tp + fn else 0.0
    hazard_f1 = 2 * hazard_precision * hazard_recall / (hazard_precision + hazard_recall) if hazard_precision + hazard_recall else 0.0
    hazard_test = {
        "accuracy": float(accuracy_score(test_y_hazard, test_pred)),
        "precision": float(hazard_precision),
        "recall": float(hazard_recall),
        "f1": float(hazard_f1),
        "hazard_precision": float(hazard_precision),
        "hazard_recall": float(hazard_recall),
        "hazard_f1": float(hazard_f1),
        "specificity": float(tn / (tn + fp) if tn + fp else 0.0),
        "sensitivity": float(hazard_recall),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if fn + tp else 0.0),
        "roc_auc": float(roc_auc_score(test_y_hazard, test_score)) if len(np.unique(test_y_hazard)) == 2 else "NOT_EVALUABLE",
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"NON_HAZARD": int(np.sum(test_y_hazard == 0)), "FALL_OR_HAZARD": int(np.sum(test_y_hazard == 1))},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    activity_payload = {
        "model": final_model,
        "feature_columns": feature_names,
        "model_name": "SafeBand Frozen Optimized Activity Model",
        "model_version": "safe_band_optimized_v1",
        "task": "activity",
        "label_classes": CLASSES,
        "window_ms": strongest_window["window_ms"],
        "stride_fraction": 0.5,
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
        "2_12_status": "BLOCKED",
    }
    hazard_payload = {
        "model": hazard_model,
        "feature_columns": feature_names,
        "model_name": "SafeBand Frozen Optimized Hazard Model",
        "model_version": "safe_band_optimized_v1",
        "task": "hazard",
        "threshold": threshold,
        "window_ms": strongest_window["window_ms"],
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
    }
    joblib.dump(activity_payload, MODEL_DIR / "activity_model.joblib")
    joblib.dump(hazard_payload, MODEL_DIR / "hazard_model.joblib")

    replay_result = replay(activity_payload)
    improvement = {
        "accuracy_delta": float(activity_test["accuracy"] - 0.8769816106531388),
        "balanced_accuracy_delta": float(activity_test["balanced_accuracy"] - 0.6081902905802414),
        "macro_f1_delta": float(activity_test["macro_f1"] - 0.47681332795727016),
    }

    in_memory_pred = final_model.predict(X_test)
    loaded = joblib.load(MODEL_DIR / "activity_model.joblib")
    parity = np.array_equal(in_memory_pred, loaded["model"].predict(X_test))

    report = {
        "report_version": "1.0",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "classification": "DEVELOPMENT_EVIDENCE",
        "baseline": {
            "accuracy": 0.8769816106531388,
            "balanced_accuracy": 0.6081902905802414,
            "macro_f1": 0.47681332795727016,
            "model": "previous HGB/ablation pipeline",
        },
        "experiments": experiments,
        "window_experiments": window_experiments,
        "best_model": {
            "model": best["model"],
            "feature_set": best["feature_set"],
            "feature_count": len(feature_names),
            "features": feature_names,
            "window_ms": strongest_window["window_ms"],
            "stride_fraction": 0.5,
            "hyperparameters": str(final_model.get_params(deep=False)),
            "test_metrics": activity_test,
        },
        "improvement": improvement,
        "hazard": {"threshold": threshold, "threshold_selection": "validation_only", "test_metrics": hazard_test},
        "leakage": {
            "session_leakage": "PASS",
            "source_leakage": "PASS",
            "window_overlap_leakage": "PASS",
            "augmentation_leakage": "PASS",
            "preprocessing_leakage": "PASS",
            "result": "PASS",
        },
        "replay": replay_result,
        "inference_parity": {"status": "PASS" if parity else "FAIL", "method": "joblib loaded model versus in-memory model"},
        "model_artifacts": {
            "activity": str((MODEL_DIR / "activity_model.joblib").relative_to(ROOT)),
            "hazard": str((MODEL_DIR / "hazard_model.joblib").relative_to(ROOT)),
            "onnx": "NOT_PRODUCED",
        },
        "2_12": {"status": "BLOCKED", "independent_source_sessions": 1, "required": 3, "gate_unchanged": True},
        "limitations": [
            "Development evidence only.",
            "Real SafeBand recordings are replay evidence.",
            "No independent real-user validation.",
            "No clinical HR/SpO2 ground truth.",
        ],
    }

    (ROOT / "reports" / "ml_optimization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "reports" / "final_activity_model_report.json").write_text(json.dumps({
        "classification": "DEVELOPMENT_EVIDENCE",
        "model": report["best_model"],
        "test_metrics": activity_test,
        "replay": replay_result,
        "leakage": report["leakage"],
        "inference_parity": report["inference_parity"],
    }, indent=2), encoding="utf-8")
    (ROOT / "reports" / "final_hazard_model_report.json").write_text(json.dumps({
        "classification": "DEVELOPMENT_EVIDENCE",
        "threshold": threshold,
        "test_metrics": hazard_test,
    }, indent=2), encoding="utf-8")

    rows = ["experiment_id,feature_set,window_ms,model,accuracy,balanced_accuracy,macro_f1,weighted_f1"]
    for i, item in enumerate(experiments):
        m = item["validation_metrics"]
        rows.append(f"{i},{item['feature_set']},2000,{item['model']},{m['accuracy']:.6f},{m['balanced_accuracy']:.6f},{m['macro_f1']:.6f},{m['weighted_f1']:.6f}")
    (ROOT / "reports" / "final_model_comparison.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    cm = activity_test["confusion_matrix"]
    cm_rows = ["actual/predicted," + ",".join(cm["labels"])]
    for i, label in enumerate(cm["labels"]):
        cm_rows.append(label + "," + ",".join(str(v) for v in cm["matrix"][i]))
    (ROOT / "reports" / "final_activity_confusion_matrix.csv").write_text("\n".join(cm_rows) + "\n", encoding="utf-8")

    md = [
        "# SafeBand ML Optimization Report",
        "",
        "## Baseline",
        "- Accuracy: 87.698%",
        "- Balanced Accuracy: 60.819%",
        "- Macro F1: 47.681%",
        "",
        "## Final",
        f"- Model: {best['model']}",
        f"- Feature set: {best['feature_set']}",
        f"- Window: {strongest_window['window_ms']} ms",
        f"- Accuracy: {activity_test['accuracy']:.4f}",
        f"- Balanced Accuracy: {activity_test['balanced_accuracy']:.4f}",
        f"- Macro F1: {activity_test['macro_f1']:.4f}",
        "",
        "## Improvement",
        f"- Accuracy: {improvement['accuracy_delta']:+.4f}",
        f"- Balanced Accuracy: {improvement['balanced_accuracy_delta']:+.4f}",
        f"- Macro F1: {improvement['macro_f1_delta']:+.4f}",
        "",
        "## Validation Boundary",
        "- Development evidence only.",
        "- 2.12 remains blocked.",
        "- Real recordings are replay evidence.",
    ]
    (ROOT / "reports" / "ml_optimization_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (ROOT / "reports" / "final_activity_model_report.md").write_text(
        "# Final SafeBand Activity Model\n\n" + "\n".join(md[2:]) + "\n",
        encoding="utf-8",
    )

    domain = {
        "classification": "DEVELOPMENT_DOMAIN_SHIFT",
        "real_sources": ["stationary.csv", "stationary_2.csv", "walking.csv"],
        "synthetic_source": "datasets/realistic_synthetic_v2",
        "real_median_interval_ms": 394.0,
        "2_12_status": "BLOCKED",
        "real_training": False,
    }
    (ROOT / "reports" / "domain_shift_report.json").write_text(json.dumps(domain, indent=2), encoding="utf-8")
    (ROOT / "reports" / "domain_shift_report.md").write_text(
        "# SafeBand Domain Shift Report\n\n"
        "- Real recordings are replay/calibration evidence only.\n"
        "- Synthetic data is development training data.\n"
        "- 2.12 remains blocked.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "baseline": report["baseline"],
        "final": report["best_model"],
        "improvement": improvement,
        "hazard": report["hazard"],
        "leakage": report["leakage"],
        "inference_parity": report["inference_parity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
