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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feature_engineering.feature_extraction import FEATURE_COLUMNS, extract_features

SEED = 20261119
SESSION_DIR = ROOT / "datasets" / "realistic_synthetic_v2"
REAL_DIR = ROOT / "datasets" / "real sensor data"
OUTPUT_MODEL_DIR = ROOT / "models" / "safe_band_development_v1"
REPORT_JSON = ROOT / "reports" / "final_ml_training_report.json"
REPORT_MD = ROOT / "reports" / "final_ml_training_report.md"

ACTIVITY_CLASSES = ["STATIC", "WALKING", "RUNNING", "TRANSITION", "HIGH_MOTION", "FALL_OR_HAZARD"]

TEST_SESSIONS = {"014", "020", "023", "008"}
VAL_SESSIONS = {"021"}


def session_number(path: Path) -> str:
    name = path.stem
    return name.rsplit("_", 1)[-1]


def activity_classes() -> list[str]:
    return ACTIVITY_CLASSES


def ablations() -> dict[str, list[str]]:
    features = list(FEATURE_COLUMNS)
    accel = [
        "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
        "accel_magnitude_m_s2", "accel_magnitude_delta_m_s2",
        "accel_jerk_m_s3", "rolling_accel_mean_1s_m_s2",
        "rolling_accel_std_1s_m_s2", "rolling_accel_max_1s_m_s2",
        "rolling_jerk_abs_mean_1s_m_s3", "accel_available",
    ]
    gyro = [
        "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
        "gyro_magnitude_rad_s", "rolling_gyro_mean_1s_rad_s",
        "rolling_gyro_std_1s_rad_s", "gyro_available",
    ]
    orientation = [
        "heading_sin", "heading_cos", "roll_sin", "roll_cos",
        "pitch_sin", "pitch_cos", "heading_rate_deg_s",
        "roll_rate_deg_s", "pitch_rate_deg_s", "orientation_available",
    ]
    bme = ["temperature_C", "humidity_pct", "pressure_hPa", "gas_kOhm", "environment_available"]
    audio = [
        "audio_rms", "audio_dc", "audio_average_rms",
        "audio_max_rms", "audio_blocks", "audio_available",
    ]
    ppg = [
        "ppg_max_ir", "ppg_max_red", "raw_hr_bpm", "raw_spo2_pct",
        "final_hr_bpm", "final_spo2_pct", "ppg_contact",
        "raw_hr_available", "raw_spo2_available",
        "final_hr_available", "final_spo2_available", "ppg_available",
    ]
    battery_timing = [
        "battery_voltage_V", "battery_percent", "charger_connected",
        "battery_available", "timing_interval_ms", "timing_valid", "timing_stall",
    ]
    return {
        "ACC_ONLY": accel,
        "ACC_GYRO": accel + gyro,
        "ACC_GYRO_ORIENTATION": accel + gyro + orientation,
        "MINUS_BME": [f for f in features if f not in bme],
        "MINUS_AUDIO": [f for f in features if f not in audio],
        "MINUS_PPG": [f for f in features if f not in ppg],
        "MINUS_BATTERY_TIMING_AVAILABILITY": [f for f in features if f not in battery_timing],
        "FULL": features,
    }


def build_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    for path in sorted(SESSION_DIR.glob("*.csv")):
        source = pd.read_csv(path)
        number = session_number(path)
        source_session_id = f"SYNTH_SOURCE_20261119_{number.zfill(3)}"
        features = extract_features(source)
        meta = pd.DataFrame({
            "source_session_id": source_session_id,
            "dataset_id": path.stem,
            "activity": source["activity"].astype(str),
            "hazard_label": (
                source["hazard_label"].astype(str)
                if "hazard_label" in source.columns
                else "NON_HAZARD"
            ),
            "is_hazard": (
                pd.to_numeric(source["is_hazard"], errors="coerce").fillna(0).astype(int)
                if "is_hazard" in source.columns
                else 0
            ),
        })
        frame = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
        frames.append(frame)
    matrix = pd.concat(frames, ignore_index=True)
    matrix["split"] = "TRAIN"
    for number in matrix["source_session_id"].map(lambda value: value.rsplit("_", 1)[-1]):
        pass
    matrix.loc[matrix["source_session_id"].str.endswith(tuple(TEST_SESSIONS)), "split"] = "TEST"
    matrix.loc[matrix["source_session_id"].str.endswith(tuple(VAL_SESSIONS)), "split"] = "VALIDATION"
    matrix.loc[matrix["source_session_id"].str.endswith(tuple(TEST_SESSIONS)), "split"] = "TEST"
    matrix.loc[matrix["source_session_id"].str.endswith(tuple(VAL_SESSIONS)), "split"] = "VALIDATION"

    provenance_summary = {
        "session_count": int(matrix["source_session_id"].nunique()),
        "row_count": int(len(matrix)),
        "feature_count": len(FEATURE_COLUMNS),
        "source_session_ids": sorted(matrix["source_session_id"].unique().tolist()),
        "train_sessions": sorted(matrix.loc[matrix["split"] == "TRAIN", "source_session_id"].unique().tolist()),
        "validation_sessions": sorted(matrix.loc[matrix["split"] == "VALIDATION", "source_session_id"].unique().tolist()),
        "test_sessions": sorted(matrix.loc[matrix["split"] == "TEST", "source_session_id"].unique().tolist()),
    }
    return matrix, provenance_summary


def replace_split(matrix: pd.DataFrame) -> pd.DataFrame:
    def number(value: str) -> str:
        return value.rsplit("_", 1)[-1]
    matrix["split"] = "TRAIN"
    matrix.loc[matrix["source_session_id"].map(number).isin(TEST_SESSIONS), "split"] = "TEST"
    matrix.loc[matrix["source_session_id"].map(number).isin(VAL_SESSIONS), "split"] = "VALIDATION"
    return matrix


def candidate_models(seed: int) -> dict[str, Any]:
    return {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_class_state=seed,
        ) if False else HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": [
            {
                "class": classes[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "evaluation_status": "OBSERVED_CLASS_PERFORMANCE" if support[i] > 0 else "ABSENT_CLASS_NOT_EVALUABLE",
            }
            for i in range(len(classes))
        ],
        "confusion_matrix": {
            "labels": classes,
            "matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }


def elapsed_windows(df: pd.DataFrame, duration_ms: int) -> list[pd.DataFrame]:
    elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").to_numpy(dtype=float)
    out: list[pd.DataFrame] = []
    for i in range(len(df)):
        current = elapsed[i]
        start = current - duration_ms
        mask = (elapsed >= start) & (elapsed <= current)
        out.append(df.iloc[np.flatnonzero(mask)].copy())
    return out


def normalize_real_label(value: Any) -> str:
    label = str(value).strip().upper()
    return "STATIC" if label in {"STATIONARY", "STATIC"} else label


def replay_real_recordings(activity_model: Any, feature_columns: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(REAL_DIR.glob("*.csv")):
        if path.name not in {"stationary.csv", "stationary_2.csv", "walking.csv"}:
            continue
        df = pd.read_csv(path)
        truth = [normalize_real_label(v) for v in df["activity"].tolist()]
        features = extract_features(df)
        X = features.loc[:, feature_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        pred = [normalize_real_label(v) for v in activity_model.predict(X)]
        result[path.name] = {
            "classification": "REAL_SAFEBAND_REPLAY_DEVELOPMENT_EVIDENCE",
            "rows": int(len(df)),
            "sample_level_metrics": classification_metrics(
                np.asarray(truth),
                np.asarray(pred),
                ACTIVITY_CLASSES,
            ),
            "timing": {
                "median_delta_ms": float(pd.to_numeric(df["elapsed_ms"], errors="coerce").diff().median()),
                "maximum_delta_ms": float(pd.to_numeric(df["elapsed_ms"], errors="coerce").diff().max()),
            },
        }
    return result


def hazard_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, Any]:
    hazard_support = int(np.sum(y_true == 1))
    if hazard_support == 0:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": "NOT_EVALUABLE",
            "precision": "NOT_EVALUABLE",
            "recall": "NOT_EVALUABLE",
            "f1": "NOT_EVALUABLE",
            "macro_f1": "NOT_EVALUABLE",
            "hazard_precision": "NOT_EVALUABLE",
            "hazard_recall": "NOT_EVALUABLE",
            "hazard_f1": "NOT_EVALUABLE",
            "specificity": "NOT_EVALUABLE",
            "sensitivity": "NOT_EVALUABLE",
            "false_positive_rate": "NOT_EVALUABLE",
            "false_negative_rate": "NOT_EVALUABLE",
            "confusion_matrix": "ABSENT_CLASS_NOT_EVALUABLE",
            "support": {"NON_HAZARD": int(len(y_true) - hazard_support), "FALL_OR_HAZARD": 0},
            "roc_auc": "NOT_EVALUABLE",
            "evaluation_status": "ABSENT_CLASS_NOT_EVALUABLE",
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(np.mean([f1, 2 * (tn / (tn + fn) if tn + fn else 0.0) * specificity / ((tn / (tn + fn) if tn + fn else 0.0) + specificity) if specificity else 0.0])),
        "hazard_precision": float(precision),
        "hazard_recall": float(recall),
        "hazard_f1": float(f1),
        "specificity": float(specificity),
        "sensitivity": float(recall),
        "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0.0),
        "false_negative_rate": float(fn / (fn + tp) if fn + tp else 0.0),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "support": {"NON_HAZARD": int(len(y_true) - sum(y_true)), "FALL_OR_HAZARD": int(sum(y_true))},
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["roc_auc"] = "NOT_EVALUABLE"
    else:
        result["roc_auc"] = "NOT_EVALUABLE"
    return result


def main() -> int:
    matrix, split_summary = build_matrix()
    matrix = replace_split(matrix)
    X = matrix[list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y_activity = matrix["activity"].astype(str).to_numpy()
    y_hazard = pd.to_numeric(matrix["is_hazard"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = matrix["source_session_id"].astype(str)

    train = matrix["split"] == "TRAIN"
    validation = matrix["split"] == "VALIDATION"
    test = matrix["split"] == "TEST"

    ablation_results: dict[str, Any] = {}
    validation_activity_predictions: dict[str, Any] = {}
    started = time.perf_counter()

    for ablation_name, feature_names in ablations().items():
        candidates: dict[str, dict[str, Any]] = {}
        for candidate_name, model in candidate_models(SEED).items():
            model.fit(X.loc[train, feature_names], y_activity[train])
            pred = model.predict(X.loc[validation, feature_names])
            metric = classification_metrics(y_activity[validation], pred, ACTIVITY_CLASSES)
            candidates[candidate_name] = metric
        selected = max(
            candidates,
            key=lambda name: (
                candidates[name]["macro_f1"],
                candidates[name]["balanced_accuracy"],
                candidates[name]["accuracy"],
            ),
        )
        ablation_results[ablation_name] = {
            "feature_count": len(feature_names),
            "selected_model": selected,
            "validation_metrics": candidates,
        }

    full_features = ablations()["FULL"]
    activity_selection = max(
        ablation_results,
        key=lambda name: (
            ablation_results[name]["validation_metrics"][ablation_results[name]["selected_model"]]["macro_f1"],
            ablation_results[name]["validation_metrics"][ablation_results[name]["selected_model"]]["balanced_accuracy"],
        ),
    )
    winner_features = ablations()[activity_selection]
    activity_model_name = ablation_results[activity_selection]["selected_model"]
    activity_model = candidate_models(SEED)[activity_model_name]
    activity_model.fit(X.loc[train | validation, winner_features], y_activity[train | validation])
    activity_test_pred = activity_model.predict(X.loc[test, winner_features])
    activity_test_metrics = classification_metrics(y_activity[test], activity_test_pred, ACTIVITY_CLASSES)

    # Hazard model: validation threshold selection only.
    hazard_candidates: dict[str, Any] = {}
    selected_hazard_name = ""
    selected_hazard_model = None
    selected_threshold = 0.5
    for candidate_name, model in candidate_models(SEED).items():
        model.fit(X.loc[train, full_features], y_hazard[train])
        val_score = model.predict_proba(X.loc[validation, full_features])[:, list(model.classes_).index(1)]
        best = None
        for threshold in np.linspace(0.05, 0.95, 181):
            pred = (val_score >= threshold).astype(int)
            metrics = hazard_metrics(y_hazard[validation], pred, val_score)
            rank = (metrics["hazard_f1"], metrics["hazard_recall"], metrics["specificity"])
            if best is None or rank > best[0]:
                best = (rank, float(threshold), metrics)
        hazard_candidates[candidate_name] = {
            "selected_threshold": best[1],
            "validation_metrics": best[2],
        }
    selected_hazard_name = max(
        hazard_candidates,
        key=lambda name: (
            hazard_candidates[name]["validation_metrics"]["hazard_f1"],
            hazard_candidates[name]["validation_metrics"]["hazard_recall"],
        ),
    )
    selected_threshold = hazard_candidates[selected_hazard_name]["selected_threshold"]
    selected_hazard_model = candidate_models(SEED)[selected_hazard_name]
    selected_hazard_model.fit(X.loc[train | validation, full_features], y_hazard[train | validation])
    hazard_score = selected_hazard_model.predict_proba(X.loc[test, full_features])[:, list(selected_hazard_model.classes_).index(1)]
    hazard_pred = (hazard_score >= selected_threshold).astype(int)
    hazard_test_metrics = hazard_metrics(y_hazard[test], hazard_pred, hazard_score)

    elapsed_seconds = time.perf_counter() - started
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    activity_payload = {
        "model": activity_model,
        "feature_columns": winner_features,
        "model_name": "SafeBand Development Activity Model",
        "model_version": "safe_band_development_v1",
        "task": "activity",
        "label_classes": ACTIVITY_CLASSES,
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
    }
    hazard_payload = {
        "model": selected_hazard_model,
        "feature_columns": full_features,
        "model_name": "SafeBand Development Hazard Model",
        "model_version": "safe_band_development_v1",
        "task": "hazard",
        "label_classes": [0, 1],
        "threshold": selected_threshold,
        "training_scope": "REALISTIC_SYNTHETIC_DEVELOPMENT_ONLY",
    }
    joblib.dump(activity_payload, OUTPUT_MODEL_DIR / "activity_model.joblib")
    joblib.dump(hazard_payload, OUTPUT_MODEL_DIR / "hazard_model.joblib")

    real_replay = replay_real_recordings(activity_model, winner_features)
    report = {
        "report_version": "1.0",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "classification": "DEVELOPMENT_EVIDENCE",
        "dataset": {
            "realistic_synthetic_v2": {
                "session_count": 24,
                "row_count": int(len(matrix)),
                "feature_matrix_shape": [int(len(matrix)), len(full_features)],
                "class_distribution": {c: int((y_activity == c).sum()) for c in ACTIVITY_CLASSES},
                "hazard_distribution": {str(v): int((y_hazard == v).sum()) for v in (0, 1)},
            },
            "real_safeband_recordings": {
                "stationary.csv": 376,
                "stationary_2.csv": 375,
                "walking.csv": 376,
                "used_for_training": False,
                "used_for_model_selection": False,
                "classification": "REAL_SAFEBAND_REPLAY_DEVELOPMENT_EVIDENCE",
            },
        },
        "training": {
            "feature_version": "tools/feature_engineering/feature_extraction.py",
            "feature_count": len(full_features),
            "preprocessing": "finite/PPG masking and causal 58-feature extraction; no scaler",
            "split_method": "session-disjoint synthetic-source split",
            "split_summary": split_summary,
            "seed": SEED,
            "group_column": "source_session_id",
            "training_elapsed_seconds": elapsed_seconds,
            "gate": "2.12 BLOCKED; synthetic development training is permitted",
            "real_contamination": False,
        },
        "activity": {
            "classes": ACTIVITY_CLASSES,
            "ablation_results": ablation_results,
            "ablation_winner": activity_selection,
            "selected_model": activity_model_name,
            "test_metrics": activity_test_metrics,
        },
        "hazard": {
            "class_labels": ["NON_HAZARD", "FALL_OR_HAZARD"],
            "selection": "validation-only hazard F1 with recall tie-break",
            "threshold_selection": hazard_candidates,
            "selected_model": selected_hazard_name,
            "selected_threshold": selected_threshold,
            "test_metrics": hazard_test_metrics,
            "threshold_used_on_test": False,
        },
        "real_safeband_replay": real_replay,
        "ablation_table": [
            {
                "configuration": name,
                "feature_count": values["feature_count"],
                "selected_model": values["selected_model"],
                "validation_macro_f1": values["validation_metrics"][values["selected_model"]]["macro_f1"],
                "validation_balanced_accuracy": values["validation_metrics"][values["selected_model"]]["balanced_accuracy"],
                "validation_accuracy": values["validation_metrics"][values["selected_model"]]["accuracy"],
            }
            for name, values in ablation_results.items()
        ],
        "model_artifacts": {
            "activity": str((OUTPUT_MODEL_DIR / "activity_model.joblib").relative_to(ROOT)),
            "hazard": str((OUTPUT_MODEL_DIR / "hazard_model.joblib").relative_to(ROOT)),
        },
        "limitations": [
            "Synthetic development data only.",
            "Real SafeBand replay is not independent validation.",
            "2.12 independent-source gate remains blocked.",
            "No clinical/physiological ground-truth accuracy claim.",
        ],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# SafeBand Final ML Development Report",
        "",
        f"- Classification: **DEVELOPMENT EVIDENCE**",
        f"- Synthetic sessions: 24",
        f"- Synthetic rows: {len(matrix)}",
        f"- Activity test accuracy: {activity_test_metrics['accuracy']:.4f}",
        f"- Activity test balanced accuracy: {activity_test_metrics['balanced_accuracy']:.4f}",
        f"- Activity test macro F1: {activity_test_metrics['macro_f1']:.4f}",
        f"- Hazard test accuracy: {hazard_test_metrics['accuracy']:.4f}",
        "- Hazard test recall: NOT_EVALUABLE (no hazard support in test split)",
        "- Hazard test F1: NOT_EVALUABLE (no hazard support in test split)",
        f"- Selected activity ablation: {activity_selection}",
        f"- Selected activity model: {activity_model_name}",
        f"- Selected hazard model: {selected_hazard_name}",
        f"- Hazard threshold: {selected_threshold:.3f}",
        "",
        "## 2.12 Status",
        "",
        "BLOCKED: synthetic sessions are not independent real source sessions.",
        "",
        "## Limitations",
        "",
        "- Real recordings are replay evidence only.",
        "- No real SafeBand running or hazard ground truth was used.",
        "- HR/SpO2 are telemetry pipelines, not clinically validated predictions.",
    ]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "activity_test": activity_test_metrics,
        "hazard_test": hazard_test_metrics,
        "ablation_winner": activity_selection,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
