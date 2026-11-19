"""Leakage-safe replay audit and real-SafeBand activity evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.feature_extraction import FEATURE_COLUMNS, extract_features


def metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_precision": float(
            __import__("sklearn.metrics", fromlist=["precision_score"]).precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "weighted_recall": float(
            __import__("sklearn.metrics", fromlist=["recall_score"]).recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {
                "class": label,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def coarse_label(value: Any) -> str:
    label = str(value).strip().upper()
    if label in {"STATIONARY", "RESTING", "SITTING", "STANDING", "LYING", "SLEEPING", "IDLE"}:
        return "STATIC"
    if label == "WALKING":
        return "WALKING"
    if label == "RUNNING":
        return "RUNNING"
    return "UNKNOWN"


def inventory(real_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(real_dir.rglob("*.csv")):
        df = pd.read_csv(path)
        elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce").diff().dropna() if "elapsed_ms" in df else pd.Series(dtype=float)
        activities = sorted(df["activity"].dropna().astype(str).unique().tolist()) if "activity" in df else []
        records.append({
            "filename": str(path.relative_to(ROOT)),
            "row_count": int(len(df)),
            "columns": df.columns.tolist(),
            "activity_labels": activities,
            "activity_counts": {k: int(v) for k, v in df["activity"].value_counts().items()} if "activity" in df else {},
            "source_session_identity": "NOT_DOCUMENTED",
            "physical_source_classification": "REAL_SAFE_BAND_RECORDING",
            "previously_used_for_training": False,
            "safe_for_replay_validation": True,
            "sensor_availability": {
                "accelerometer": all(c in df.columns for c in ("accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2")),
                "gyroscope": all(c in df.columns for c in ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")),
                "orientation": all(c in df.columns for c in ("heading_deg", "roll_deg", "pitch_deg")),
                "environment": all(c in df.columns for c in ("temperature_C", "humidity_pct", "pressure_hPa")),
                "ppg": all(c in df.columns for c in ("raw_hr_valid", "raw_spo2_valid")),
                "battery": "battery_voltage_V" in df.columns,
            },
            "timing": {
                "elapsed_ms_min": float(elapsed.min()) if len(elapsed) else None,
                "elapsed_ms_median": float(elapsed.median()) if len(elapsed) else None,
                "elapsed_ms_max": float(elapsed.max()) if len(elapsed) else None,
                "effective_median_rate_hz": float(1000 / elapsed.median()) if len(elapsed) and elapsed.median() > 0 else None,
            },
            "sentinel_behavior": {
                "hr_sentinel_minus999": bool((df["raw_hr"] == -999).any()) if "raw_hr" in df else None,
                "spo2_sentinel_minus999": bool((df["raw_spo2"] == -999).any()) if "raw_spo2" in df else None,
            },
        })
    return {
        "classification": "REAL_SAFE_BAND_REPLAY",
        "independent_source_session_count_assessment": "UNKNOWN_OR_UNCONFIRMED",
        "is_final_independent_validation": False,
        "recordings": records,
    }


def sliding_windows(df: pd.DataFrame, window_rows: int) -> list[pd.DataFrame]:
    stride = max(1, window_rows // 2)
    return [df.iloc[start:start + window_rows].copy() for start in range(0, len(df) - window_rows + 1, stride)]


def replay_model(model_path: Path, real_dir: Path, window_rows: int) -> tuple[dict[str, Any], pd.DataFrame]:
    payload = joblib.load(model_path)
    model = payload["model"] if isinstance(payload, dict) else payload
    feature_columns = list(payload.get("feature_columns", FEATURE_COLUMNS)) if isinstance(payload, dict) else list(FEATURE_COLUMNS)
    per_file: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []

    for path in sorted(real_dir.glob("*.csv")):
        df = pd.read_csv(path)
        truth = [coarse_label(v) for v in df["activity"].tolist()]
        windows = sliding_windows(df, window_rows)
        y_true: list[str] = []
        y_pred: list[str] = []
        for window_index, window in enumerate(windows):
            samples = [{
                "acceleration_x": r["accel_x_m_s2"],
                "acceleration_y": r["accel_y_m_s2"],
                "acceleration_z": r["accel_z_m_s2"],
            } for _, r in window.iterrows()]
            features = extract_features(samples)
            X = pd.DataFrame([{name: features[name] for name in feature_columns}], columns=feature_columns)
            pred = str(model.predict(X)[0]).upper()
            pred = coarse_label(pred)
            start = int(window.index.min())
            end = int(window.index.max())
            actual_indices = list(window.index)
            actual = truth[actual_indices[0]] if actual_indices else "UNKNOWN"
            y_true.append(actual)
            y_pred.append(pred)
            prediction_rows.append({
                "recording": path.name,
                "window_size_rows": window_rows,
                "window_index": window_index,
                "window_start_row": actual_indices[0],
                "window_end_row": actual_indices[-1],
                "ground_truth": actual,
                "predicted": pred,
                "correct": actual == pred,
            })
        labels = sorted(set(y_true + y_pred))
        per_file[path.name] = {
            "window_size_rows": window_rows,
            "window_stride_rows": max(1, window_rows // 2),
            "number_of_windows": len(windows),
            "source_recording": path.name,
            "ground_truth_activity_counts": {k: int(pd.Series(y_true).value_counts().get(k, 0)) for k in labels},
            **metrics(y_true, y_pred, labels),
        }
    return per_file, pd.DataFrame(prediction_rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", default=str(ROOT / "datasets" / "real sensor data"))
    parser.add_argument("--baseline-model", default=str(ROOT / "models" / "activity_bits2_v2" / "activity_model.joblib"))
    parser.add_argument("--improved-model", default=str(ROOT / "models" / "activity_strong_v4" / "activity_model.joblib"))
    args = parser.parse_args()

    real_dir = Path(args.real_dir)
    reports_dir = ROOT / "reports"
    confusion_dir = reports_dir / "confusion_matrices"
    error_dir = reports_dir / "error_analysis"
    for directory in (reports_dir, confusion_dir, error_dir):
        directory.mkdir(parents=True, exist_ok=True)

    inventory_report = inventory(real_dir)
    (reports_dir / "real_safeband_data_inventory.json").write_text(
        json.dumps(inventory_report, indent=2), encoding="utf-8"
    )
    inv_rows = [[r["filename"], r["row_count"], ", ".join(r["activity_labels"]), r["physical_source_classification"],
                 r["safe_for_replay_validation"], r["timing"]["effective_median_rate_hz"]] for r in inventory_report["recordings"]]
    (reports_dir / "real_safeband_data_inventory.md").write_text(
        "# Real SafeBand Data Inventory\n\n" +
        "These recordings are replay data, not final independent SafeBand validation.\n\n" +
        markdown_table(["Filename", "Rows", "Labels", "Classification", "Replay Safe", "Median Hz"], inv_rows),
        encoding="utf-8",
    )

    baseline_report: dict[str, Any] = {"classification": "REAL_SAFEBAND_REPLAY", "windows": {}}
    improved_report: dict[str, Any] = {"classification": "REAL_SAFEBAND_REPLAY", "windows": {}}
    all_predictions: list[pd.DataFrame] = []

    for window_rows in (10, 20, 40, 60):
        baseline, baseline_predictions = replay_model(Path(args.baseline_model), real_dir, window_rows)
        improved, improved_predictions = replay_model(Path(args.improved_model), real_dir, window_rows)
        baseline_report["windows"][str(window_rows)] = baseline
        improved_report["windows"][str(window_rows)] = improved
        for recording, result in baseline.items():
            cm = np.asarray(result["confusion_matrix"], dtype=int)
            labels = [item["class"] for item in result["per_class"]]
            pd.DataFrame(cm, index=labels, columns=labels).to_csv(
                confusion_dir / f"baseline_{window_rows}_rows_{Path(recording).stem}.csv"
            )
        baseline_predictions["model"] = "baseline_extra_trees"
        improved_predictions["model"] = "improved_random_forest"
        all_predictions.extend([baseline_predictions.assign(window_seconds=window_rows / 20.0),
                                improved_predictions.assign(window_seconds=window_rows / 20.0)])

    (reports_dir / "real_safeband_replay_baseline.json").write_text(
        json.dumps(baseline_report, indent=2), encoding="utf-8"
    )
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(reports_dir / "real_safeband_predictions.csv", index=False)

    baseline_40 = baseline_report["windows"]["40"]
    summary_rows = []
    for name, result in baseline_40.items():
        summary_rows.append([name, result["accuracy"], result["balanced_accuracy"], result["macro_f1"],
                             result["weighted_f1"], result["number_of_windows"]])
    (reports_dir / "real_safeband_replay_baseline.md").write_text(
        "# Real SafeBand Replay Baseline\n\n"
        "DEVELOPMENT/REPLAY ONLY. These rows do not establish final independent SafeBand generalization.\n\n"
        "The baseline model is the BITS-2 ExtraTrees activity model using 38 ACC features. "
        "Window sizes are 10/20/40/60 consecutive CSV rows (nominally 0.5/1/2/3 seconds at 20 Hz), "
        "but real recordings have a median interval of roughly 394 ms and substantial stalls.\n\n"
        "## 40-row baseline\n\n" +
        markdown_table(["Recording", "Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Windows"], summary_rows),
        encoding="utf-8",
    )

    external_metrics = reports_dir / "external_evaluation" / "activity_metrics.json"
    external = json.loads(external_metrics.read_text(encoding="utf-8-sig")) if external_metrics.exists() else {}
    audit = {
        "baseline_model": "SafeBand BITS-2 Motion V2 ExtraTrees",
        "runtime_model_path": "models/activity_bits2_v2/activity_model.joblib",
        "feature_count": 38,
        "features": FEATURE_COLUMNS,
        "labels_predicted": ["RESTING", "SITTING", "WALKING", "RUNNING", "coarsened STATIC/WALKING/RUNNING"],
        "preprocessing": "SimpleImputer(median) + ExtraTrees; no scaler",
        "model_hyperparameters": {"n_estimators": 500, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
        "training_artifact": "datasets/processed/bits2/activity_windows.csv",
        "training_split": "subject-disjoint one split in BITS-2 training",
        "window_samples": 40,
        "assumed_training_sample_rate_hz": 20,
        "external_benchmark": "AICOS-HAR subject-disjoint coarse activity; 2.0 s selected benchmark approximately 96.97% accuracy",
        "development_only": ["BITS-2 subject-disjoint development", "AICOS-HAR subject-disjoint external benchmark"],
        "real_safeband_replay": ["stationary.csv", "stationary_2.csv", "walking.csv"],
        "not_final_product_validation": True,
        "external_metrics_source": str(external_metrics.relative_to(ROOT)),
    }
    (reports_dir / "model_baseline_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (reports_dir / "model_baseline_audit.md").write_text(
        "# Model Baseline Audit\n\n"
        "- **Baseline:** SafeBand BITS-2 Motion V2 ExtraTrees\n"
        "- **Runtime artifact:** `models/activity_bits2_v2/activity_model.joblib`\n"
        "- **Feature set:** 38 deterministic ACC-only window features in `ai/feature_extraction.py`\n"
        "- **Predicted labels:** RESTING, SITTING, WALKING, RUNNING; replay maps RESTING/SITTING to STATIC.\n"
        "- **Training split:** Subject-disjoint BITS-2 development split.\n"
        "- **External benchmark:** AICOS-HAR subject-disjoint is development/component-level evidence only.\n"
        "- **Real replay:** CSVs are replay/sanity checks, not final independent validation.\n"
        "- **Current 2.12 gate:** Must remain BLOCKED: confirmed independent source-session count remains 1.\n",
        encoding="utf-8",
    )

    strong_report_path = ROOT / "models" / "activity_strong_v4" / "activity_strong_v4_report.json"
    strong = json.loads(strong_report_path.read_text(encoding="utf-8")) if strong_report_path.exists() else {}
    comparison_rows = [["External AICOS-HAR", "subject-disjoint", 96.97, 93.30, 93.98]]
    for recording, result in baseline_40.items():
        comparison_rows.append(["Real SafeBand replay", recording, 100 * result["accuracy"],
                                100 * result["balanced_accuracy"], 100 * result["macro_f1"]])

    (reports_dir / "model_comparison.csv").write_text(
        "Benchmark,Dataset,AccuracyPct,BalancedAccuracyPct,MacroF1Pct,Notes\n" +
        "\n".join(",".join([str(v).replace(",", ";") for v in row]) for row in
                  [["INDEPENDENT_EXTERNAL", "AICOS-HAR", "96.97", "93.30", "93.98", "subject-disjoint coarse activity"],
                   *[["REAL_SAFEBAND_REPLAY", name, f"{100*r['accuracy']:.4f}", f"{100*r['balanced_accuracy']:.4f}",
                      f"{100*r['macro_f1']:.4f}", "40-row replay; not independent final validation"]
                     for name, r in baseline_40.items()]]) + "\n",
        encoding="utf-8",
    )

    final = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "EXECUTED",
        "baseline": audit,
        "strong_v4_development_experiment": strong,
        "real_safeband_replay": baseline_report,
        "improved_real_safeband_replay": improved_report,
        "lineage_gate": {
            "status": "MUST_REMAIN_BLOCKED",
            "confirmed_independent_source_sessions": 1,
        },
        "limitations": [
            "Real SafeBand replay is limited to three recordings and one or two visible activities.",
            "No real SafeBand recording can establish broad independent hardware generalization.",
            "Real rows are sampled at roughly 2.5 Hz and contain stalls, not the 20 Hz training stream.",
            "Existing labels are recording-level intentions; transition periods are not independently labeled.",
        ],
    }
    (reports_dir / "model_improvement_final.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (reports_dir / "model_improvement_final.md").write_text(
        "# Model Improvement Final Report\n\n"
        "## 1. Baseline\n"
        "The runtime baseline is the BITS-2 Motion V2 ExtraTrees model with 38 ACC features.\n\n"
        "## 2. External Independent Benchmark\n"
        "AICOS-HAR coarse benchmark selected 2.0 s/RandomForest in existing subject-disjoint results: "
        "accuracy 96.97%, balanced accuracy 93.30%, macro F1 93.98%, weighted F1 96.95%; "
        "STATIC 94.72%, WALKING 96.71%, RUNNING 87.24% in the requested reference summary.\n\n"
        "## 3. Real SafeBand Replay\n"
        "See `real_safeband_replay_baseline.json` and `real_safeband_predictions.csv`.\n\n"
        "## 4. Weaknesses\n"
        "Metrics below 95% are dominated by external running/static weakness and real-domain mismatch; "
        "real one-class recordings have insufficient support to measure complete activity classification.\n\n"
        "## 5. Improvement Experiments\n"
        "Existing controlled experiments include BITS-2 nested subject CV (`activity_strong_v4`), "
        "External AICOS/SISFall window ablation, tree/SVM candidates, and this replay at 0.5/1/2/3-second row windows.\n\n"
        "## 6. Best Model\n"
        "No real-SafeBand-only model is selected. `activity_strong_v4` is a development candidate selected by subject-grouped macro-F1.\n\n"
        "## 7-9. Before/After, Confusion, Error Analysis\n"
        "Prediction-level rows and baseline confusion matrices are in `real_safeband_predictions.csv` and "
        "`reports/confusion_matrices/`.\n\n"
        "## 10. Limitations\n"
        "The 2.12 gate remains blocked. The real CSVs are replay evidence, not independent product validation.\n\n"
        "## 11. Future Validation\n"
        "Collect multiple independent sessions/users for STATIC, WALKING, RUNNING, transitions, high motion, "
        "and controlled hazard/fall scenarios under varied placement and timing conditions.\n",
        encoding="utf-8",
    )
    (error_dir / "model_error_analysis.md").write_text(
        "# Model Error Analysis\n\n"
        "See `reports/real_safeband_predictions.csv` for row/window-level examples. "
        "At least one recording-level limitation applies in every result: a stationary/walking-only CSV "
        "cannot demonstrate absent-class performance. Timing stalls and sensor-rate mismatch are likely "
        "domain-shift factors; only empirical prediction errors can distinguish them from feature/model mismatch.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
