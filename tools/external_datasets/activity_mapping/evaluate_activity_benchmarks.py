from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.external_datasets.activity_mapping.build_activity_dataset import (  # noqa: E402
    MODEL_FEATURES,
)

SEED = 42
WINDOW_SIZES = (0.5, 1.0, 2.0, 3.0)
MAX_FIT_ROWS = 30_000
GROUP_SPLITS = 5


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {
                "class": classes[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(len(classes))
        ],
    }


def grouped_partitions(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = GROUP_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Need {n_splits} distinct subject groups; found {len(unique_groups)}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels, groups))
    test_indices = folds[0][1]
    validation_indices = folds[1][1]
    train_indices = np.setdiff1d(
        np.arange(len(labels)),
        np.concatenate((test_indices, validation_indices)),
        assume_unique=False,
    )
    train_groups = set(groups[train_indices])
    validation_groups = set(groups[validation_indices])
    test_groups = set(groups[test_indices])
    if (
        train_groups & validation_groups
        or train_groups & test_groups
        or validation_groups & test_groups
    ):
        raise AssertionError("Subject leakage detected across grouped partitions")
    return train_indices, validation_indices, test_indices


def _bounded_training_sample(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int = MAX_FIT_ROWS,
    seed: int = SEED,
) -> np.ndarray:
    if len(indices) <= max_rows:
        return indices
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[indices])
    per_class = max(1, max_rows // len(classes))
    selected: list[np.ndarray] = []
    for label in classes:
        candidates = indices[labels[indices] == label]
        take = min(len(candidates), per_class)
        selected.append(np.sort(rng.choice(candidates, size=take, replace=False)))
    result = np.concatenate(selected)
    if len(result) > max_rows:
        result = np.sort(rng.choice(result, size=max_rows, replace=False))
    return result


def _estimator_factories() -> dict[str, Any]:
    return {
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def _task_frames(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame, list[str], str]]:
    aicos = frame[frame["dataset_id"] == "aicos_har"].copy()
    sisfall = frame[frame["dataset_id"] == "sisfall"].copy()
    tasks: list[tuple[str, pd.DataFrame, list[str], str]] = []

    aicos_coarse = aicos[aicos["activity_safeband"].isin(["STATIC", "WALKING", "RUNNING"])]
    tasks.append((
        "coarse_activity_aicos_har",
        aicos_coarse,
        ["STATIC", "WALKING", "RUNNING"],
        "activity_safeband",
    ))

    aicos_binary = aicos[aicos["activity_safeband"].isin(["STATIC", "WALKING", "RUNNING"])]
    aicos_binary["task_label"] = np.where(
        aicos_binary["activity_safeband"].eq("STATIC"), "STATIC", "MOVING"
    )
    tasks.append(("binary_motion_aicos_har", aicos_binary, ["STATIC", "MOVING"], "task_label"))

    fine_source_labels = {"STANDING", "SITTING", "LYINGDOWN", "WALKING", "RUNNING"}
    aicos_fine = aicos[aicos["source_activity_detail"].isin(fine_source_labels)]
    tasks.append((
        "fine_grained_experiment_aicos_har",
        aicos_fine,
        ["STANDING", "SITTING", "LYINGDOWN", "WALKING", "RUNNING"],
        "source_activity_detail",
    ))

    sisfall_coarse = sisfall[
        sisfall["activity_safeband"].isin(["WALKING", "RUNNING", "TRANSITION", "HIGH_MOTION"])
    ]
    tasks.append((
        "coarse_activity_sisfall_adl",
        sisfall_coarse,
        ["WALKING", "RUNNING", "TRANSITION", "HIGH_MOTION"],
        "activity_safeband",
    ))

    sisfall_hazard = sisfall[sisfall["hazard_label"].isin([0, 1, "0", "1"])].copy()
    sisfall_hazard["task_label"] = np.where(
        pd.to_numeric(sisfall_hazard["hazard_label"]).eq(1), "STAGED_FALL", "NONFALL_ADL"
    )
    tasks.append((
        "separate_hazard_sisfall",
        sisfall_hazard,
        ["NONFALL_ADL", "STAGED_FALL"],
        "task_label",
    ))
    return tasks


def evaluate_task(
    frame: pd.DataFrame,
    classes: list[str],
    label_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    result_by_window: dict[str, Any] = {}
    confusion_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    best_window_key: str | None = None
    best_window_validation: tuple[float, float] | None = None
    best_window_model_name = ""
    best_window_importances: np.ndarray | None = None

    for window in WINDOW_SIZES:
        subset = frame[frame["window_seconds"].astype(float).eq(window)].copy()
        subset = subset[subset[label_column].isin(classes)].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No rows for {label_column} at {window:.1f}s")
        class_indices = {label: index for index, label in enumerate(classes)}
        labels = subset[label_column].map(class_indices).to_numpy(dtype=np.int64)
        group_values = (
            subset["dataset_id"].astype(str)
            + "::"
            + subset["subject_id"].astype(str)
        ).to_numpy()
        train, validation, test = grouped_partitions(labels, group_values)

        X = subset.loc[:, MODEL_FEATURES].to_numpy(dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError(f"Non-finite model features at window={window}")
        train_fit = _bounded_training_sample(train, labels)
        validation_eval = _bounded_training_sample(
            validation, labels, max_rows=20_000, seed=SEED + 1
        )
        candidates: list[tuple[float, float, str, Any, dict[str, Any]]] = []
        for model_name, factory in _estimator_factories().items():
            model = factory()
            started = time.perf_counter()
            model.fit(X[train_fit], labels[train_fit])
            fit_seconds = time.perf_counter() - started
            validation_prediction = model.predict(X[validation_eval])
            validation_metrics = _metrics(
                labels[validation_eval], validation_prediction, classes
            )
            candidates.append((
                validation_metrics["macro_f1"],
                validation_metrics["balanced_accuracy"],
                model_name,
                model,
                {
                    "metrics": validation_metrics,
                    "fit_seconds": fit_seconds,
                },
            ))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, selected_name, selected_model, selection_details = candidates[0]
        test_prediction = selected_model.predict(X[test])
        test_metrics = _metrics(labels[test], test_prediction, classes)
        matrix = confusion_matrix(labels[test], test_prediction, labels=np.arange(len(classes)))
        for actual_index, actual_label in enumerate(classes):
            for predicted_index, predicted_label in enumerate(classes):
                confusion_rows.append({
                    "task": "",
                    "window_seconds": window,
                    "selected_model": selected_name,
                    "actual": actual_label,
                    "predicted": predicted_label,
                    "count": int(matrix[actual_index, predicted_index]),
                })
        result_by_window[f"{window:.1f}"] = {
            "window_seconds": window,
            "window_rows": int(len(subset)),
            "subject_groups": int(len(np.unique(group_values))),
            "partition_groups": {
                "train": int(len(np.unique(group_values[train]))),
                "validation": int(len(np.unique(group_values[validation]))),
                "test": int(len(np.unique(group_values[test]))),
            },
            "partition_rows": {
                "train": int(len(train)),
                "train_fitted": int(len(train_fit)),
                "validation": int(len(validation)),
                "validation_evaluated": int(len(validation_eval)),
                "test": int(len(test)),
            },
            "model_selection": {
                "selected_model": selected_name,
                "validation": selection_details["metrics"],
                "fit_seconds": selection_details["fit_seconds"],
                "other_candidates": [
                    {
                        "model": item[2],
                        "validation": item[4]["metrics"],
                        "fit_seconds": item[4]["fit_seconds"],
                    }
                    for item in candidates[1:]
                ],
            },
            "test": test_metrics,
            "confusion_matrix": {
                "labels": classes,
                "matrix": matrix.tolist(),
            },
        }
        validation_key = (
            selection_details["metrics"]["macro_f1"],
            selection_details["metrics"]["balanced_accuracy"],
        )
        if best_window_validation is None or validation_key > best_window_validation:
            best_window_key = f"{window:.1f}"
            best_window_validation = validation_key
            best_window_model_name = selected_name
            best_window_importances = np.asarray(selected_model.feature_importances_)

    assert best_window_key is not None
    assert best_window_importances is not None
    ranked = np.argsort(best_window_importances)[::-1]
    for rank, feature_index in enumerate(ranked[:20], start=1):
        importance_rows.append({
            "task": "",
            "best_window_seconds": float(best_window_key),
            "selected_model": best_window_model_name,
            "rank": rank,
            "feature": MODEL_FEATURES[int(feature_index)],
            "importance": float(best_window_importances[int(feature_index)]),
        })
    return {
        "classes": classes,
        "label_column": label_column,
        "window_results": result_by_window,
        "best_window_by_validation": float(best_window_key),
        "best_model_at_best_window": best_window_model_name,
        "selection_rule": "Validation macro-F1, then validation balanced accuracy; test metrics are not used to select the window or model.",
    }, confusion_rows, importance_rows


def _write_markdown_report(
    path: Path,
    reports: dict[str, Any],
    frame: pd.DataFrame,
) -> None:
    lines = [
        "# SafeBand External Activity Benchmark",
        "",
        "**DEVELOPMENT / COMPONENT-LEVEL EVALUATION**",
        "",
        "This report is not SafeBand product validation. External source scores do not estimate target-device performance.",
        "",
        "## Processed data",
        "",
        f"- Window rows: {len(frame):,}",
        f"- Physical sources processed: {', '.join(sorted(frame['dataset_id'].unique()))}",
        f"- Subjects: {frame[['dataset_id', 'subject_id']].drop_duplicates().shape[0]:,} source-subject pairs",
        f"- Recording/session IDs: {frame[['dataset_id', 'session_id']].drop_duplicates().shape[0]:,}",
        f"- Model features: {len(MODEL_FEATURES)}",
        f"- Window lengths: {', '.join(f'{value:.1f}s' for value in WINDOW_SIZES)}",
        "- Grouping: `dataset_id + subject_id`; no subject group crosses train, validation, and test.",
        "- Model inputs: acceleration-derived, rotation-invariant features plus timing/availability. Dataset, subject, session, labels, and provenance are metadata only.",
        "",
        "### Normalized classes and counts",
        "",
        "| Dataset | Original label | SafeBand class | Windows at 1.0 s |",
        "|---|---|---|---:|",
    ]
    one_second = frame[frame["window_seconds"].astype(float).eq(1.0)]
    label_counts = (
        one_second.groupby(
            ["dataset_id", "activity_original", "activity_safeband"],
            dropna=False,
        ).size().reset_index(name="count")
    )
    for row in label_counts.itertuples(index=False):
        lines.append(
            f"| {row.dataset_id} | {row.activity_original} | "
            f"{row.activity_safeband} | {int(row.count):,} |"
        )
    lines.extend(["", "## Benchmark metrics", ""])
    for task_name, task in reports["benchmarks"].items():
        lines.extend([
            f"### {task_name}",
            "",
            f"Classes: {', '.join(task['classes'])}.",
            f"Best window selected on validation: **{task['best_window_by_validation']:.1f}s** "
            f"({task['best_model_at_best_window']}).",
            "",
            "| Window | Selected model | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for window_key, result in task["window_results"].items():
            metrics = result["test"]
            lines.append(
                f"| {float(window_key):.1f}s | {result['model_selection']['selected_model']} | "
                f"{metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | "
                f"{metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |"
            )
        best_key = f"{task['best_window_by_validation']:.1f}"
        best = task["window_results"][best_key]
        lines.extend([
            "",
            f"Per-class test metrics at validation-selected {best_key}s window:",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in best["test"]["per_class"]:
            lines.append(
                f"| {item['class']} | {item['precision']:.4f} | "
                f"{item['recall']:.4f} | {item['f1']:.4f} | {item['support']:,} |"
            )
        matrix = best["confusion_matrix"]
        lines.extend([
            "",
            "Confusion matrix at the validation-selected window (rows = actual, columns = predicted):",
            "",
            "| Actual / predicted | " + " | ".join(matrix["labels"]) + " |",
            "|---|" + "---:|" * len(matrix["labels"]),
        ])
        for label, values in zip(matrix["labels"], matrix["matrix"]):
            lines.append(f"| {label} | " + " | ".join(str(value) for value in values) + " |")
        lines.append("")

    lines.extend([
        "## Dataset compatibility decisions",
        "",
        "| Dataset | Decision | Reason |",
        "|---|---|---|",
    ])
    compat = pd.read_csv(
        REPO_ROOT / "reports" / "external_evaluation" / "activity_dataset_compatibility.csv"
    )
    for row in compat.itertuples(index=False):
        lines.append(
            f"| {row.dataset_id} | {row.decision} | {row.compatibility_notes} |"
        )

    lines.extend([
        "",
        "## Feature importance",
        "",
        "Top features for each benchmark's validation-selected model/window are in "
        "`activity_feature_importance.csv` (up to 20 per task).",
        "",
        "| Task | Rank | Feature | Importance |",
        "|---|---:|---|---:|",
    ])
    importance = pd.read_csv(
        REPO_ROOT / "reports" / "external_evaluation" / "activity_feature_importance.csv"
    )
    for row in importance.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.rank} | {row.feature} | {row.importance:.6f} |"
        )
    lines.extend([
        "",
        "## Limitations and lineage",
        "",
        "- Fine-grained standing/sitting/lying classes are a secondary experiment; their scores are not a production taxonomy.",
        "- SisFall ADL labels describe scripted multi-action activities. Its fall records are staged scenarios, not real-world fall outcomes.",
        "- AICOS-HAR and SisFall are separate source domains and are not pooled for a cross-dataset accuracy claim.",
        "- BITS-2, FORTH-TRACE, and PAMAP2 are rejected from normalized training because the local evidence does not establish all needed units/time semantics.",
        "- SafeBand recordings remain descriptive anchors only; their two stationary file session identities are unverified.",
        "- The 2.12 SafeBand gate remains BLOCKED at one confirmed independent source session. No external dataset was added to that count.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grouped external activity development benchmarks.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "external_processed" / "activity" / "activity_windows.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "external_evaluation",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Processed external activity dataset not found: {args.input}")
    frame = pd.read_csv(args.input, compression="gzip")
    required = {
        *MODEL_FEATURES, "dataset_id", "subject_id", "session_id", "window_seconds",
        "activity_safeband", "source_activity_detail", "hazard_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset missing required columns: {sorted(missing)}")

    reports: dict[str, Any] = {
        "evaluation_scope": "DEVELOPMENT / COMPONENT-LEVEL EVALUATION",
        "input": str(args.input),
        "source_rows": int(len(frame)),
        "model_features": list(MODEL_FEATURES),
        "metadata_excluded_from_features": [
            "dataset_id", "subject_id", "session_id", "source_path",
            "activity_original", "source_activity_detail", "activity_safeband",
            "mapping_status", "hazard_label",
        ],
        "group_split": (
            "Five-fold StratifiedGroupKFold, using two disjoint folds for validation/test "
            "and remaining subject groups for train. Group key is dataset_id + subject_id."
        ),
        "fit_row_cap": MAX_FIT_ROWS,
        "benchmarks": {},
        "limitations": [
            "AICOS-HAR and SisFall are evaluated in separate cohorts; scores are not cross-dataset or product-performance estimates.",
            "Static subtypes are source labels in a secondary diagnostic, not primary SafeBand classes.",
            "SisFall fall labels are staged scenarios; its hazard score is not real-world fall incidence performance.",
            "The activity model uses acceleration-derived features only; external gyro streams are not fused.",
        ],
    }
    confusion_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for task_name, task_frame, classes, label_column in _task_frames(frame):
        if task_frame.empty:
            continue
        task_report, matrices, importances = evaluate_task(
            task_frame, classes, label_column
        )
        reports["benchmarks"][task_name] = task_report
        for row in matrices:
            row["task"] = task_name
            confusion_rows.append(row)
        for row in importances:
            row["task"] = task_name
            importance_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "activity_metrics.json"
    metrics_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_rows).to_csv(
        args.output_dir / "activity_confusion_matrix.csv", index=False
    )
    pd.DataFrame(importance_rows).to_csv(
        args.output_dir / "activity_feature_importance.csv", index=False
    )
    _write_markdown_report(
        args.output_dir / "activity_benchmark_report.md",
        reports,
        frame,
    )
    print(json.dumps({
        "metrics": str(metrics_path),
        "benchmarks": {
            name: {
                "best_window": task["best_window_by_validation"],
                "best_model": task["best_model_at_best_window"],
                "test_metrics": task["window_results"][
                    f"{task['best_window_by_validation']:.1f}"
                ]["test"],
            }
            for name, task in reports["benchmarks"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
