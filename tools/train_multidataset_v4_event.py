#!/usr/bin/env python3
"""SafeBand Multi-Dataset V4: temporal fall-event benchmark.

This benchmark starts from the frozen V3 window-level feature contract and
adds a temporal event layer. It does NOT tune on the held-out test subjects.

Protocol
--------
1. Reproduce the V2/V3 subject split exactly (seed-controlled).
2. Train the selected V3 window model on TRAIN subjects only.
3. Generate validation fall probabilities.
4. Convert validation probabilities to recording-level fall events.
5. Select temporal parameters using validation only.
6. Refit the same window model on TRAIN + VALIDATION subjects.
7. Evaluate the selected event operating point once on held-out TEST.

The default model is the V3-selected Random Forest and the feature contract
is exactly the 84 V6 features used by V1-V3.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES = [
    "ax_mean", "ax_std", "ax_min", "ax_max", "ax_rms",
    "ay_mean", "ay_std", "ay_min", "ay_max", "ay_rms",
    "az_mean", "az_std", "az_min", "az_max", "az_rms",
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_rms",
    "mag_p10", "mag_p25", "mag_p50", "mag_p75", "mag_p90",
    "mag_sma", "mag_range", "jerk_mean", "jerk_std", "jerk_max",
    "diff_energy", "mag_zero_crossings", "xy_corr", "xz_corr", "yz_corr",
    "fft_low", "fft_mid", "fft_high",
    "pre_peak_mean", "peak_magnitude", "post_peak_mean", "peak_to_baseline",
    "peak_index_ratio", "pre_post_change", "post_peak_std", "peak_width",
    "high_energy_fraction", "recovery_ratio",
    "v6_mag_mean", "v6_mag_std", "v6_mag_min", "v6_mag_max", "v6_mag_rms",
    "v6_mag_p10", "v6_mag_p25", "v6_mag_p50", "v6_mag_p75", "v6_mag_p90",
    "v6_peak_prominence", "v6_peak_to_rms", "v6_peak_to_median",
    "v6_peak_index_ratio", "v6_peak_width_half", "v6_high_fraction",
    "v6_pre_mean", "v6_post_mean", "v6_pre_std", "v6_post_std",
    "v6_pre_post_ratio", "v6_pre_post_delta", "v6_post_quiet_ratio",
    "v6_pre_energy", "v6_post_energy", "v6_post_energy_ratio",
    "v6_jerk_abs_mean", "v6_jerk_abs_max", "v6_jerk_std",
    "v6_jerk_sign_changes", "v6_diff_energy", "v6_impulse_area",
    "v6_axis_peak_ratio", "v6_axis_std_ratio", "v6_tilt_change",
    "v6_spectral_entropy",
]
assert len(FEATURES) == 84


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bits2", required=True)
    p.add_argument("--sisfall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", choices=["rf", "et", "hgb", "logistic"], default="rf")
    p.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--min-hits", default="1,2,3,4,5")
    p.add_argument("--max-gaps", default="30,60,90,120")
    return p.parse_args()


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def normalize_subject(value: object) -> str:
    text = str(value)
    match = re.search(r"(S[AE]\d+|USER\d+)", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else text


def normalize_label(series: pd.Series, dataset: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        result = pd.to_numeric(series, errors="coerce")
    else:
        result = series.astype(str).str.strip().str.upper().map(
            {
                "FALL": 1,
                "NON_FALL": 0,
                "NONFALL": 0,
                "NON-FALL": 0,
                "TRUE": 1,
                "FALSE": 0,
                "1": 1,
                "0": 0,
            }
        )
    if result.isna().any():
        bad = int(result.isna().sum())
        raise ValueError(f"{dataset}: {bad} labels could not be parsed")
    result = result.astype(int)
    if not result.isin([0, 1]).all():
        raise ValueError(f"{dataset}: labels must be binary 0/1")
    return result


def load_dataset(path: str, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    subject_col = find_col(df, ["subject_id", "subject", "user_id", "user"])
    label_col = find_col(df, ["is_fall", "target", "y", "label", "recording_type"])
    recording_col = find_col(df, ["recording_id", "source_file", "recording", "file", "source_file_name"])
    start_col = find_col(df, ["start_sample", "start", "window_start"])

    if subject_col is None or label_col is None or recording_col is None:
        raise ValueError(
            f"{dataset}: required subject/label/recording columns not found; "
            f"subject={subject_col!r}, label={label_col!r}, recording={recording_col!r}"
        )
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: missing {len(missing)} V6 features: {missing}")
    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"{dataset}: non-numeric V6 features: {non_numeric}")

    out = df.copy()
    out["__subject"] = out[subject_col].map(normalize_subject)
    out["__recording"] = out[recording_col].astype(str)
    out["__y"] = normalize_label(out[label_col], dataset)
    out["__label"] = out["__y"]
    out["__dataset"] = dataset

    if start_col is None:
        raise ValueError(
            f"{dataset}: a start_sample/window_start column is required for temporal event evaluation"
        )
    starts = pd.to_numeric(out[start_col], errors="coerce")
    if starts.isna().any():
        raise ValueError(f"{dataset}: invalid start-sample values in {start_col}")
    out["__start"] = starts.astype(int)
    return out


def split_subjects(df: pd.DataFrame, seed: int):
    """
    Subject-independent train/validation/test split.

    Preferred behavior:
      - Stratify subjects by whether they contain at least one fall
        recording, when both groups contain at least 3 subjects.

    Fallback behavior:
      - If one subject-level class has fewer than 3 subjects, perform
        a seeded random split over subjects.

    IMPORTANT:
      - Splitting is always at subject level.
      - No recording/window is split across train/validation/test.
    """
    if "__subject" not in df.columns:
        raise ValueError("Missing required column: __subject")

    subjects = (
        df["__subject"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(subjects) < 5:
        raise ValueError(
            f"Need at least 5 unique subjects for a 60/20/20 split, "
            f"found {len(subjects)}"
        )

    subject_has_fall = (
        df.groupby("__subject")["__label"]
        .max()
        .reindex(subjects)
        .fillna(0)
        .astype(int)
    )

    fall_subjects = [
        s for s in subjects
        if subject_has_fall.loc[s] == 1
    ]

    nonfall_subjects = [
        s for s in subjects
        if subject_has_fall.loc[s] == 0
    ]

    rng = np.random.RandomState(seed)

    def split_group(group):
        group = list(group)
        rng.shuffle(group)

        n = len(group)

        # Approximately 20% validation and 20% test.
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.20)))

        # Keep at least one subject for training.
        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test = group[:n_test]
        val = group[n_test:n_test + n_val]
        train = group[n_test + n_val:]

        if not train:
            raise ValueError(
                "Subject split produced an empty training set"
            )

        return train, val, test

    # Preferred stratified split.
    if len(fall_subjects) >= 3 and len(nonfall_subjects) >= 3:
        ft, fv, fte = split_group(fall_subjects)
        nt, nv, nte = split_group(nonfall_subjects)

        train_subjects = ft + nt
        val_subjects = fv + nv
        test_subjects = fte + nte

        rng.shuffle(train_subjects)
        rng.shuffle(val_subjects)
        rng.shuffle(test_subjects)

        split_mode = "stratified_subject"

    else:
        # Some datasets, notably BITS-2, have every subject represented
        # in the fall class. In that case stratification by subject-level
        # fall presence is mathematically impossible.
        shuffled = subjects.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, int(round(n * 0.20)))
        n_val = max(1, int(round(n * 0.20)))

        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test_subjects = shuffled[:n_test]
        val_subjects = shuffled[n_test:n_test + n_val]
        train_subjects = shuffled[n_test + n_val:]

        split_mode = "random_subject_fallback"

    if not train_subjects:
        raise ValueError("Training subject split is empty")
    if not val_subjects:
        raise ValueError("Validation subject split is empty")
    if not test_subjects:
        raise ValueError("Test subject split is empty")

    # Ensure every subject appears exactly once.
    all_split_subjects = (
        set(train_subjects)
        | set(val_subjects)
        | set(test_subjects)
    )

    if len(all_split_subjects) != len(subjects):
        raise ValueError(
            "Subject split does not cover all subjects"
        )

    if (
        set(train_subjects) & set(val_subjects)
        or set(train_subjects) & set(test_subjects)
        or set(val_subjects) & set(test_subjects)
    ):
        raise ValueError(
            "Subject leakage detected between train/validation/test"
        )

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
        "mode": split_mode,
    }


def make_model(kind: str, seed: int):
    if kind == "rf":
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if kind == "et":
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if kind == "hgb":
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if kind == "logistic":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
    raise ValueError(f"Unsupported model: {kind}")


def fall_probability(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = [str(c).upper() for c in model.classes_]
    if "FALL" in classes:
        index = classes.index("FALL")
    elif 1 in model.classes_:
        index = list(model.classes_).index(1)
    elif "1" in classes:
        index = classes.index("1")
    else:
        raise ValueError(f"Model classes do not contain FALL/1: {model.classes_}")
    return probabilities[:, index].astype(float)


def event_from_recording(
    starts: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    min_hits: int,
    max_gap: int,
) -> Tuple[bool, float, int]:
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    probabilities = probabilities[order]
    hit_indices = np.flatnonzero(probabilities >= threshold)
    if len(hit_indices) == 0:
        return False, 0.0, 0

    clusters: List[List[int]] = []
    current: List[int] = []
    for idx in hit_indices.tolist():
        if not current or starts[idx] - starts[current[-1]] <= max_gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    if current:
        clusters.append(current)

    confirmed = [cluster for cluster in clusters if len(cluster) >= min_hits]
    if not confirmed:
        return False, 0.0, 0
    peak = max(float(probabilities[cluster].max()) for cluster in confirmed)
    return True, peak, len(confirmed)


def recording_metrics(df: pd.DataFrame, probabilities: np.ndarray, threshold: float, min_hits: int, max_gap: int) -> Tuple[dict, pd.DataFrame]:
    work = df[["__dataset", "__subject", "__recording", "__y", "__start"]].copy()
    work["__p"] = probabilities
    rows: List[dict] = []
    for (dataset, subject, recording), group in work.groupby(
        ["__dataset", "__subject", "__recording"], sort=False
    ):
        prediction, peak, clusters = event_from_recording(
            group["__start"].to_numpy(dtype=int),
            group["__p"].to_numpy(dtype=float),
            threshold,
            min_hits,
            max_gap,
        )
        truth = int(group["__y"].iloc[0])
        if not (group["__y"] == truth).all():
            raise ValueError(f"Inconsistent label within recording {dataset}/{recording}")
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "recording": recording,
                "truth": truth,
                "prediction": int(prediction),
                "peak_probability": peak,
                "confirmed_clusters": clusters,
                "start_min": int(group["__start"].min()),
                "start_max": int(group["__start"].max()),
            }
        )

    events = pd.DataFrame(rows)
    y = events["truth"].to_numpy(dtype=int)
    pred = events["prediction"].to_numpy(dtype=int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold": float(threshold),
        "min_hits": int(min_hits),
        "max_gap_samples": int(max_gap),
        "recordings": int(len(events)),
        "fall_events": int(y.sum()),
        "nonfall_recordings": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_alarms": int(fp),
        "missed_falls": int(fn),
        "confusion_matrix": cm.tolist(),
    }
    return metrics, events


def dataset_event_metrics(events: pd.DataFrame) -> dict:
    output: Dict[str, dict] = {}
    for dataset, group in events.groupby("dataset", sort=False):
        y = group["truth"].to_numpy(dtype=int)
        pred = group["prediction"].to_numpy(dtype=int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        output[str(dataset)] = {
            "recordings": int(len(group)),
            "fall_events": int(y.sum()),
            "nonfall_recordings": int((y == 0).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "false_alarms": int(fp),
            "missed_falls": int(fn),
            "confusion_matrix": cm.tolist(),
        }
    if len(output) == 2:
        values = list(output.values())
        output["macro"] = {
            key: float(np.mean([v[key] for v in values]))
            for key in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "false_positive_rate"]
        }
    return output


def select_operating_point(candidates: pd.DataFrame) -> dict:
    """Select validation parameters without using test data.

    Priority:
      1. Prefer candidates with recall >= 90% on BOTH datasets.
      2. Among them, minimize macro FPR.
      3. Then maximize macro F1.
      4. If none meet 90% recall on both datasets, maximize the minimum
         dataset recall, then macro F1, then minimize macro FPR.

    This is intentionally more conservative than optimizing pooled F1.
    """
    required = {"BITS2", "SisFall"}
    eligible_rows = []
    for _, row in candidates.iterrows():
        if not required.issubset(row["datasets"]):
            continue
        b = row["datasets"]["BITS2"]
        s = row["datasets"]["SisFall"]
        eligible_rows.append(
            {
                "row": row,
                "min_recall": min(b["recall"], s["recall"]),
                "macro_recall": (b["recall"] + s["recall"]) / 2.0,
                "macro_f1": (b["f1"] + s["f1"]) / 2.0,
                "macro_fpr": (b["false_positive_rate"] + s["false_positive_rate"]) / 2.0,
            }
        )
    if not eligible_rows:
        raise ValueError("No operating-point candidates were generated")

    strong = [x for x in eligible_rows if x["min_recall"] >= 0.90]
    if strong:
        chosen = sorted(
            strong,
            key=lambda x: (x["macro_fpr"], -x["macro_f1"], -x["min_recall"]),
        )[0]
        selection_rule = "minimize macro FPR subject to >=90% recall on both datasets"
    else:
        chosen = sorted(
            eligible_rows,
            key=lambda x: (-x["min_recall"], -x["macro_f1"], x["macro_fpr"]),
        )[0]
        selection_rule = "maximize minimum dataset recall; then macro F1; then minimize macro FPR"

    result = chosen["row"].to_dict()
    result["selection_rule"] = selection_rule
    result["validation_min_dataset_recall"] = chosen["min_recall"]
    result["validation_macro_recall"] = chosen["macro_recall"]
    result["validation_macro_f1"] = chosen["macro_f1"]
    result["validation_macro_fpr"] = chosen["macro_fpr"]
    return result


def evaluate_operating_points(df: pd.DataFrame, probabilities: np.ndarray, thresholds: Iterable[float], min_hits_values: Iterable[int], max_gaps: Iterable[int]) -> pd.DataFrame:
    records: List[dict] = []
    for threshold in thresholds:
        for min_hits in min_hits_values:
            for max_gap in max_gaps:
                pooled, events = recording_metrics(df, probabilities, threshold, min_hits, max_gap)
                per_dataset = dataset_event_metrics(events)
                records.append(
                    {
                        "threshold": float(threshold),
                        "min_hits": int(min_hits),
                        "max_gap_samples": int(max_gap),
                        "pooled": pooled,
                        "datasets": per_dataset,
                    }
                )
    return pd.DataFrame(records)


def clean_float_list(text: str, name: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values or any(not np.isfinite(x) or not 0.0 <= x <= 1.0 for x in values):
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return values


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = clean_float_list(args.thresholds, "thresholds")
    min_hits_values = [int(x.strip()) for x in args.min_hits.split(",") if x.strip()]
    max_gaps = [int(x.strip()) for x in args.max_gaps.split(",") if x.strip()]
    if not min_hits_values or any(x < 1 for x in min_hits_values):
        raise ValueError("min-hits values must be >= 1")
    if not max_gaps or any(x < 0 for x in max_gaps):
        raise ValueError("max-gap values must be >= 0")

    bits2 = load_dataset(args.bits2, "BITS2")
    sisfall = load_dataset(args.sisfall, "SisFall")
    bs = split_subjects(bits2, args.seed)
    ss = split_subjects(sisfall, args.seed)

    btrain = bits2[bits2["__subject"].isin(bs["train"])].copy()
    bval = bits2[bits2["__subject"].isin(bs["val"])].copy()
    btest = bits2[bits2["__subject"].isin(bs["test"])].copy()
    strain = sisfall[sisfall["__subject"].isin(ss["train"])].copy()
    sval = sisfall[sisfall["__subject"].isin(ss["val"])].copy()
    stest = sisfall[sisfall["__subject"].isin(ss["test"])].copy()

    train = pd.concat([btrain, strain], ignore_index=True)
    val = pd.concat([bval, sval], ignore_index=True)
    test = pd.concat([btest, stest], ignore_index=True)

    print(f"[Preflight] BITS2 rows={len(bits2)} subjects={bits2['__subject'].nunique()}")
    print(f"[Preflight] SisFall rows={len(sisfall)} subjects={sisfall['__subject'].nunique()}")
    print(f"[Preflight] features={len(FEATURES)} window=60@20Hz step=30")
    print(f"[Preflight] train/val/test BITS2={len(btrain)}/{len(bval)}/{len(btest)}")
    print(f"[Preflight] train/val/test SisFall={len(strain)}/{len(sval)}/{len(stest)}")
    if bval.empty or sval.empty:
        raise ValueError("Subject split produced an empty validation set; use datasets with enough subjects")
    if btest.empty or stest.empty:
        raise ValueError("Subject split produced an empty test set; use datasets with enough subjects")

    model = make_model(args.model, args.seed)
    t0 = time.perf_counter()
    model.fit(train[FEATURES], train["__y"].to_numpy(dtype=int))
    val_probability = fall_probability(model, val[FEATURES])
    fit_seconds = time.perf_counter() - t0
    print(f"[Validation] window model fit={fit_seconds:.2f}s")

    candidates = evaluate_operating_points(
        val,
        val_probability,
        thresholds,
        min_hits_values,
        max_gaps,
    )
    selected = select_operating_point(candidates)
    print(
        "[Validation] selected "
        f"threshold={selected['threshold']:.3f} "
        f"min_hits={selected['min_hits']} "
        f"max_gap={selected['max_gap_samples']} "
        f"rule={selected['selection_rule']}"
    )
    for dataset in ["BITS2", "SisFall"]:
        m = selected["datasets"][dataset]
        print(
            f"  {dataset}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} F1={m['f1']:.4f} FPR={m['false_positive_rate']:.4f}"
        )

    final_train = pd.concat([train, val], ignore_index=True)
    final_model = make_model(args.model, args.seed)
    t0 = time.perf_counter()
    final_model.fit(final_train[FEATURES], final_train["__y"].to_numpy(dtype=int))
    final_fit_seconds = time.perf_counter() - t0
    test_probability = fall_probability(final_model, test[FEATURES])

    test_pooled, test_events = recording_metrics(
        test,
        test_probability,
        float(selected["threshold"]),
        int(selected["min_hits"]),
        int(selected["max_gap_samples"]),
    )
    test_datasets = dataset_event_metrics(test_events)

    report = {
        "version": "multidataset_v4_temporal_event",
        "seed": args.seed,
        "model": args.model,
        "schema": {
            "feature_count": 84,
            "window_contract": "60 samples @ 20Hz, step 30",
            "split_unit": "subject",
            "features": FEATURES,
        },
        "selection": {
            "validation_only": True,
            "selection_rule": selected["selection_rule"],
            "candidate_count": int(len(candidates)),
            "thresholds": thresholds,
            "min_hits": min_hits_values,
            "max_gaps": max_gaps,
        },
        "splits": {
            "BITS2": {
                "train_subjects": sorted(bs["train"]),
                "val_subjects": sorted(bs["val"]),
                "test_subjects": sorted(bs["test"]),
                "rows": {"train": len(btrain), "val": len(bval), "test": len(btest)},
            },
            "SisFall": {
                "train_subjects": sorted(ss["train"]),
                "val_subjects": sorted(ss["val"]),
                "test_subjects": sorted(ss["test"]),
                "rows": {"train": len(strain), "val": len(sval), "test": len(stest)},
            },
        },
        "validation_selected_operating_point": selected,
        "test": {
            "operating_point": {
                "threshold": float(selected["threshold"]),
                "min_hits": int(selected["min_hits"]),
                "max_gap_samples": int(selected["max_gap_samples"]),
            },
            "datasets": test_datasets,
            "pooled": test_pooled,
        },
        "timing": {
            "train_only_window_fit_seconds": fit_seconds,
            "final_train_validation_fit_seconds": final_fit_seconds,
        },
        "notes": [
            "Validation probabilities were generated by a model trained on TRAIN subjects only.",
            "The selected event operating point was never chosen using held-out TEST subjects.",
            "The final window model was refit on TRAIN+VALIDATION before the single TEST evaluation.",
            "Event metrics are recording-level; pooled window metrics are intentionally not the primary V4 metric.",
        ],
    }

    artifact = {
        "model": final_model,
        "feature_columns": FEATURES,
        "model_kind": args.model,
        "seed": args.seed,
        "window_contract": "60 samples @ 20Hz, step 30",
        "event_config": {
            "threshold": float(selected["threshold"]),
            "min_hits": int(selected["min_hits"]),
            "max_gap_samples": int(selected["max_gap_samples"]),
        },
    }
    joblib.dump(artifact, out / "multidataset_v4_event_model.joblib")
    (out / "multidataset_v4_event_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    test_events.to_csv(out / "multidataset_v4_test_recording_predictions.csv", index=False)

    print("\n=== FINAL V4 EVENT TEST ===")
    for dataset in ["BITS2", "SisFall"]:
        print(dataset + ": " + json.dumps(test_datasets[dataset], indent=2))
    print("MACRO: " + json.dumps(test_datasets.get("macro", {}), indent=2))
    print("POOLED: " + json.dumps(test_pooled, indent=2))
    print(f"Saved: {out / 'multidataset_v4_event_model.joblib'}")
    print(f"Saved: {out / 'multidataset_v4_event_report.json'}")
    print(f"Saved: {out / 'multidataset_v4_test_recording_predictions.csv'}")


if __name__ == "__main__":
    main()
