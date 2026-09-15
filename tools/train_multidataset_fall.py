#!/usr/bin/env python3
"""
SafeBand multi-dataset fall benchmark.

Combines precomputed 84-feature V6 window tables from BITS-2 and SisFall.
Splits by SUBJECT (never by window), fits preprocessing on TRAIN only, and
reports:
  1) combined model on held-out BITS-2 subjects
  2) combined model on held-out SisFall subjects
  3) per-dataset and pooled test metrics
  4) model comparison: Logistic, RandomForest, ExtraTrees, HistGradientBoosting

Expected input CSV columns:
  subject_id, recording_id (or source_file), label/is_fall, plus 84 V6 features.

For BITS-2, the existing fall_windows_20.csv is acceptable if it contains
subject_id and is_fall. For SisFall, the generated
datasets/processed/sisfall_windows.csv is acceptable.

Important:
- No random window split.
- No test-domain fitting.
- Dataset membership is retained for domain-wise evaluation.
"""

from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bits2", required=True)
    p.add_argument("--sisfall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-subjects", type=int, default=4)
    return p.parse_args()


def find_col(df, names, required=True):
    lower = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    if required:
        raise ValueError(f"Missing one of columns: {names}")
    return None


def normalize_subject(v):
    s = str(v)
    # Preserve BITS-2 user IDs and SisFall SA/SE IDs.
    m = re.search(r"(S[AE]\d+|user\d+)", s, re.I)
    return m.group(1).upper() if m else s


V6_FEATURES = [
    # 38 base motion features
    "ax_mean","ax_std","ax_min","ax_max","ax_rms",
    "ay_mean","ay_std","ay_min","ay_max","ay_rms",
    "az_mean","az_std","az_min","az_max","az_rms",
    "mag_mean","mag_std","mag_min","mag_max","mag_rms",
    "mag_p10","mag_p25","mag_p50","mag_p75","mag_p90",
    "mag_sma","mag_range",
    "jerk_mean","jerk_std","jerk_max","diff_energy","mag_zero_crossings",
    "xy_corr","xz_corr","yz_corr",
    "fft_low","fft_mid","fft_high",
    # 10 event features
    "pre_peak_mean","peak_magnitude","post_peak_mean",
    "peak_to_baseline","peak_index_ratio","pre_post_change",
    "post_peak_std","peak_width","high_energy_fraction","recovery_ratio",
    # 36 V6 morphology features
    "v6_mag_mean","v6_mag_std","v6_mag_min","v6_mag_max","v6_mag_rms",
    "v6_mag_p10","v6_mag_p25","v6_mag_p50","v6_mag_p75","v6_mag_p90",
    "v6_peak_prominence","v6_peak_to_rms","v6_peak_to_median",
    "v6_peak_index_ratio","v6_peak_width_half","v6_high_fraction",
    "v6_pre_mean","v6_post_mean","v6_pre_std","v6_post_std",
    "v6_pre_post_ratio","v6_pre_post_delta","v6_post_quiet_ratio",
    "v6_pre_energy","v6_post_energy","v6_post_energy_ratio",
    "v6_jerk_abs_mean","v6_jerk_abs_max","v6_jerk_std",
    "v6_jerk_sign_changes","v6_diff_energy","v6_impulse_area",
    "v6_axis_peak_ratio","v6_axis_std_ratio","v6_tilt_change",
    "v6_spectral_entropy",
]

assert len(V6_FEATURES) == 84

def identify_recording_column(df):
    return find_col(df, ["recording_id", "source_file", "recording", "file", "source_file_name"], required=False)


def identify_label_column(df):
    # BITS-2 normally has is_fall. SisFall's prepared CSV uses
    # recording_type with values FALL/NONFALL. Support both explicitly.
    col = find_col(
        df,
        ["is_fall", "target", "y", "label", "recording_type"],
        required=True,
    )
    return col


def load_dataset(path, dataset_name):
    df = pd.read_csv(path)
    subject_col = find_col(df, ["subject_id", "subject", "user_id", "user"])
    label_col = identify_label_column(df)
    rec_col = identify_recording_column(df)

    df = df.copy()
    df["__subject"] = df[subject_col].map(normalize_subject)
    if rec_col:
        df["__recording"] = df[rec_col].astype(str)
    else:
        # A recording identifier is strongly preferred. Fall/nonfall windows
        # from a single recording should never be split across train/test.
        if "source_file" in df.columns:
            df["__recording"] = df["source_file"].astype(str)
        else:
            df["__recording"] = np.arange(len(df)).astype(str)

    y_raw = df[label_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        y = pd.to_numeric(y_raw, errors="coerce")
    else:
        # Accept the exact labels used by both prepared datasets.
        y = y_raw.astype(str).str.strip().str.upper().map({
            "FALL": 1,
            "NON_FALL": 0,
            "NONFALL": 0,
            "NON-FALL": 0,
            "TRUE": 1,
            "FALSE": 0,
            "1": 1,
            "0": 0,
        })
    if y.isna().any():
        raise ValueError(f"{dataset_name}: unable to parse labels in {label_col}")
    df["__y"] = y.astype(int)

    # The CSVs contain numeric metadata as well (for example label_id,
    # start_sample, end_sample and window_samples). Therefore "all numeric
    # columns" is NOT the V6 feature contract. Select the explicit 84-feature
    # contract instead, in the exact order expected by V6.
    missing = [c for c in V6_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_name}: missing {len(missing)} V6 features: {missing}"
        )
    non_numeric = [c for c in V6_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"{dataset_name}: V6 features that are not numeric: {non_numeric}"
        )
    numeric = V6_FEATURES.copy()

    df["__dataset"] = dataset_name
    return df, numeric


def grouped_three_way_split(df, seed, test_size, val_size, min_subjects):
    rng = np.random.RandomState(seed)
    subjects = np.array(sorted(df["__subject"].unique()))
    if len(subjects) < min_subjects:
        raise ValueError(f"{df['__dataset'].iloc[0]} has only {len(subjects)} subjects")

    # Stratify at subject level by whether subject has any falls. This avoids
    # assigning individual windows based on their labels.
    stats = df.groupby("__subject")["__y"].agg(["sum", "count"])
    fall_subjects = np.array(stats.index[stats["sum"] > 0])
    nonfall_subjects = np.array(stats.index[stats["sum"] == 0])

    # For SisFall, most elderly subjects are non-fall-only; keeping these
    # represented in train/test is useful for FPR evaluation.
    rng.shuffle(fall_subjects)
    rng.shuffle(nonfall_subjects)

    def split_group(arr):
        n = len(arr)
        n_test = max(1, int(round(n * test_size))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_size))) if n - n_test >= 3 else 0
        test = arr[:n_test]
        val = arr[n_test:n_test+n_val]
        train = arr[n_test+n_val:]
        return train, val, test

    ftr, fva, fte = split_group(fall_subjects)
    ntr, nva, nte = split_group(nonfall_subjects)

    train = set(ftr) | set(ntr)
    val = set(fva) | set(nva)
    test = set(fte) | set(nte)

    # Guarantee disjointness and non-empty sets.
    if not train or not val or not test:
        raise ValueError(
            f"Unable to form 3-way subject split for {df['__dataset'].iloc[0]}. "
            f"subjects={len(subjects)}, fall_subjects={len(fall_subjects)}, "
            f"nonfall_subjects={len(nonfall_subjects)}"
        )
    if train & val or train & test or val & test:
        raise AssertionError("Subject leakage in split")

    return train, val, test


def make_pipeline(kind, seed):
    if kind == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            )),
        ])
    if kind == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500, class_weight="balanced", random_state=seed,
                n_jobs=-1, min_samples_leaf=2
            )),
        ])
    if kind == "et":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=500, class_weight="balanced", random_state=seed,
                n_jobs=-1, min_samples_leaf=2
            )),
        ])
    if kind == "hgb":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=seed
            )),
        ])
    raise ValueError(kind)


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    out = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": cm,
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "predicted_positives": int(np.sum(pred)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["roc_auc"] = None
    return out


def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    bits2, bfeat = load_dataset(args.bits2, "BITS2")
    sisfall, sfeat = load_dataset(args.sisfall, "SisFall")

    if bfeat != sfeat:
        raise ValueError(
            "Feature column/order mismatch between BITS-2 and SisFall. "
            f"BITS-2 count={len(bfeat)}, SisFall count={len(sfeat)}"
        )
    features = bfeat

    datasets = [bits2, sisfall]

    # Hard schema guard before any expensive training.
    for dname, dframe in [("BITS2", bits2), ("SisFall", sisfall)]:
        print(
            f"[Schema] {dname}: rows={len(dframe)}, "
            f"subjects={dframe['__subject'].nunique()}, "
            f"falls={int(dframe['__y'].sum())}, "
            f"nonfalls={int((dframe['__y'] == 0).sum())}, "
            f"features={len(features)}"
        )
    split_map = {}
    for df in datasets:
        tr, va, te = grouped_three_way_split(
            df, args.seed, args.test_size, args.val_size, args.min_subjects
        )
        split_map[df["__dataset"].iloc[0]] = {
            "train": tr, "val": va, "test": te
        }

    # Combined train/validation/test. Subjects remain isolated WITHIN each
    # dataset. A subject identifier is never used as a feature.
    train_df = pd.concat([
        bits2[bits2["__subject"].isin(split_map["BITS2"]["train"])],
        sisfall[sisfall["__subject"].isin(split_map["SisFall"]["train"])]
    ], ignore_index=True)
    val_df = pd.concat([
        bits2[bits2["__subject"].isin(split_map["BITS2"]["val"])],
        sisfall[sisfall["__subject"].isin(split_map["SisFall"]["val"])]
    ], ignore_index=True)
    test_df = pd.concat([
        bits2[bits2["__subject"].isin(split_map["BITS2"]["test"])],
        sisfall[sisfall["__subject"].isin(split_map["SisFall"]["test"])]
    ], ignore_index=True)

    Xtr, ytr = train_df[features], train_df["__y"].to_numpy()
    Xva, yva = val_df[features], val_df["__y"].to_numpy()
    Xte, yte = test_df[features], test_df["__y"].to_numpy()

    reports = {
        "schema": {
            "feature_count": len(features),
            "features": features,
            "window_contract": "60 samples @ 20Hz, step 30",
            "split_unit": "subject",
            "seed": args.seed,
        },
        "datasets": {
            "BITS2": {
                "rows": len(bits2), "subjects": bits2["__subject"].nunique(),
                "falls": int(bits2["__y"].sum())
            },
            "SisFall": {
                "rows": len(sisfall), "subjects": sisfall["__subject"].nunique(),
                "falls": int(sisfall["__y"].sum())
            },
        },
        "splits": {},
        "models": {}
    }

    for name in ["BITS2", "SisFall"]:
        d = datasets[0] if name == "BITS2" else datasets[1]
        reports["splits"][name] = {}
        for part in ["train", "val", "test"]:
            sub = d[d["__subject"].isin(split_map[name][part])]
            reports["splits"][name][part] = {
                "subjects": sorted(map(str, split_map[name][part])),
                "rows": len(sub),
                "falls": int(sub["__y"].sum()),
                "nonfalls": int((sub["__y"] == 0).sum())
            }

    print(f"[MultiDataset] BITS-2 rows={len(bits2)} subjects={bits2['__subject'].nunique()}")
    print(f"[MultiDataset] SisFall rows={len(sisfall)} subjects={sisfall['__subject'].nunique()}")
    print(f"[MultiDataset] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Select model on combined validation F1, then refit the selected model on
    # train+validation. Validation is never used for feature fitting outside
    # the model pipeline's train-time fitting.
    candidates = {}
    for kind in ["logistic", "rf", "et", "hgb"]:
        print(f"[MultiDataset] fitting {kind}...")
        model = make_pipeline(kind, args.seed)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        pva = model.predict_proba(Xva)[:, list(model.classes_).index(1)]
        val_m = metrics(yva, pva)
        candidates[kind] = {
            "validation": val_m,
            "fit_seconds": time.perf_counter() - t0
        }
        print(
            f"  {kind}: val F1={val_m['f1']:.4f} "
            f"recall={val_m['recall']:.4f} FPR="
            f"{val_m['confusion_matrix'][0][1] / max(1, val_m['confusion_matrix'][0][0] + val_m['confusion_matrix'][0][1]):.4f}"
        )

    selected = max(
        candidates,
        key=lambda k: (
            candidates[k]["validation"]["f1"],
            candidates[k]["validation"]["balanced_accuracy"]
        )
    )

    final_train = pd.concat([train_df, val_df], ignore_index=True)
    final_model = make_pipeline(selected, args.seed)
    final_model.fit(final_train[features], final_train["__y"].to_numpy())

    ptest = final_model.predict_proba(Xte)[:, list(final_model.classes_).index(1)]

    reports["models"]["candidates"] = candidates
    reports["models"]["selected"] = selected
    reports["models"]["selected_test"] = metrics(yte, ptest)

    for ds in ["BITS2", "SisFall", "POOLED"]:
        if ds == "POOLED":
            sub = test_df
            pp = ptest
        else:
            mask = test_df["__dataset"].to_numpy() == ds
            sub = test_df.loc[mask]
            pp = ptest[mask]
        reports["models"]["selected_test_" + ds.lower()] = metrics(
            sub["__y"].to_numpy(), pp
        )

    # A useful transfer diagnostic: models trained on one dataset and tested
    # on the other. This is deliberately a secondary diagnostic, not the
    # final model-selection criterion.
    transfer = {}
    for train_name, train_source, test_name, test_source in [
        ("BITS2", bits2, "SisFall", sisfall),
        ("SisFall", sisfall, "BITS2", bits2),
    ]:
        train_subjects = split_map[train_name]["train"] | split_map[train_name]["val"]
        test_subjects = split_map[test_name]["test"]
        tr = train_source[train_source["__subject"].isin(train_subjects)]
        te = test_source[test_source["__subject"].isin(test_subjects)]
        m = make_pipeline(selected, args.seed)
        m.fit(tr[features], tr["__y"].to_numpy())
        pp = m.predict_proba(te[features])[:, list(m.classes_).index(1)]
        transfer[f"{train_name}_to_{test_name}"] = metrics(te["__y"].to_numpy(), pp)

    reports["transfer_diagnostics"] = transfer

    joblib.dump({
        "model": final_model,
        "feature_columns": features,
        "selected_model": selected,
        "split_seed": args.seed,
        "training_datasets": ["BITS2", "SisFall"],
        "window_contract": "60 samples @ 20Hz, step 30",
    }, outdir / "multidataset_fall_model.joblib")

    (outdir / "multidataset_fall_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    print("\n[MultiDataset] SELECTED:", selected)
    print("[MultiDataset] POOLED TEST:", json.dumps(
        reports["models"]["selected_test_pooled"], indent=2
    ))
    print("[MultiDataset] BITS-2 TEST:", json.dumps(
        reports["models"]["selected_test_bits2"], indent=2
    ))
    print("[MultiDataset] SisFall TEST:", json.dumps(
        reports["models"]["selected_test_sisfall"], indent=2
    ))
    print(f"[MultiDataset] saved: {outdir / 'multidataset_fall_model.joblib'}")
    print(f"[MultiDataset] saved: {outdir / 'multidataset_fall_report.json'}")


if __name__ == "__main__":
    main()
