#!/usr/bin/env python3
"""
SafeBand Multi-Dataset Fall Benchmark V2
=========================================

Purpose:
    Compare dataset-balanced training strategies using the existing 84-feature
    V6 window contract, while preserving subject-independent evaluation.

Strategies:
    1. sample_weight_equal_dataset
       Every dataset contributes equal total training weight.
    2. balanced_subsample
       Randomly subsample the larger dataset to the smaller dataset size.
    3. unweighted
       V1 baseline.

Models:
    Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting.

Evaluation:
    - Combined held-out subjects
    - BITS-2 held-out subjects
    - SisFall held-out subjects
    - Macro-average of BITS-2/SisFall metrics
    - Dataset-specific confusion matrices and ROC-AUC
    - Optional transfer diagnostics

No test data is used for selection.
Model selection:
    highest mean of BITS-2 and SisFall validation F1.
"""

from __future__ import annotations
import argparse, json, re, time
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


V6_FEATURES = [
    "ax_mean","ax_std","ax_min","ax_max","ax_rms",
    "ay_mean","ay_std","ay_min","ay_max","ay_rms",
    "az_mean","az_std","az_min","az_max","az_rms",
    "mag_mean","mag_std","mag_min","mag_max","mag_rms",
    "mag_p10","mag_p25","mag_p50","mag_p75","mag_p90",
    "mag_sma","mag_range",
    "jerk_mean","jerk_std","jerk_max","diff_energy","mag_zero_crossings",
    "xy_corr","xz_corr","yz_corr",
    "fft_low","fft_mid","fft_high",
    "pre_peak_mean","peak_magnitude","post_peak_mean",
    "peak_to_baseline","peak_index_ratio","pre_post_change",
    "post_peak_std","peak_width","high_energy_fraction","recovery_ratio",
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


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--bits2", required=True)
    p.add_argument("--sisfall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.20)
    return p.parse_args()


def col(df, candidates):
    low = {str(c).lower(): c for c in df.columns}
    for x in candidates:
        if x.lower() in low:
            return low[x.lower()]
    return None


def subject_key(v):
    s = str(v)
    m = re.search(r"(S[AE]\d+|user\d+)", s, re.I)
    return m.group(1).upper() if m else s


def load(path, dataset):
    df = pd.read_csv(path)

    sc = col(df, ["subject_id", "subject", "user_id", "user"])
    if sc is None:
        raise ValueError(f"{dataset}: subject identifier not found")

    lc = col(df, ["is_fall", "target", "y", "label", "recording_type"])
    if lc is None:
        raise ValueError(f"{dataset}: fall label not found")

    rc = col(df, ["recording_id", "source_file", "recording", "file"])
    if rc is None:
        raise ValueError(
            f"{dataset}: recording identifier not found. "
            "A recording-level identifier is required to protect against window leakage."
        )

    missing = [x for x in V6_FEATURES if x not in df.columns]
    if missing:
        raise ValueError(f"{dataset}: missing V6 features: {missing}")

    nonnum = [x for x in V6_FEATURES if not pd.api.types.is_numeric_dtype(df[x])]
    if nonnum:
        raise ValueError(f"{dataset}: non-numeric V6 features: {nonnum}")

    out = df.copy()
    out["__subject"] = out[sc].map(subject_key)
    out["__recording"] = out[rc].astype(str)

    raw = out[lc]
    if pd.api.types.is_numeric_dtype(raw):
        y = pd.to_numeric(raw, errors="coerce")
    else:
        y = raw.astype(str).str.strip().str.upper().map({
            "FALL": 1, "NON_FALL": 0, "NONFALL": 0,
            "NON-FALL": 0, "TRUE": 1, "FALSE": 0,
            "1": 1, "0": 0,
        })
    if y.isna().any():
        raise ValueError(f"{dataset}: unrecognized labels in {lc}")
    out["__y"] = y.astype(int)
    out["__dataset"] = dataset
    return out


def split_subjects(df, seed, test_size, val_size):
    rng = np.random.RandomState(seed)
    stats = df.groupby("__subject")["__y"].agg(["sum", "count"])
    fall = np.array(stats.index[stats["sum"] > 0])
    nonfall = np.array(stats.index[stats["sum"] == 0])
    rng.shuffle(fall)
    rng.shuffle(nonfall)

    def one(a):
        n = len(a)
        nt = max(1, round(n * test_size)) if n >= 3 else 0
        nv = max(1, round(n * val_size)) if n - nt >= 3 else 0
        return a[nt + nv:], a[nt:nt + nv], a[:nt]

    ft, fv, ftest = one(fall)
    nt, nv, ntest = one(nonfall)

    tr, va, te = set(ft) | set(nt), set(fv) | set(nv), set(ftest) | set(ntest)

    if not tr or not va or not te:
        raise ValueError(
            f"{df['__dataset'].iloc[0]}: failed to create train/val/test "
            f"subject split; subjects={df['__subject'].nunique()}"
        )
    if tr & va or tr & te or va & te:
        raise AssertionError("Subject leakage detected")
    return tr, va, te


def pipe(kind, seed):
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
                n_estimators=500, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=seed
            )),
        ])
    if kind == "et":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=500, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=seed
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


def score(y, p, threshold=.5):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "predicted_positives": int(np.sum(pred)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    )
    out["false_positive_rate"] = float(
        cm[0, 1] / max(1, cm[0, 0] + cm[0, 1])
    )
    return out


def split_frame(df, split):
    return df[df["__subject"].isin(split)].copy()


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    b = load(a.bits2, "BITS2")
    s = load(a.sisfall, "SisFall")

    if len(V6_FEATURES) != 84:
        raise AssertionError("V6 contract corrupted")

    bs = split_subjects(b, a.seed, a.test_size, a.val_size)
    ss = split_subjects(s, a.seed, a.test_size, a.val_size)
    splits = {"BITS2": bs, "SisFall": ss}

    bt, bv, btest = map(lambda x: split_frame(b, x), bs)
    st, sv, stest = map(lambda x: split_frame(s, x), ss)

    print(f"[Schema] BITS2 rows={len(b)} subjects={b.__subject.nunique()} falls={b.__y.sum()}")
    print(f"[Schema] SisFall rows={len(s)} subjects={s.__subject.nunique()} falls={s.__y.sum()}")
    print(f"[Split] BITS2 train/val/test={len(bt)}/{len(bv)}/{len(btest)}")
    print(f"[Split] SisFall train/val/test={len(st)}/{len(sv)}/{len(stest)}")

    Xb = b[V6_FEATURES]
    Xs = s[V6_FEATURES]

    # Validation is kept as two independent domains. Model selection uses the
    # mean validation F1, preventing SisFall's larger sample count from
    # dominating selection.
    val_combined = pd.concat([bv, sv], ignore_index=True)
    train_combined = pd.concat([bt, st], ignore_index=True)
    test_combined = pd.concat([btest, stest], ignore_index=True)

    strategies = ["unweighted", "equal_dataset_weight", "balanced_subsample"]
    models = ["logistic", "rf", "et", "hgb"]

    reports = {
        "schema": {
            "feature_count": 84,
            "window_contract": "60 samples @ 20Hz, step 30",
            "split_unit": "subject",
            "seed": a.seed,
            "features": V6_FEATURES,
        },
        "dataset_sizes": {
            "BITS2": {"rows": len(b), "subjects": int(b.__subject.nunique()),
                      "falls": int(b.__y.sum())},
            "SisFall": {"rows": len(s), "subjects": int(s.__subject.nunique()),
                        "falls": int(s.__y.sum())},
        },
        "splits": {},
        "validation": {},
        "test": {},
    }

    for name, sp in splits.items():
        reports["splits"][name] = {}
        df = b if name == "BITS2" else s
        for part, ids in zip(["train", "val", "test"], sp):
            z = split_frame(df, ids)
            reports["splits"][name][part] = {
                "subjects": sorted(map(str, ids)),
                "rows": len(z),
                "falls": int(z.__y.sum()),
                "nonfalls": int((z.__y == 0).sum()),
            }

    # Build training data for each strategy.
    def training_data(strategy):
        if strategy == "balanced_subsample":
            n = min(len(bt), len(st))
            br = bt.sample(n=n, random_state=a.seed)
            sr = st.sample(n=n, random_state=a.seed)
            return pd.concat([br, sr], ignore_index=True), None

        tr = pd.concat([bt, st], ignore_index=True)
        if strategy == "equal_dataset_weight":
            weights = np.where(tr["__dataset"].to_numpy() == "BITS2",
                               0.5 / len(bt), 0.5 / len(st))
            # Normalize so average weight is 1; relative dataset contribution
            # remains exactly 50/50.
            weights = weights * len(tr)
            return tr, weights
        return tr, None

    best = None
    for strategy in strategies:
        tr, weights = training_data(strategy)
        print(f"\n[Strategy] {strategy}: training rows={len(tr)}")
        reports["validation"].setdefault(strategy, {})

        for kind in models:
            print(f"  fitting {kind}...")
            m = pipe(kind, a.seed)
            fit_kwargs = {}
            if weights is not None:
                fit_kwargs["model__sample_weight"] = weights
            t0 = time.perf_counter()
            m.fit(tr[V6_FEATURES], tr["__y"].to_numpy(), **fit_kwargs)

            # Separate-domain validation, then macro average.
            p_b = m.predict_proba(bv[V6_FEATURES])[:, list(m.classes_).index(1)]
            p_s = m.predict_proba(sv[V6_FEATURES])[:, list(m.classes_).index(1)]
            mb = score(bv["__y"].to_numpy(), p_b)
            ms = score(sv["__y"].to_numpy(), p_s)

            macro = {
                "accuracy": (mb["accuracy"] + ms["accuracy"]) / 2,
                "balanced_accuracy": (mb["balanced_accuracy"] + ms["balanced_accuracy"]) / 2,
                "precision": (mb["precision"] + ms["precision"]) / 2,
                "recall": (mb["recall"] + ms["recall"]) / 2,
                "f1": (mb["f1"] + ms["f1"]) / 2,
                "roc_auc": (
                    (mb["roc_auc"] + ms["roc_auc"]) / 2
                    if mb["roc_auc"] is not None and ms["roc_auc"] is not None else None
                ),
            }

            reports["validation"][strategy][kind] = {
                "BITS2": mb,
                "SisFall": ms,
                "macro": macro,
                "fit_seconds": time.perf_counter() - t0,
                "train_rows": len(tr),
            }

            key = (macro["f1"], macro["balanced_accuracy"])
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "strategy": strategy,
                    "model": kind,
                }
            print(
                f"    BITS2 F1={mb['f1']:.4f} AUC={mb['roc_auc']:.4f} | "
                f"SisFall F1={ms['f1']:.4f} AUC={ms['roc_auc']:.4f} | "
                f"macro F1={macro['f1']:.4f}"
            )

    # Refit the selected strategy on train+validation, still respecting
    # dataset balancing. Test remains untouched.
    final_b = pd.concat([bt, bv], ignore_index=True)
    final_s = pd.concat([st, sv], ignore_index=True)
    final_all = pd.concat([final_b, final_s], ignore_index=True)

    if best["strategy"] == "balanced_subsample":
        n = min(len(final_b), len(final_s))
        final_train = pd.concat([
            final_b.sample(n=n, random_state=a.seed),
            final_s.sample(n=n, random_state=a.seed)
        ], ignore_index=True)
        weights = None
    elif best["strategy"] == "equal_dataset_weight":
        final_train = final_all
        weights = np.where(final_train["__dataset"].to_numpy() == "BITS2",
                           0.5 / len(final_b), 0.5 / len(final_s))
        weights = weights * len(final_train)
    else:
        final_train = final_all
        weights = None

    final_model = pipe(best["model"], a.seed)
    kwargs = {"model__sample_weight": weights} if weights is not None else {}
    final_model.fit(final_train[V6_FEATURES], final_train["__y"].to_numpy(), **kwargs)

    test_all = pd.concat([btest, stest], ignore_index=True)
    p_all = final_model.predict_proba(test_all[V6_FEATURES])[:, list(final_model.classes_).index(1)]

    reports["selected"] = {
        "strategy": best["strategy"],
        "model": best["model"],
        "selection_metric": "mean(BITS2 validation F1, SisFall validation F1)",
    }

    reports["test"]["BITS2"] = score(
        btest["__y"].to_numpy(),
        p_all[test_all["__dataset"].to_numpy() == "BITS2"]
    )
    reports["test"]["SisFall"] = score(
        stest["__y"].to_numpy(),
        p_all[test_all["__dataset"].to_numpy() == "SisFall"]
    )
    reports["test"]["macro_dataset_average"] = {
        k: (reports["test"]["BITS2"][k] + reports["test"]["SisFall"][k]) / 2
        for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    }
    reports["test"]["pooled"] = score(test_all["__y"].to_numpy(), p_all)

    # Secondary transfer diagnostics using the selected algorithm only.
    transfer = {}
    m_bt = pipe(best["model"], a.seed)
    m_bt.fit(pd.concat([bt, bv])[V6_FEATURES], pd.concat([bt, bv])["__y"].to_numpy())
    p = m_bt.predict_proba(stest[V6_FEATURES])[:, list(m_bt.classes_).index(1)]
    transfer["BITS2_to_SisFall"] = score(stest["__y"].to_numpy(), p)

    m_st = pipe(best["model"], a.seed)
    m_st.fit(pd.concat([st, sv])[V6_FEATURES], pd.concat([st, sv])["__y"].to_numpy())
    p = m_st.predict_proba(btest[V6_FEATURES])[:, list(m_st.classes_).index(1)]
    transfer["SisFall_to_BITS2"] = score(btest["__y"].to_numpy(), p)
    reports["transfer_diagnostics"] = transfer

    artifact = {
        "model": final_model,
        "feature_columns": V6_FEATURES,
        "selected_strategy": best["strategy"],
        "selected_model": best["model"],
        "seed": a.seed,
        "window_contract": "60 samples @ 20Hz, step 30",
        "training_datasets": ["BITS2", "SisFall"],
    }
    joblib.dump(artifact, out / "multidataset_v2_fall_model.joblib")
    (out / "multidataset_v2_fall_report.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )

    print("\n=== FINAL V2 ===")
    print("Selected strategy:", best["strategy"])
    print("Selected model:", best["model"])
    print("BITS-2 TEST:", json.dumps(reports["test"]["BITS2"], indent=2))
    print("SisFall TEST:", json.dumps(reports["test"]["SisFall"], indent=2))
    print("DATASET-MACRO:", json.dumps(reports["test"]["macro_dataset_average"], indent=2))
    print("POOLED:", json.dumps(reports["test"]["pooled"], indent=2))
    print("Saved:", out / "multidataset_v2_fall_model.joblib")
    print("Saved:", out / "multidataset_v2_fall_report.json")


if __name__ == "__main__":
    main()
