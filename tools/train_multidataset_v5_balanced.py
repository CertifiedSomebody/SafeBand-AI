#!/usr/bin/env python3
"""SafeBand Multi-Dataset V5: dataset-balanced, leakage-safe fall benchmark.

V4 established that temporal confirmation works, but the held-out BITS-2
false-positive rate remains much higher than SisFall. V5 therefore changes
TRAINING, not the held-out test protocol:

* keep the exact V4 subject split;
* keep the frozen 84-feature contract and 60@20Hz / step-30 windows;
* give each dataset equal total training weight so SisFall cannot dominate
  simply because it has more windows;
* within each dataset, balance FALL/NON_FALL weights;
* optionally perform a second training pass with train-only hard-negative
  weighting, using out-of-fold predictions on TRAIN subjects only;
* select the temporal operating point on VALIDATION only;
* refit on TRAIN+VALIDATION with the selected weighting strategy;
* evaluate TEST exactly once.

No test data is used for model selection or hard-negative mining.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_multidataset_v4_event import (  # noqa: E402
    FEATURES,
    dataset_event_metrics,
    evaluate_operating_points,
    fall_probability,
    load_dataset,
    recording_metrics,
    select_operating_point,
    split_subjects,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bits2", required=True)
    p.add_argument("--sisfall", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", default="rf,et,hgb")
    p.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--min-hits", default="1,2,3,4,5")
    p.add_argument("--max-gaps", default="30,60,90,120")
    p.add_argument("--hard-negative", action="store_true",
                   help="Mine hard NON_FALL windows using train-only GroupKFold OOF probabilities.")
    p.add_argument("--hard-negative-quantile", type=float, default=0.90,
                   help="Within TRAIN non-falls, probability quantile above which windows get extra weight.")
    p.add_argument("--hard-negative-weight", type=float, default=3.0)
    return p.parse_args()


def make_model(kind: str, seed: int):
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, class_weight=None, random_state=seed,
            n_jobs=-1, min_samples_leaf=2, max_features="sqrt"
        )
    if kind == "et":
        return ExtraTreesClassifier(
            n_estimators=500, class_weight=None, random_state=seed,
            n_jobs=-1, min_samples_leaf=2, max_features="sqrt"
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=seed
        )
    raise ValueError(f"Unknown model: {kind}")


def equal_dataset_class_weights(df: pd.DataFrame) -> np.ndarray:
    """Give each dataset 50% of total weight and each class 50% within it."""
    w = np.ones(len(df), dtype=float)
    for dataset in sorted(df["__dataset"].unique()):
        dm = df["__dataset"].eq(dataset).to_numpy()
        n_dataset = max(1, int(dm.sum()))
        for label in (0, 1):
            mask = dm & df["__y"].eq(label).to_numpy()
            n_class = int(mask.sum())
            if n_class:
                # Total dataset weight = 1; each class gets 0.5.
                w[mask] = 0.5 / n_class
    return w * len(df)


def train_weighted(model_kind: str, df: pd.DataFrame, seed: int, weights: np.ndarray):
    model = make_model(model_kind, seed)
    model.fit(df[FEATURES], df["__y"].to_numpy(dtype=int), sample_weight=weights)
    return model


def hard_negative_oof_weights(
    train: pd.DataFrame,
    model_kind: str,
    seed: int,
    base_weights: np.ndarray,
    quantile: float,
    multiplier: float,
) -> tuple[np.ndarray, dict]:
    """Mine hard negatives using only TRAIN subjects via GroupKFold."""
    subjects = train["__subject"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    n_splits = min(5, len(unique_subjects))
    if n_splits < 3:
        return base_weights, {"enabled": False, "reason": "fewer than 3 train subjects"}

    oof = np.full(len(train), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr_i, va_i) in enumerate(gkf.split(train, train["__y"], groups=subjects)):
        model = train_weighted(model_kind, train.iloc[tr_i], seed + fold + 100, base_weights[tr_i])
        oof[va_i] = fall_probability(model, train.iloc[va_i][FEATURES])

    nonfall = train["__y"].to_numpy(dtype=int) == 0
    valid_nonfall = nonfall & np.isfinite(oof)
    if not valid_nonfall.any():
        return base_weights, {"enabled": False, "reason": "no OOF non-fall predictions"}

    cutoff = float(np.quantile(oof[valid_nonfall], quantile))
    hard = valid_nonfall & (oof >= cutoff)
    weights = base_weights.copy()
    weights[hard] *= float(multiplier)

    info = {
        "enabled": True,
        "folds": n_splits,
        "quantile": quantile,
        "cutoff_probability": cutoff,
        "hard_negative_windows": int(hard.sum()),
        "hard_negative_fraction_of_nonfalls": float(hard.sum() / max(1, valid_nonfall.sum())),
        "weight_multiplier": multiplier,
        "oof_auc": float(roc_auc_score(train["__y"], oof)),
    }
    return weights, info


def score_candidate(
    kind: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    seed: int,
    hard_negative: bool,
    quantile: float,
    multiplier: float,
) -> dict:
    base = equal_dataset_class_weights(train)
    hard_info = {"enabled": False}
    weights = base
    if hard_negative:
        weights, hard_info = hard_negative_oof_weights(
            train, kind, seed, base, quantile, multiplier
        )
    t0 = time.perf_counter()
    model = train_weighted(kind, train, seed, weights)
    fit_seconds = time.perf_counter() - t0
    p = fall_probability(model, val[FEATURES])
    thresholds = np.array([0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80])
    candidates = evaluate_operating_points(
        val, p, thresholds, [1,2,3,4,5], [30,60,90,120]
    )
    selected = select_operating_point(candidates)
    return {
        "model": model,
        "weights": weights,
        "validation": selected,
        "fit_seconds": fit_seconds,
        "hard_negative": hard_info,
    }


def main() -> None:
    args = parse_args()
    if not 0.5 <= args.hard_negative_quantile < 1.0:
        raise ValueError("--hard-negative-quantile must be in [0.5, 1.0)")
    if args.hard_negative_weight < 1.0:
        raise ValueError("--hard-negative-weight must be >= 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    min_hits = [int(x) for x in args.min_hits.split(",") if x.strip()]
    max_gaps = [int(x) for x in args.max_gaps.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]

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

    print(f"[V5] BITS2 rows={len(bits2)}; SisFall rows={len(sisfall)}")
    print(f"[V5] train={len(train)} val={len(val)} test={len(test)}")
    print(f"[V5] weighting=equal dataset + equal class")
    print(f"[V5] hard_negative={args.hard_negative}")

    results = {}
    for kind in models:
        print(f"\n=== {kind.upper()} ===")
        result = score_candidate(
            kind, train, val, args.seed, args.hard_negative,
            args.hard_negative_quantile, args.hard_negative_weight,
        )
        selected = result["validation"]
        print(
            f"validation: recall={selected['validation_min_dataset_recall']:.4f} "
            f"macro_f1={selected['validation_macro_f1']:.4f} "
            f"macro_fpr={selected['validation_macro_fpr']:.4f}"
        )
        print(
            f"point: threshold={selected['threshold']} "
            f"min_hits={selected['min_hits']} gap={selected['max_gap_samples']}"
        )
        results[kind] = result

    # Select only from validation data. Priority is the same safety constraint
    # used by V4: maximize the minimum dataset recall, then macro F1, then FPR.
    chosen_kind = max(
        results,
        key=lambda k: (
            results[k]["validation"]["validation_min_dataset_recall"],
            results[k]["validation"]["validation_macro_f1"],
            -results[k]["validation"]["validation_macro_fpr"],
        ),
    )
    chosen = results[chosen_kind]
    op = chosen["validation"]
    print(f"\n[V5] selected model={chosen_kind}")

    # Refit on TRAIN+VALIDATION using exactly the same weighting strategy.
    final_train = pd.concat([train, val], ignore_index=True)
    base_final = equal_dataset_class_weights(final_train)
    final_weights = base_final
    hard_info_final = {"enabled": False}
    if args.hard_negative:
        final_weights, hard_info_final = hard_negative_oof_weights(
            final_train, chosen_kind, args.seed + 500,
            base_final, args.hard_negative_quantile, args.hard_negative_weight,
        )

    t0 = time.perf_counter()
    final_model = train_weighted(chosen_kind, final_train, args.seed, final_weights)
    final_fit_seconds = time.perf_counter() - t0
    test_probability = fall_probability(final_model, test[FEATURES])
    test_pooled, test_events = recording_metrics(
        test, test_probability,
        float(op["threshold"]), int(op["min_hits"]), int(op["max_gap_samples"]),
    )
    test_datasets = dataset_event_metrics(test_events)

    report = {
        "version": "multidataset_v5_dataset_balanced_hard_negative",
        "seed": args.seed,
        "model": chosen_kind,
        "schema": {
            "feature_count": len(FEATURES),
            "window_contract": "60 samples @ 20Hz, step 30",
            "split_unit": "subject",
            "features": FEATURES,
        },
        "training_change": {
            "dataset_total_weight": "equal",
            "class_total_weight_within_dataset": "equal",
            "hard_negative_mining": args.hard_negative,
            "hard_negative_quantile": args.hard_negative_quantile,
            "hard_negative_weight": args.hard_negative_weight,
            "train_only_oof": True,
        },
        "candidate_validation": {
            k: {
                "validation_selected": v["validation"],
                "fit_seconds": v["fit_seconds"],
                "hard_negative": v["hard_negative"],
            } for k, v in results.items()
        },
        "selected_model": chosen_kind,
        "validation_operating_point": op,
        "splits": {
            "BITS2": {"train_subjects": sorted(bs["train"]), "val_subjects": sorted(bs["val"]), "test_subjects": sorted(bs["test"]), "rows": {"train": len(btrain), "val": len(bval), "test": len(btest)}},
            "SisFall": {"train_subjects": sorted(ss["train"]), "val_subjects": sorted(ss["val"]), "test_subjects": sorted(ss["test"]), "rows": {"train": len(strain), "val": len(sval), "test": len(stest)}},
        },
        "test": {
            "operating_point": {"threshold": float(op["threshold"]), "min_hits": int(op["min_hits"]), "max_gap_samples": int(op["max_gap_samples"])},
            "datasets": test_datasets,
            "pooled": test_pooled,
        },
        "timing": {"final_fit_seconds": final_fit_seconds},
        "notes": [
            "V5 changes training weighting while preserving the V4 subject split and 84-feature contract.",
            "Hard-negative mining, when enabled, uses GroupKFold OOF predictions over TRAIN subjects only.",
            "Validation selects the operating point; TEST is held out until the final evaluation.",
            "V5 is a research benchmark and is not deployment-ready without SafeBand hardware validation.",
        ],
    }

    artifact = {
        "model": final_model,
        "feature_columns": FEATURES,
        "model_kind": chosen_kind,
        "seed": args.seed,
        "window_contract": "60 samples @ 20Hz, step 30",
        "event_config": {"threshold": float(op["threshold"]), "min_hits": int(op["min_hits"]), "max_gap_samples": int(op["max_gap_samples"])},
        "training_version": "multidataset_v5_dataset_balanced_hard_negative",
    }
    joblib.dump(artifact, out / "multidataset_v5_event_model.joblib")
    (out / "multidataset_v5_event_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    test_events.to_csv(out / "multidataset_v5_test_recording_predictions.csv", index=False)

    print("\n=== V5 FINAL TEST ===")
    for dataset in ["BITS2", "SisFall"]:
        print(dataset + ": " + json.dumps(test_datasets[dataset], indent=2))
    print("MACRO: " + json.dumps(test_datasets.get("macro", {}), indent=2))
    print("POOLED: " + json.dumps(test_pooled, indent=2))


if __name__ == "__main__":
    main()
