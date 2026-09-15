from __future__ import annotations

import sys
import json
import argparse
import copy
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ai.ppg_motion_features import extract_features
from ai.hr_temporal import HRTemporalFusion


# ============================================================
# SUBJECT SPLIT
# ============================================================

def split_subjects(subjects):
    ids = np.array(sorted(set(subjects.tolist())))

    if len(ids) < 3:
        raise ValueError("At least 3 subjects are required.")

    n_train = round(0.60 * len(ids))
    n_val = round(0.20 * len(ids))

    train_subjects = ids[:n_train]
    val_subjects = ids[n_train:n_train + n_val]
    test_subjects = ids[n_train + n_val:]

    return train_subjects, val_subjects, test_subjects


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    error = np.abs(y_true - y_pred)

    return {
        "n": int(len(y_true)),

        "mae_bpm":
            float(mean_absolute_error(y_true, y_pred)),

        "rmse_bpm":
            float(np.sqrt(
                mean_squared_error(y_true, y_pred)
            )),

        "r2":
            float(r2_score(y_true, y_pred)),

        "accuracy_within_1_bpm_pct":
            float(np.mean(error <= 1.0) * 100),

        "accuracy_within_3_bpm_pct":
            float(np.mean(error <= 3.0) * 100),

        "accuracy_within_5_bpm_pct":
            float(np.mean(error <= 5.0) * 100),

        "accuracy_within_10_bpm_pct":
            float(np.mean(error <= 10.0) * 100),

        "bias_bpm":
            float(np.mean(y_pred - y_true)),
    }


# ============================================================
# CACHE PATHS
# ============================================================

def get_cache_paths(input_path):
    p = Path(input_path)

    feature_cache = (
        p.parent /
        f"{p.stem}_v2features.npz"
    )

    candidate_cache = (
        p.parent /
        f"{p.stem}_v4candidates.npz"
    )

    return feature_cache, candidate_cache


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def build_or_load_features(
    z,
    feature_names,
    cache_path,
    force_rebuild=False,
):
    if cache_path.exists() and not force_rebuild:

        cache = np.load(
            cache_path,
            allow_pickle=False,
        )

        cached_names = list(
            cache["feature_names"].astype(str)
        )

        if cached_names == list(feature_names):

            X = cache["X"].astype(
                np.float32,
                copy=False,
            )

            print(
                f"[2/6] Feature cache loaded: "
                f"{cache_path}"
            )

            print(
                f"      Shape: {X.shape}"
            )

            return X

        print(
            "[2/6] Existing feature cache "
            "has incompatible schema."
        )

        print(
            "      Rebuilding cache..."
        )

    bvp = z["bvp"]
    acc = z["acc"]

    bvp_fs = float(z["bvp_fs"])
    acc_fs = float(z["acc_fs"])

    n = len(bvp)
    feature_count = len(feature_names)

    X = np.empty(
        (n, feature_count),
        dtype=np.float32,
    )

    start = time.time()

    print(
        "[2/6] Extracting exact V2 features..."
    )

    for i in range(n):

        features = extract_features(
            bvp[i],
            acc[i],
            bvp_fs,
            acc_fs,
        )

        actual = set(features)
        expected = set(feature_names)

        if actual != expected:

            missing = sorted(
                expected - actual
            )

            extra = sorted(
                actual - expected
            )

            raise RuntimeError(
                "\nV2 FEATURE CONTRACT FAILURE\n"
                f"Window: {i}\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

        X[i] = [
            features[name]
            for name in feature_names
        ]

        if (
            (i + 1) % 5000 == 0
            or i + 1 == n
        ):
            elapsed = time.time() - start

            print(
                f"      {i + 1}/{n} "
                f"({100.0 * (i + 1) / n:.1f}%) "
                f"elapsed={elapsed:.1f}s"
            )

    np.savez_compressed(
        cache_path,
        X=X,
        feature_names=np.asarray(
            feature_names,
            dtype="U64",
        ),
    )

    print(
        f"[2/6] Feature cache created: "
        f"{cache_path}"
    )

    return X


# ============================================================
# CANDIDATE / QUALITY CACHE
# ============================================================

def build_or_load_candidates(
    z,
    cache_path,
    force_rebuild=False,
):
    if cache_path.exists() and not force_rebuild:

        cache = np.load(
            cache_path,
            allow_pickle=False,
        )

        required = {
            "candidate",
            "quality",
            "motion",
        }

        if required.issubset(cache.files):

            print(
                f"[3/6] Candidate cache loaded: "
                f"{cache_path}"
            )

            return (
                cache["candidate"],
                cache["quality"],
                cache["motion"],
            )

    bvp = z["bvp"]
    acc = z["acc"]

    bvp_fs = float(z["bvp_fs"])
    acc_fs = float(z["acc_fs"])

    n = len(bvp)

    candidate = np.full(
        n,
        np.nan,
        dtype=np.float32,
    )

    quality = np.zeros(
        n,
        dtype=np.float32,
    )

    motion = np.zeros(
        n,
        dtype=np.float32,
    )

    start = time.time()

    print(
        "[3/6] Extracting HR candidates "
        "and quality signals..."
    )

    for i in range(n):

        features = extract_features(
            bvp[i],
            acc[i],
            bvp_fs,
            acc_fs,
        )

        candidate[i] = features.get(
            "candidate_hr_median",
            np.nan,
        )

        spread = features.get(
            "candidate_hr_spread",
            np.nan,
        )

        if np.isfinite(spread):
            quality[i] = np.exp(
                -float(spread) / 12.0
            )
        else:
            quality[i] = 0.0

        # IMPORTANT:
        # Use the existing V2 feature scale.
        motion[i] = float(
            features.get(
                "acc_dynamic_std",
                0.0,
            )
        )

        if (
            (i + 1) % 5000 == 0
            or i + 1 == n
        ):
            elapsed = time.time() - start

            print(
                f"      {i + 1}/{n} "
                f"({100.0 * (i + 1) / n:.1f}%) "
                f"elapsed={elapsed:.1f}s"
            )

    np.savez_compressed(
        cache_path,
        candidate=candidate,
        quality=quality,
        motion=motion,
    )

    print(
        f"[3/6] Candidate cache created: "
        f"{cache_path}"
    )

    return candidate, quality, motion


# ============================================================
# TEMPORAL PASS
# ============================================================

def temporal_pass(
    indices,
    subjects,
    window_index,
    ml_predictions,
    candidates,
    quality,
    motion,
    history,
    max_change_rate,
):
    """
    indices are ALWAYS global dataset indices.

    Output is local to `indices`.

    This explicit mapping prevents the exact IndexError
    that occurred in the previous version.
    """

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    output = np.empty(
        len(indices),
        dtype=float,
    )

    global_to_local = {
        int(global_index): local_index
        for local_index, global_index
        in enumerate(indices)
    }

    for subject in sorted(
        set(subjects[indices])
    ):

        subject_indices = indices[
            subjects[indices] == subject
        ]

        subject_indices = (
            subject_indices[
                np.argsort(
                    window_index[
                        subject_indices
                    ]
                )
            ]
        )

        tracker = HRTemporalFusion(
            history=history,
            max_change_bpm_per_sec=max_change_rate,
        )

        previous_window = None

        for global_index in subject_indices:

            if previous_window is None:
                dt = 2.0
            else:
                dt = max(
                    0.5,
                    float(
                        window_index[global_index]
                        - previous_window
                    ) * 2.0,
                )

            result = tracker.update(
                ml_predictions[global_index],
                candidates[global_index],
                quality[global_index],
                dt,
                motion[global_index],
            )

            local_position = global_to_local[
                int(global_index)
            ]

            output[local_position] = result

            previous_window = int(
                window_index[global_index]
            )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--report-out",
        required=True,
    )

    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------

    print("=" * 65)
    print("SafeBand HR Temporal V4")
    print("=" * 65)

    print(
        "[1/6] Loading dataset and V2 model..."
    )

    z = np.load(
        args.input,
        allow_pickle=True,
    )

    required_arrays = {
        "bvp",
        "acc",
        "hr",
        "subject",
        "window_index",
        "bvp_fs",
        "acc_fs",
    }

    missing = (
        required_arrays -
        set(z.files)
    )

    if missing:
        raise ValueError(
            f"Missing arrays: {sorted(missing)}"
        )

    y = z["hr"].astype(float)

    subjects = (
        z["subject"]
        .astype(str)
    )

    window_index = (
        z["window_index"]
        .astype(int)
    )

    artifact = joblib.load(
        args.model
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise ValueError(
            "Invalid SafeBand model artifact."
        )

    if (
        "model" not in artifact
        or "feature_names"
        not in artifact
    ):
        raise ValueError(
            "Model artifact must contain "
            "'model' and 'feature_names'."
        )

    feature_names = list(
        artifact["feature_names"]
    )

    if len(feature_names) != 31:
        raise ValueError(
            "Expected exactly 31 V2 features, "
            f"found {len(feature_names)}."
        )

    print(
        f"Windows : {len(y)}"
    )

    print(
        f"Subjects: {len(set(subjects))}"
    )

    print(
        f"Features: {len(feature_names)}"
    )

    # --------------------------------------------------------
    # SUBJECT SPLIT
    # --------------------------------------------------------

    train_subjects, val_subjects, test_subjects = (
        split_subjects(subjects)
    )

    train_mask = np.isin(
        subjects,
        train_subjects,
    )

    val_mask = np.isin(
        subjects,
        val_subjects,
    )

    test_mask = np.isin(
        subjects,
        test_subjects,
    )

    if (
        set(train_subjects)
        & set(val_subjects)
        or
        set(train_subjects)
        & set(test_subjects)
        or
        set(val_subjects)
        & set(test_subjects)
    ):
        raise RuntimeError(
            "SUBJECT LEAKAGE DETECTED."
        )

    print(
        f"Train={train_mask.sum()} "
        f"Val={val_mask.sum()} "
        f"Test={test_mask.sum()}"
    )

    # --------------------------------------------------------
    # 2. FEATURES
    # --------------------------------------------------------

    feature_cache, candidate_cache = (
        get_cache_paths(args.input)
    )

    X = build_or_load_features(
        z,
        feature_names,
        feature_cache,
        args.force_rebuild_cache,
    )

    # --------------------------------------------------------
    # 3. CANDIDATES
    # --------------------------------------------------------

    candidates, quality, motion = (
        build_or_load_candidates(
            z,
            candidate_cache,
            args.force_rebuild_cache,
        )
    )

    # --------------------------------------------------------
    # 4. TRAIN-ONLY SELECTOR
    # --------------------------------------------------------

    print(
        "[4/6] Training ONE TRAIN-only "
        "selector model..."
    )

    selector = copy.deepcopy(
        artifact["model"]
    )

    # Speed optimization:
    # temporal parameters don't require 700 trees.
    # The final TEST still uses the supplied frozen V2 model.
    try:
        estimator = (
            selector.named_steps["model"]
        )

        if hasattr(
            estimator,
            "n_estimators",
        ):
            estimator.set_params(
                n_estimators=min(
                    150,
                    int(
                        estimator.n_estimators
                    ),
                )
            )

    except Exception:
        pass

    selector.fit(
        X[train_mask],
        y[train_mask],
    )

    validation_ml = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    validation_ml[val_mask] = (
        selector.predict(
            X[val_mask]
        )
    )

    validation_indices = (
        np.flatnonzero(val_mask)
    )

    # --------------------------------------------------------
    # 5. TEMPORAL SEARCH
    # --------------------------------------------------------

    print(
        "[5/6] Validation-only temporal search..."
    )

    best = None

    configs = [
        (1, 8.0),
        (1, 12.0),
        (1, 16.0),

        (3, 8.0),
        (3, 12.0),
        (3, 16.0),

        (5, 8.0),
        (5, 12.0),
        (5, 16.0),
    ]

    for number, (
        history,
        max_rate,
    ) in enumerate(
        configs,
        start=1,
    ):

        prediction = temporal_pass(
            validation_indices,
            subjects,
            window_index,
            validation_ml,
            candidates,
            quality,
            motion,
            history,
            max_rate,
        )

        result = calculate_metrics(
            y[validation_indices],
            prediction,
        )

        # Primary:
        # lowest validation MAE
        #
        # Secondary:
        # highest ±5 BPM accuracy
        #
        # Tertiary:
        # highest ±3 BPM accuracy
        ranking_key = (
            result["mae_bpm"],
            -result[
                "accuracy_within_5_bpm_pct"
            ],
            -result[
                "accuracy_within_3_bpm_pct"
            ],
        )

        print(
            f"    [{number}/9] "
            f"history={history} "
            f"rate={max_rate:g} "
            f"MAE={result['mae_bpm']:.3f} "
            f"±5={result['accuracy_within_5_bpm_pct']:.2f}%"
        )

        if (
            best is None
            or ranking_key < best[0]
        ):
            best = (
                ranking_key,
                history,
                max_rate,
                result,
            )

    (
        _,
        selected_history,
        selected_rate,
        validation_temporal,
    ) = best

    validation_raw = calculate_metrics(
        y[validation_indices],
        validation_ml[
            validation_indices
        ],
    )

    print()
    print(
        "Selected temporal configuration:"
    )

    print(
        f"  history={selected_history}"
    )

    print(
        f"  max_change="
        f"{selected_rate:g} BPM/s"
    )

    # --------------------------------------------------------
    # 6. HELD-OUT TEST
    # --------------------------------------------------------

    print(
        "[6/6] Evaluating HELD-OUT TEST..."
    )

    test_indices = (
        np.flatnonzero(test_mask)
    )

    # IMPORTANT:
    # This is the supplied final V2 artifact.
    # It is NOT retrained or modified.
    test_ml = np.full(
        len(y),
        np.nan,
        dtype=float,
    )

    test_ml[test_mask] = (
        artifact["model"].predict(
            X[test_mask]
        )
    )

    test_temporal = temporal_pass(
        test_indices,
        subjects,
        window_index,
        test_ml,
        candidates,
        quality,
        motion,
        selected_history,
        selected_rate,
    )

    test_raw = calculate_metrics(
        y[test_indices],
        test_ml[test_indices],
    )

    test_fused = calculate_metrics(
        y[test_indices],
        test_temporal,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {

        "experiment":
            "SafeBand HR Temporal V4 Optimized",

        "dataset": {
            "windows": int(len(y)),
            "subjects": int(len(set(subjects))),
            "bvp_fs_hz":
                float(z["bvp_fs"]),
            "acc_fs_hz":
                float(z["acc_fs"]),
            "window_sec": 8.0,
            "shift_sec": 2.0,
        },

        "split": {
            "method":
                "deterministic_subject_level_60_20_20",

            "train_subjects":
                train_subjects.tolist(),

            "validation_subjects":
                val_subjects.tolist(),

            "test_subjects":
                test_subjects.tolist(),

            "train_windows":
                int(train_mask.sum()),

            "validation_windows":
                int(val_mask.sum()),

            "test_windows":
                int(test_mask.sum()),

            "subject_overlap":
                False,
        },

        "feature_contract": {
            "source":
                "loaded V2 model artifact",

            "count":
                len(feature_names),

            "verified":
                True,

            "cache":
                str(feature_cache),
        },

        "validation_selection": {

            "selector_model":
                "V2-compatible TRAIN-only "
                "model, max 150 trees",

            "raw_ml":
                validation_raw,

            "temporal_fused":
                validation_temporal,

            "history":
                selected_history,

            "max_change_bpm_per_sec":
                selected_rate,
        },

        "held_out_test": {

            "raw_v2":
                test_raw,

            "temporal_fused":
                test_fused,
        },

        "improvement_vs_raw": {

            "mae_delta_bpm":
                float(
                    test_fused["mae_bpm"]
                    -
                    test_raw["mae_bpm"]
                ),

            "rmse_delta_bpm":
                float(
                    test_fused["rmse_bpm"]
                    -
                    test_raw["rmse_bpm"]
                ),

            "accuracy_within_5_delta_pp":
                float(
                    test_fused[
                        "accuracy_within_5_bpm_pct"
                    ]
                    -
                    test_raw[
                        "accuracy_within_5_bpm_pct"
                    ]
                ),
        },

        "notes": [
            "Feature extraction is cached.",
            "Candidate extraction is cached.",
            "Only one TRAIN-only selector model is fitted.",
            "Temporal configurations reuse the same validation predictions.",
            "Temporal parameters are selected using validation subjects only.",
            "The supplied V2 artifact is used unchanged for held-out test.",
            "Subject state resets independently.",
            "Accuracy is reported as tolerance accuracy within BPM.",
            "MAE, RMSE and R2 are also reported.",
            "PPG-DaLiA is Empatica E4 BVP.",
            "This is not a MAX30102 performance claim.",
        ],
    }

    report_path = Path(
        args.report_out
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 65)
    print("V4 OPTIMIZED COMPLETE")
    print("=" * 65)

    print()
    print("TEST RAW V2")
    print(
        json.dumps(
            test_raw,
            indent=2,
        )
    )

    print()
    print("TEST TEMPORAL V4")
    print(
        json.dumps(
            test_fused,
            indent=2,
        )
    )

    print()
    print(
        "Report:",
        report_path,
    )


if __name__ == "__main__":
    main()