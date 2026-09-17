"""
SafeBand-AI — V2 vs V4 PPG feature forensic comparator.

GOAL
----
Determine whether the V4 HR temporal pipeline is numerically reproducing
the established V2 31-feature representation.

This must be run BEFORE trusting V4 model metrics.

The comparator:
    * loads the exact same NPZ windows
    * imports V2 and V4 feature implementations
    * extracts features from identical BVP/ACC windows
    * checks feature names
    * checks feature ordering
    * checks feature dimensionality
    * checks finite values
    * computes absolute/relative differences
    * identifies the largest-difference features
    * produces a machine-readable JSON report

It intentionally FAILS rather than silently adapting mismatched schemas.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ----------------------------------------------------------------------
# Expected sampling contract
# ----------------------------------------------------------------------

BVP_FS = 64.0
ACC_FS = 32.0

BVP_SAMPLES = 512
ACC_SAMPLES = 256


# ----------------------------------------------------------------------
# Explicit known extractor names.
#
# We first try these. If the installed source uses a different name,
# the script can accept --v2-function / --v4-function explicitly.
# ----------------------------------------------------------------------

V2_CANDIDATES = (
    "extract_features",
    "extract_ppg_motion_features",
    "extract_hr_features",
    "compute_features",
)

V4_CANDIDATES = (
    "extract_features",
    "extract_hr_features",
    "extract_ppg_motion_features",
    "compute_features",
)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def public_callables(module: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for name in dir(module):
        if name.startswith("_"):
            continue

        obj = getattr(module, name)

        if callable(obj):
            result[name] = obj

    return result


def choose_extractor(
    module_name: str,
    preferred_names: tuple[str, ...],
    explicit_name: str | None,
) -> tuple[str, Any]:

    module = importlib.import_module(module_name)

    callables = public_callables(module)

    if explicit_name:
        fn = callables.get(explicit_name)

        if fn is None:
            available = sorted(callables.keys())

            raise AttributeError(
                f"Function '{explicit_name}' was not found in "
                f"{module_name}.\n\n"
                f"Available public callables:\n"
                f"{available}"
            )

        return explicit_name, fn

    found = []

    for name in preferred_names:
        if name in callables:
            found.append((name, callables[name]))

    if len(found) == 1:
        return found[0]

    if len(found) > 1:
        raise RuntimeError(
            f"Multiple possible extractors found in {module_name}:\n"
            f"{[name for name, _ in found]}\n\n"
            f"Pass the exact one using the corresponding "
            f"--v2-function or --v4-function option."
        )

    raise RuntimeError(
        f"No known feature extractor found in {module_name}.\n\n"
        f"Pass the exact function using the corresponding "
        f"--v2-function / --v4-function option."
    )


def call_extractor(
    fn: Any,
    bvp: np.ndarray,
    acc: np.ndarray,
) -> Any:

    sig = inspect.signature(fn)

    parameters = [
        p
        for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    names = [
        p.name.lower()
        for p in parameters
    ]

    count = len(parameters)

    # --------------------------------------------------------------
    # One-argument function
    # --------------------------------------------------------------

    if count == 1:
        return fn(bvp)

    # --------------------------------------------------------------
    # Two-argument function
    # --------------------------------------------------------------

    if count == 2:

        # Explicitly detect motion/ACC-oriented API.
        if any(
            token in names
            for token in (
                "acc",
                "accelerometer",
                "motion",
                "motion_data",
            )
        ):
            return fn(bvp, acc)

        # Otherwise assume the second parameter is sampling frequency.
        return fn(bvp, BVP_FS)

    # --------------------------------------------------------------
    # Three-argument function
    # --------------------------------------------------------------

    if count == 3:

        # Most likely:
        #     (bvp, acc, fs)
        return fn(bvp, acc, BVP_FS)

    # --------------------------------------------------------------
    # Four-argument PPG + ACC function
    # --------------------------------------------------------------

    if count == 4:
        return fn(bvp, acc, BVP_FS, ACC_FS)

    raise RuntimeError(
        f"Unsupported extractor signature:\n{fn}"
    )


def normalize_result(result: Any) -> tuple[list[str], np.ndarray]:
    """
    Normalize common feature-return formats.

    Supported:
        dict
        (feature_names, feature_values)
        numpy array
        list / tuple of numerical feature values
    """

    # --------------------------------------------------------------
    # Dict → ideal case because names are explicit.
    # --------------------------------------------------------------

    if isinstance(result, dict):

        names = [
            str(k)
            for k in result.keys()
        ]

        values = np.asarray(
            list(result.values()),
            dtype=np.float64,
        ).reshape(-1)

        return names, values

    # --------------------------------------------------------------
    # Tuple:
    #     (names, values)
    # --------------------------------------------------------------

    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], (list, tuple))
    ):

        names = [
            str(x)
            for x in result[0]
        ]

        values = np.asarray(
            result[1],
            dtype=np.float64,
        ).reshape(-1)

        return names, values

    # --------------------------------------------------------------
    # Raw ndarray/list
    # --------------------------------------------------------------

    if isinstance(result, np.ndarray):

        values = np.asarray(
            result,
            dtype=np.float64,
        ).reshape(-1)

        names = [
            f"feature_{i:03d}"
            for i in range(values.size)
        ]

        return names, values

    if isinstance(result, (list, tuple)):

        values = np.asarray(
            result,
            dtype=np.float64,
        ).reshape(-1)

        names = [
            f"feature_{i:03d}"
            for i in range(values.size)
        ]

        return names, values

    raise TypeError(
        f"Unsupported extractor output type: "
        f"{type(result).__name__}"
    )


def validate_window(
    bvp: np.ndarray,
    acc: np.ndarray,
) -> None:

    if bvp.shape != (BVP_SAMPLES,):
        raise ValueError(
            f"Expected BVP window "
            f"({BVP_SAMPLES},), got {bvp.shape}"
        )

    if acc.shape != (ACC_SAMPLES, 3):
        raise ValueError(
            f"Expected ACC window "
            f"({ACC_SAMPLES},3), got {acc.shape}"
        )

    if not np.isfinite(bvp).all():
        raise ValueError("BVP window contains NaN/Inf.")

    if not np.isfinite(acc).all():
        raise ValueError("ACC window contains NaN/Inf.")


# ----------------------------------------------------------------------
# Main forensic routine
# ----------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Forensic V2/V4 PPG feature comparison."
    )

    parser.add_argument(
        "--input",
        default="datasets/processed/ppgdalia_windows.npz",
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260916,
    )

    parser.add_argument(
        "--v2-module",
        default="ai.ppg_motion_features",
    )

    parser.add_argument(
        "--v4-module",
        default="ai.ppg_motion_features",
    )

    parser.add_argument(
        "--v2-function",
        default="",
    )

    parser.add_argument(
        "--v4-function",
        default="",
    )

    parser.add_argument(
        "--out",
        default="models/ppgdalia_v2_v4_feature_diff.json",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input NPZ does not exist:\n{input_path}"
        )

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    data = np.load(
        input_path,
        allow_pickle=False,
    )

    required = {
        "bvp",
        "acc",
        "hr",
        "subject",
        "window_index",
    }

    missing = required - set(data.files)

    if missing:
        raise ValueError(
            f"Missing arrays in NPZ: {sorted(missing)}"
        )

    bvp_all = np.asarray(data["bvp"])
    acc_all = np.asarray(data["acc"])
    subjects = np.asarray(data["subject"])
    window_index = np.asarray(data["window_index"])

    if bvp_all.ndim != 2 or bvp_all.shape[1] != BVP_SAMPLES:
        raise ValueError(
            f"Unexpected BVP dataset shape: {bvp_all.shape}"
        )

    if (
        acc_all.ndim != 3
        or acc_all.shape[1] != ACC_SAMPLES
        or acc_all.shape[2] != 3
    ):
        raise ValueError(
            f"Unexpected ACC dataset shape: {acc_all.shape}"
        )

    if bvp_all.shape[0] != acc_all.shape[0]:
        raise ValueError(
            "BVP and ACC window counts differ."
        )

    total_windows = bvp_all.shape[0]

    # ------------------------------------------------------------------
    # Deterministic sample selection
    # ------------------------------------------------------------------

    sample_count = min(
        args.sample_count,
        total_windows,
    )

    rng = np.random.default_rng(args.seed)

    indices = np.sort(
        rng.choice(
            total_windows,
            size=sample_count,
            replace=False,
        )
    )

    # ------------------------------------------------------------------
    # Select extractors
    # ------------------------------------------------------------------

    v2_name, v2_fn = choose_extractor(
        args.v2_module,
        V2_CANDIDATES,
        args.v2_function or None,
    )

    v4_name, v4_fn = choose_extractor(
        args.v4_module,
        V4_CANDIDATES,
        args.v4_function or None,
    )

    print("=" * 72)
    print("SafeBand-AI V2 ↔ V4 FEATURE FORENSICS")
    print("=" * 72)
    print(f"Input             : {input_path}")
    print(f"Total windows     : {total_windows}")
    print(f"Windows compared  : {sample_count}")
    print(f"Random seed       : {args.seed}")
    print()
    print(
        f"V2 extractor      : "
        f"{args.v2_module}.{v2_name}"
    )
    print(
        f"V4 extractor      : "
        f"{args.v4_module}.{v4_name}"
    )
    print()

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    reference_names: list[str] | None = None
    difference_matrix: list[np.ndarray] = []

    per_window = []

    # ------------------------------------------------------------------
    # Compare identical windows
    # ------------------------------------------------------------------

    for position, idx in enumerate(indices, start=1):

        bvp = np.asarray(
            bvp_all[idx],
            dtype=np.float64,
        )

        acc = np.asarray(
            acc_all[idx],
            dtype=np.float64,
        )

        validate_window(
            bvp,
            acc,
        )

        v2_result = call_extractor(
            v2_fn,
            bvp,
            acc,
        )

        v4_result = call_extractor(
            v4_fn,
            bvp,
            acc,
        )

        v2_names, v2_values = normalize_result(
            v2_result
        )

        v4_names, v4_values = normalize_result(
            v4_result
        )

        # --------------------------------------------------------------
        # Structural checks
        # --------------------------------------------------------------

        if v2_names != v4_names:
            raise AssertionError(
                "FEATURE NAME / ORDER MISMATCH\n\n"
                f"Window index: {idx}\n\n"
                f"V2 names:\n{v2_names}\n\n"
                f"V4 names:\n{v4_names}\n"
            )

        if v2_values.shape != v4_values.shape:
            raise AssertionError(
                "FEATURE VECTOR LENGTH MISMATCH\n\n"
                f"Window index: {idx}\n"
                f"V2 shape: {v2_values.shape}\n"
                f"V4 shape: {v4_values.shape}\n"
            )

        if not np.isfinite(v2_values).all():
            raise ValueError(
                f"V2 produced non-finite features "
                f"at window {idx}"
            )

        if not np.isfinite(v4_values).all():
            raise ValueError(
                f"V4 produced non-finite features "
                f"at window {idx}"
            )

        if reference_names is None:
            reference_names = v2_names

        # --------------------------------------------------------------
        # Numerical comparison
        # --------------------------------------------------------------

        diff = np.abs(
            v2_values - v4_values
        )

        relative = diff / np.maximum(
            np.abs(v2_values),
            1e-12,
        )

        difference_matrix.append(diff)

        per_window.append(
            {
                "row_index": int(idx),
                "subject": str(subjects[idx]),
                "window_index": int(window_index[idx]),
                "exact_equal": bool(
                    np.array_equal(
                        v2_values,
                        v4_values,
                    )
                ),
                "allclose_1e-12": bool(
                    np.allclose(
                        v2_values,
                        v4_values,
                        atol=1e-12,
                        rtol=1e-12,
                    )
                ),
                "allclose_1e-10": bool(
                    np.allclose(
                        v2_values,
                        v4_values,
                        atol=1e-10,
                        rtol=1e-10,
                    )
                ),
                "max_abs_diff": float(
                    np.max(diff)
                ),
                "mean_abs_diff": float(
                    np.mean(diff)
                ),
                "max_relative_diff": float(
                    np.max(relative)
                ),
            }
        )

        if (
            position == 1
            or position % 16 == 0
            or position == sample_count
        ):
            print(
                f"Compared "
                f"{position:>4}/{sample_count}"
            )

    # ------------------------------------------------------------------
    # Global difference statistics
    # ------------------------------------------------------------------

    if reference_names is None:
        raise RuntimeError(
            "No features were produced."
        )

    D = np.vstack(
        difference_matrix
    )

    feature_reports = []

    for j, name in enumerate(reference_names):

        column = D[:, j]

        feature_reports.append(
            {
                "index": j,
                "feature": name,
                "max_abs_diff": float(
                    np.max(column)
                ),
                "mean_abs_diff": float(
                    np.mean(column)
                ),
                "median_abs_diff": float(
                    np.median(column)
                ),
                "nonzero_count": int(
                    np.count_nonzero(column)
                ),
                "zero_fraction": float(
                    np.mean(column == 0.0)
                ),
            }
        )

    feature_reports.sort(
        key=lambda x: x["max_abs_diff"],
        reverse=True,
    )

    exact_equal_fraction = float(
        np.mean(
            [
                row["exact_equal"]
                for row in per_window
            ]
        )
    )

    allclose_1e10_fraction = float(
        np.mean(
            [
                row["allclose_1e-10"]
                for row in per_window
            ]
        )
    )

    global_max = float(
        np.max(D)
    )

    global_mean = float(
        np.mean(D)
    )

    # ------------------------------------------------------------------
    # PASS / FAIL gate
    # ------------------------------------------------------------------

    equivalent = (
        global_max <= 1e-10
        and allclose_1e10_fraction == 1.0
        and exact_equal_fraction == 1.0
    )

    status = (
        "PASS"
        if equivalent
        else "MISMATCH"
    )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    report = {
        "status": status,
        "input": str(input_path),
        "sample_count": sample_count,
        "seed": args.seed,
        "v2_extractor": (
            f"{args.v2_module}.{v2_name}"
        ),
        "v4_extractor": (
            f"{args.v4_module}.{v4_name}"
        ),
        "feature_count": len(reference_names),
        "feature_names": reference_names,
        "global_summary": {
            "global_max_abs_diff": global_max,
            "global_mean_abs_diff": global_mean,
            "exact_equal_fraction": exact_equal_fraction,
            "allclose_1e10_fraction": allclose_1e10_fraction,
        },
        "feature_differences_sorted": feature_reports,
        "window_results": per_window,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FORENSIC RESULT")
    print("=" * 72)

    print(
        f"Feature count          : "
        f"{len(reference_names)}"
    )

    print(
        f"Global max abs diff    : "
        f"{global_max:.12g}"
    )

    print(
        f"Global mean abs diff   : "
        f"{global_mean:.12g}"
    )

    print(
        f"Exact-equal windows    : "
        f"{exact_equal_fraction:.2%}"
    )

    print(
        f"Allclose (1e-10)       : "
        f"{allclose_1e10_fraction:.2%}"
    )

    print()
    print("Largest feature differences:")

    for row in feature_reports[:10]:

        print(
            f"  {row['index']:>2} "
            f"{row['feature']:<35} "
            f"max={row['max_abs_diff']:.12g} "
            f"mean={row['mean_abs_diff']:.12g}"
        )

    print()
    print(f"STATUS: {status}")
    print(f"Report: {out_path}")

    if not equivalent:
        print()
        print(
            "V2 and V4 are NOT numerically equivalent."
        )
        print(
            "Do NOT compare their HR metrics as though "
            "they use the same feature pipeline."
        )
        raise SystemExit(2)

    print()
    print(
        "V2 and V4 are numerically equivalent on "
        "the sampled windows."
    )


if __name__ == "__main__":
    main()