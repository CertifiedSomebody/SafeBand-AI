"""
SafeBand-AI — PPG-DaLiA forensic preflight.

Purpose
-------
Validate the exact prepared PPG-DaLiA artifact before doing any V2/V4
feature comparison.

This script does NOT train anything.

Checks
------
1. Required NPZ arrays exist.
2. Shapes match the established PPG-DaLiA window contract.
3. All numerical arrays are finite.
4. Subject IDs are valid.
5. (subject, window_index) pairs are unique.
6. Deterministic 60/20/20 subject split is non-overlapping.
7. Basic HR range diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "bvp",
    "acc",
    "hr",
    "subject",
    "window_index",
}

BVP_SAMPLES = 512          # 8 s × 64 Hz
ACC_SAMPLES = 256          # 8 s × 32 Hz
ACC_AXES = 3


def deterministic_subject_split(subjects: list[str]) -> dict[str, list[str]]:
    """
    Reproduce the established 60/20/20 subject ordering.

    The prepared artifact uses subject IDs such as S1, S2, ..., S15.
    We sort explicitly and then split deterministically.
    """
    subjects = sorted(subjects, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    n = len(subjects)

    n_train = round(n * 0.60)
    n_val = round(n * 0.20)

    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    return {
        "train_subjects": subjects[:n_train],
        "val_subjects": subjects[n_train:n_train + n_val],
        "test_subjects": subjects[n_train + n_val:],
    }


def assert_no_subject_overlap(split: dict[str, list[str]]) -> None:
    names = list(split.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = set(split[names[i]])
            b = set(split[names[j]])

            overlap = a & b

            if overlap:
                raise AssertionError(
                    f"Subject leakage detected between "
                    f"{names[i]} and {names[j]}: {sorted(overlap)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="datasets/processed/ppgdalia_windows.npz",
        help="Prepared PPG-DaLiA NPZ.",
    )

    parser.add_argument(
        "--out",
        default="models/ppgdalia_forensic_preflight.json",
        help="Output JSON report.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found:\n{input_path}"
        )

    print("=" * 72)
    print("SafeBand-AI PPG-DaLiA FORENSIC PREFLIGHT")
    print("=" * 72)

    data = np.load(input_path, allow_pickle=False)

    print(f"Input: {input_path}")
    print(f"Arrays: {data.files}")

    missing = REQUIRED_ARRAYS - set(data.files)

    if missing:
        raise ValueError(
            f"Missing required arrays: {sorted(missing)}"
        )

    bvp = np.asarray(data["bvp"])
    acc = np.asarray(data["acc"])
    hr = np.asarray(data["hr"])
    subject = np.asarray(data["subject"])
    window_index = np.asarray(data["window_index"])

    n = len(hr)

    # ------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------

    expected_bvp_shape = (n, BVP_SAMPLES)
    expected_acc_shape = (n, ACC_SAMPLES, ACC_AXES)

    if bvp.shape != expected_bvp_shape:
        raise ValueError(
            f"Invalid BVP shape: {bvp.shape}; "
            f"expected {expected_bvp_shape}"
        )

    if acc.shape != expected_acc_shape:
        raise ValueError(
            f"Invalid ACC shape: {acc.shape}; "
            f"expected {expected_acc_shape}"
        )

    if len(subject) != n:
        raise ValueError("subject length does not match number of windows.")

    if len(window_index) != n:
        raise ValueError(
            "window_index length does not match number of windows."
        )

    # ------------------------------------------------------------------
    # Finite-value validation
    # ------------------------------------------------------------------

    for name, arr in (
        ("bvp", bvp),
        ("acc", acc),
        ("hr", hr),
    ):
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} contains NaN or infinite values."
            )

    # ------------------------------------------------------------------
    # Subject diagnostics
    # ------------------------------------------------------------------

    subjects = sorted(
        {str(x) for x in subject},
        key=lambda x: int(x[1:]) if x[1:].isdigit() else x,
    )

    if not subjects:
        raise ValueError("No subjects found.")

    windows_by_subject = {
        s: int(np.sum(subject.astype(str) == s))
        for s in subjects
    }

    split = deterministic_subject_split(subjects)

    assert_no_subject_overlap(split)

    # ------------------------------------------------------------------
    # (subject, window_index) uniqueness
    # ------------------------------------------------------------------

    pair_keys = [
        (str(subject[i]), int(window_index[i]))
        for i in range(n)
    ]

    duplicate_count = len(pair_keys) - len(set(pair_keys))

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"(subject, window_index) keys."
        )

    # ------------------------------------------------------------------
    # HR diagnostics
    # ------------------------------------------------------------------

    hr_min = float(np.min(hr))
    hr_max = float(np.max(hr))
    hr_mean = float(np.mean(hr))
    hr_std = float(np.std(hr))

    report = {
        "status": "PASS",
        "input": str(input_path),
        "array_shapes": {
            name: list(data[name].shape)
            for name in data.files
        },
        "windows_total": n,
        "subjects_total": len(subjects),
        "subjects": subjects,
        "windows_by_subject": windows_by_subject,
        "split": split,
        "subject_overlap_check": "PASS",
        "duplicate_subject_window_keys": duplicate_count,
        "hr_bpm": {
            "min": hr_min,
            "max": hr_max,
            "mean": hr_mean,
            "std": hr_std,
        },
        "contract": {
            "bvp_samples": BVP_SAMPLES,
            "bvp_fs_hz": 64,
            "acc_samples": ACC_SAMPLES,
            "acc_axes": ACC_AXES,
            "acc_fs_hz": 32,
            "window_seconds": 8,
            "shift_seconds": 2,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Windows: {n}")
    print(f"Subjects: {len(subjects)}")
    print(f"HR range: {hr_min:.3f} – {hr_max:.3f} BPM")
    print()
    print("TRAIN SUBJECTS:")
    print(split["train_subjects"])
    print()
    print("VALIDATION SUBJECTS:")
    print(split["val_subjects"])
    print()
    print("TEST SUBJECTS:")
    print(split["test_subjects"])
    print()
    print(f"Duplicate subject/window keys: {duplicate_count}")
    print()
    print(f"STATUS: PASS")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()