
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
import numpy as np
import pandas as pd


def pick_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    keys = list(z.keys())

    arrays = {}
    for k in keys:
        a = z[k]
        if isinstance(a, np.ndarray):
            arrays[k] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "nan": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
                "inf": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
            }

    X_key = None
    for k in keys:
        a = z[k]
        if isinstance(a, np.ndarray) and a.ndim == 3 and np.issubdtype(a.dtype, np.number):
            X_key = k
            break
    if X_key is None:
        raise RuntimeError(f"Could not find a numeric 3-D window array. NPZ keys: {keys}")

    X = np.asarray(z[X_key], dtype=np.float32)
    if X.shape[1] not in (3, 4) and X.shape[2] in (3, 4):
        X = np.transpose(X, (0, 2, 1))
    if X.shape[1] not in (3, 4):
        raise RuntimeError(f"Expected 3/4 channels, got {X.shape}")

    df = pd.read_csv(args.manifest)
    label_col = pick_column(df, ["activity", "activity_label", "label", "class", "Activity"])
    acq_col = pick_column(df, ["acquisition_id", "AcquisitionID", "acquisition", "trial_id"])
    participant_col = pick_column(df, ["participant_id", "ParticipantID", "participant", "subject_id", "subject"])
    fold_col = pick_column(df, ["fold", "Fold", "partition", "split"])

    report = {
        "npz_keys": arrays,
        "window_array_key": X_key,
        "window_shape": list(X.shape),
        "window_nan_values": int(np.isnan(X).sum()),
        "window_inf_values": int(np.isinf(X).sum()),
        "windows_with_nonfinite": int(np.any(~np.isfinite(X), axis=tuple(range(1, X.ndim))).sum()),
        "channel_zero_std_windows": {},
        "manifest_rows": int(len(df)),
        "manifest_columns": list(df.columns),
        "resolved_columns": {
            "label": label_col,
            "acquisition": acq_col,
            "participant": participant_col,
            "fold": fold_col,
        },
    }

    for c in range(X.shape[1]):
        s = np.std(X[:, c, :], axis=1)
        report["channel_zero_std_windows"][str(c)] = int(np.count_nonzero(s <= 1e-8))

    # Correlation-feature risk: a correlation coefficient is undefined if either
    # axis has zero variance. Count these windows explicitly.
    zero_axes = np.std(X[:, :3, :], axis=2) <= 1e-8
    report["corrcoef_undefined_windows"] = int(np.any(zero_axes, axis=1).sum())
    report["corrcoef_undefined_fraction"] = float(report["corrcoef_undefined_windows"] / max(1, len(X)))

    if label_col:
        report["class_counts"] = {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).items()}
    if fold_col:
        report["fold_counts"] = {str(k): int(v) for k, v in df[fold_col].value_counts(dropna=False).items()}
    if participant_col:
        report["unique_participants"] = int(df[participant_col].nunique(dropna=True))
    if acq_col:
        report["unique_acquisitions"] = int(df[acq_col].nunique(dropna=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nAudit report: {out}")
    print("PASS: raw windows contain no non-finite values." if report["window_nan_values"] == 0 and report["window_inf_values"] == 0
          else "FAIL: raw windows contain NaN/Inf values.")
    print(f"Correlation-risk windows: {report['corrcoef_undefined_windows']} / {len(X)}")


if __name__ == "__main__":
    main()
