from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import joblib
import numpy as np

from ai.ppg_signal import estimate_hr_bvp_only
from ai.ppg_motion_features import combined_quality_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    names = artifact["feature_names"]
    bvp_fs = float(artifact["bvp_fs_hz"])
    acc_fs = float(artifact["acc_fs_hz"])

    z = np.load(args.input, allow_pickle=True)
    bvp = z["bvp"]
    acc = z["acc"]

    if bvp.ndim != 2 or acc.ndim != 3 or len(bvp) != len(acc):
        raise ValueError("Invalid BVP/ACC arrays.")

    X = []
    for i in range(len(bvp)):
        f = combined_quality_features(bvp[i], acc[i], bvp_fs, acc_fs)
        f.update(estimate_hr_bvp_only(bvp[i], bvp_fs))
        X.append([f[k] for k in names])

    X = np.asarray(X, dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError("Inference features contain NaN/Inf.")

    pred = model.predict(X)
    print(f"Predicted HR: mean={np.mean(pred):.2f} BPM")
    print(f"Predicted HR: min={np.min(pred):.2f} BPM max={np.max(pred):.2f} BPM")


if __name__ == "__main__":
    main()
