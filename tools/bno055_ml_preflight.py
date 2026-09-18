
"""Preflight checker for the BNO055 ML benchmark."""
from pathlib import Path
import argparse, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_bno055_ml import load_windows, CHANNEL_SETS, CLASS_NAMES

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=ROOT/"datasets/synthetic/bno055/bno055_windows_v2_1.npz")
    args = p.parse_args()
    X,y,g,s,e,labels = load_windows(args.data)
    print(f"[PASS] NPZ: {args.data}")
    print(f"[PASS] X shape: {X.shape}")
    print(f"[PASS] y shape: {y.shape}")
    print(f"[PASS] subjects: {len(np.unique(g))}")
    print(f"[PASS] classes: {labels}")
    print(f"[PASS] event-overlap windows: {int(e.sum())}")
    for name, idx in CHANNEL_SETS.items():
        print(f"[PASS] channel set {name}: {len(idx)} channels")
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    if len(counts) != 9 or np.any(counts == 0):
        raise ValueError(f"Invalid class counts: {counts.tolist()}")
    print(f"[PASS] class counts: {counts.tolist()}")
    print("[PASS] BNO055 ML preflight completed.")

if __name__ == "__main__":
    main()
