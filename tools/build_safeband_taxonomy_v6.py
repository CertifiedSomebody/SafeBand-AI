from pathlib import Path
import sys, argparse, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_MAP = {
    "LyingDown": "LYING",
    "Sitting": "SITTING",
    "Standing": "STANDING",
    "Walking": "WALKING",
    "Running": "RUNNING",
    "Upstairs": "STAIRS",
    "Downstairs": "STAIRS",
    "RampUp": "TRANSITION",
    "RampDown": "TRANSITION",
    "LiftUp": "TRANSITION",
    "LiftDown": "TRANSITION",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    X = z["X"]
    m = pd.read_csv(args.manifest)
    if len(X) != len(m):
        raise ValueError(f"X rows ({len(X)}) != manifest rows ({len(m)})")
    if "activity" not in m:
        raise ValueError("Manifest must contain 'activity'")

    m = m.copy()
    m["safeband_activity"] = m["activity"].map(CLASS_MAP)
    if m["safeband_activity"].isna().any():
        bad = sorted(m.loc[m["safeband_activity"].isna(), "activity"].unique())
        raise ValueError(f"Unmapped activities: {bad}")

    keep = m["fold"].astype(str).eq("test") | m["fold"].astype(str).isin(["1","2","3","4","5"])
    X = X[keep.to_numpy()]
    m = m.loc[keep].reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X.astype(np.float32))
    manifest_out = out.with_suffix(".csv")
    m.to_csv(manifest_out, index=False)

    print(json.dumps({
        "windows": int(len(X)),
        "shape": list(X.shape),
        "classes": sorted(m["safeband_activity"].unique()),
        "class_counts": {k:int(v) for k,v in m["safeband_activity"].value_counts().sort_index().items()},
        "test_windows": int((m["fold"].astype(str)=="test").sum()),
        "train_dev_windows": int((m["fold"].astype(str)!="test").sum()),
        "output_npz": str(out),
        "output_manifest": str(manifest_out),
    }, indent=2))

if __name__ == "__main__":
    main()
