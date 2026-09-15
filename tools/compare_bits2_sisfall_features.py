"""BITS-2 vs SisFall feature distribution comparison.

Use this after both datasets have been prepared with the same 84-feature V6
contract. No model fitting or threshold tuning occurs here.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

META = {
    "subject_id", "source_file", "activity_code", "recording_type",
    "trial", "start_sample", "window_samples",
    "window_id", "label_id", "label", "end_sample",
}

def feature_stats(df, cols):
    out = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            out[c] = {"count": 0}
            continue
        out[c] = {
            "count": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(np.min(x)),
            "p01": float(np.quantile(x, .01)),
            "p50": float(np.quantile(x, .50)),
            "p99": float(np.quantile(x, .99)),
            "max": float(np.max(x)),
            "constant": bool(np.ptp(x) == 0),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits2", required=True)
    ap.add_argument("--sisfall", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b = pd.read_csv(args.bits2)
    s = pd.read_csv(args.sisfall)

    bcols = set(b.columns) - META
    scols = set(s.columns) - META
    common = sorted(c for c in bcols & scols)

    bs = feature_stats(b, common)
    ss = feature_stats(s, common)

    rows = []
    for c in common:
        bm, sm = bs[c].get("mean"), ss[c].get("mean")
        bsd, ssd = bs[c].get("std"), ss[c].get("std")
        pooled = max(1e-12, (bsd + ssd) / 2.0)
        rows.append({
            "feature": c,
            "bits2_mean": bm,
            "sisfall_mean": sm,
            "mean_delta": sm - bm,
            "mean_shift_pooled_sd": (sm - bm) / pooled,
            "bits2_std": bsd,
            "sisfall_std": ssd,
            "std_ratio": ssd / max(abs(bsd), 1e-12),
        })

    rows.sort(key=lambda r: abs(r["mean_shift_pooled_sd"]), reverse=True)

    result = {
        "bits2": args.bits2,
        "sisfall": args.sisfall,
        "bits2_rows": int(len(b)),
        "sisfall_rows": int(len(s)),
        "common_feature_count": len(common),
        "missing_from_bits2": sorted(scols - bcols),
        "missing_from_sisfall": sorted(bcols - scols),
        "bits2_stats": bs,
        "sisfall_stats": ss,
        "largest_distribution_shifts": rows[:30],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
