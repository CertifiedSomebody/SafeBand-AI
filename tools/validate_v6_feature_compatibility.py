"""Validate the V6 feature contract and compare BITS-2/SisFall feature distributions.

The script accepts already-generated feature CSVs. It never changes the model
and never tunes a threshold.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

def stats(df, cols):
    rows = []
    for c in cols:
        a = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        finite = a[np.isfinite(a)]
        if len(finite) == 0:
            rows.append({"feature": c, "count": 0, "nan_or_inf": int(len(a))})
            continue
        rows.append({
            "feature": c,
            "count": int(len(finite)),
            "nan_or_inf": int(len(a) - len(finite)),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "p01": float(np.quantile(finite, .01)),
            "p50": float(np.quantile(finite, .50)),
            "p99": float(np.quantile(finite, .99)),
            "max": float(np.max(finite)),
            "constant": bool(np.ptp(finite) == 0),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits2", required=True)
    ap.add_argument("--sisfall", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = joblib.load(args.model)
    model = payload["model"]
    cols = list(payload["feature_columns"])

    b = pd.read_csv(args.bits2, usecols=lambda c: c in cols)
    s = pd.read_csv(args.sisfall, usecols=lambda c: c in cols)

    missing_bits2 = [c for c in cols if c not in b.columns]
    missing_sisfall = [c for c in cols if c not in s.columns]

    result = {
        "model": args.model,
        "model_classes": [str(c) for c in model.classes_],
        "feature_count": len(cols),
        "feature_order_exact": (
            list(b.columns) == cols and list(s.columns) == cols
        ),
        "missing_in_bits2": missing_bits2,
        "missing_in_sisfall": missing_sisfall,
        "bits2_rows": int(len(b)),
        "sisfall_rows": int(len(s)),
    }

    if missing_bits2 or missing_sisfall:
        result["status"] = "FAIL_SCHEMA"
    else:
        result["status"] = "OK"
        bs = stats(b, cols)
        ss = stats(s, cols)
        s_by = {r["feature"]: r for r in ss}
        b_by = {r["feature"]: r for r in bs}

        shifts = []
        for c in cols:
            bm, bsig = b_by[c]["mean"], b_by[c]["std"]
            sm, ssig = s_by[c]["mean"], s_by[c]["std"]
            pooled = max(1e-12, (bsig + ssig) / 2.0)
            shifts.append({
                "feature": c,
                "bits2_mean": bm,
                "sisfall_mean": sm,
                "mean_delta": sm - bm,
                "mean_shift_in_pooled_sd": (sm - bm) / pooled,
                "bits2_std": bsig,
                "sisfall_std": ssig,
                "std_ratio_sisfall_over_bits2": ssig / max(abs(bsig), 1e-12),
            })

        result["bits2_stats"] = bs
        result["sisfall_stats"] = ss
        result["largest_mean_shifts"] = sorted(
            shifts,
            key=lambda x: abs(x["mean_shift_in_pooled_sd"]),
            reverse=True,
        )[:20]
        result["constant_features"] = {
            "bits2": [r["feature"] for r in bs if r.get("constant")],
            "sisfall": [r["feature"] for r in ss if r.get("constant")],
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
