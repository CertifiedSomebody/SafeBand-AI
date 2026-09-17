from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RAW_SENSOR = ["temperature_c", "pressure_hpa", "humidity_rh", "gas_resistance_ohms"]
LABEL = "IAQ_class"


def discover(path: str | None) -> Path:
    if path:
        p = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"AIRWISE CSV not found: {p}")
        return p
    d = ROOT / "datasets" / "processed" / "airwise_bme680"
    candidates = sorted(d.glob("*.csv"))
    preferred = [p for p in candidates if p.name == "airwise_bme680_features.csv"]
    candidates = preferred + [p for p in candidates if p not in preferred]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No processed AIRWISE CSV found under {d}.")
    raise RuntimeError("Multiple processed AIRWISE CSVs found; pass --input explicitly: " + ", ".join(map(str, candidates)))


def main():
    ap = argparse.ArgumentParser(description="Audit AIRWISE labels, source columns and possible IAQ-proxy leakage.")
    ap.add_argument("--input")
    args = ap.parse_args()
    p = discover(args.input)
    d = pd.read_csv(p)

    required = {"Date", "location", LABEL, *RAW_SENSOR}
    missing = sorted(required - set(d.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    proxy = pd.to_numeric(d["IAQ_proxy"], errors="coerce") if "IAQ_proxy" in d else None

    result = {
        "path": str(p),
        "rows": int(len(d)),
        "columns": d.columns.tolist(),
        "missing_counts": d.isna().sum().to_dict(),
        "invalid_dates": int(d["Date"].isna().sum()),
        "class_counts": d[LABEL].value_counts().to_dict(),
        "locations": d["location"].value_counts().to_dict(),
        "date_min": str(d["Date"].min()),
        "date_max": str(d["Date"].max()),
        "numeric_summary": d[RAW_SENSOR].describe().to_dict(),
        "iaq_proxy_present": "IAQ_proxy" in d.columns,
    }

    if proxy is not None:
        result["iaq_proxy_summary"] = proxy.describe().to_dict()
        tmp = pd.DataFrame({"label": d[LABEL].astype(str), "proxy": proxy}).dropna()
        result["iaq_proxy_by_class"] = tmp.groupby("label")["proxy"].agg(["count", "min", "median", "max"]).to_dict(orient="index")

        # Detect whether classes have non-overlapping proxy ranges. This does not prove
        # the exact label-generation formula, but it is a strong leakage warning when true.
        ranges = tmp.groupby("label")["proxy"].agg(["min", "max"]).sort_values("min")
        labels = ranges.index.tolist()
        overlap_pairs = []
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if max(ranges.loc[a, "min"], ranges.loc[b, "min"]) <= min(ranges.loc[a, "max"], ranges.loc[b, "max"]):
                    overlap_pairs.append([a, b])
        result["proxy_range_overlap_pairs"] = overlap_pairs
        result["proxy_label_leakage_warning"] = (
            "IAQ_proxy appears in the prepared data. Its relationship to IAQ_class must be treated as label-adjacent; "
            "the primary V2 model therefore excludes IAQ_proxy and all IAQ_proxy-derived columns."
        )

    # Basic temporal integrity checks per location.
    result["duplicate_location_date_rows"] = int(d.duplicated(["location", "Date"]).sum())
    result["rows_per_location"] = d.groupby("location").size().to_dict()
    result["nonpositive_gas_resistance_rows"] = int((pd.to_numeric(d["gas_resistance_ohms"], errors="coerce") <= 0).sum())

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
