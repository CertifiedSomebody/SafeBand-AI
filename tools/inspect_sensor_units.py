"""Inspect BITS-2 and SisFall accelerometer numeric conventions.

This is intentionally a diagnostic script. It does not train a model.

It reports raw ranges and the transformations currently used by the adapters,
so a cross-dataset result can be interpreted without guessing about units.
"""

from __future__ import annotations
import argparse, csv, io, json, math, os, re, statistics, zipfile
from pathlib import Path

SISFALL_PAT = re.compile(r"^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$", re.I)

def summary(vals):
    if not vals:
        return {"count": 0}
    a = sorted(vals)
    def pct(q):
        pos = (len(a)-1)*q
        lo, hi = math.floor(pos), math.ceil(pos)
        return a[lo] if lo == hi else a[lo] + (a[hi]-a[lo])*(pos-lo)
    return {
        "count": len(vals),
        "min": min(vals),
        "p01": pct(.01),
        "p50": pct(.50),
        "p99": pct(.99),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "max_abs": max(abs(x) for x in vals),
    }

def inspect_bits2(path, limit=200000):
    vals = []
    with zipfile.ZipFile(path) as z:
        members = [
            n for n in z.namelist()
            if n.lower().endswith(".csv")
            and ("/adl/" in n.lower() or "/fall/" in n.lower())
        ]
        for name in members:
            with z.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace"):
                    parts = [x.strip() for x in raw.strip().split(",")]
                    if len(parts) < 6:
                        continue
                    sensor = next((x.lower() for x in reversed(parts) if x), "")
                    if sensor != "acc":
                        continue
                    try:
                        xyz = [float(parts[i]) for i in (1, 2, 3)]
                    except (ValueError, IndexError):
                        continue
                    vals.extend(xyz)
                    if len(vals) >= limit:
                        return summary(vals)
    return summary(vals)

def inspect_sisfall(path, sensor="adxl", limit=200000):
    vals = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            base = os.path.basename(name)
            if not SISFALL_PAT.match(base):
                continue
            with z.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    parts = [x.strip() for x in raw.strip().rstrip(";").split(",")]
                    if len(parts) != 9:
                        continue
                    try:
                        v = [int(x) for x in parts]
                    except ValueError:
                        continue
                    xyz = v[:3] if sensor == "adxl" else v[6:9]
                    vals.extend(xyz)
                    if len(vals) >= limit:
                        return summary(vals)
    return summary(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits2-zip", required=True)
    ap.add_argument("--sisfall-zip", required=True)
    ap.add_argument("--sensor", choices=["adxl", "mma"], default="adxl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {
        "bits2": inspect_bits2(args.bits2_zip),
        "sisfall_raw_counts": inspect_sisfall(args.sisfall_zip, args.sensor),
        "sisfall_conversion": {
            "sensor": args.sensor,
            "target_unit": "g",
            "adxl345_g_per_count": (2.0 * 16.0) / (2 ** 13),
            "mma8451q_g_per_count": (2.0 * 8.0) / (2 ** 14),
            "note": "BITS-2 ingestion preserves its released accelerometer numeric values; no x1000 or x9.80665 conversion is applied.",
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
