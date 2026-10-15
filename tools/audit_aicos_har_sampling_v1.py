"""
AICOS-HAR bounded sensor sampling audit V1.

After path reconciliation, this is the first script that reads sensor samples.
It intentionally audits a bounded number of acquisitions/sensors first.

It reports timestamp-derived rate, duration, row count, monotonicity and
non-finite values. It does not create ML windows and does not train models.
"""
from pathlib import Path
import argparse, csv, math, statistics, sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_aicos_har_v2 import discover_sensor_files

def read_sensor(path):
    # AICOS-HAR uses | separators. Skip malformed/header lines defensively.
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = [x.strip() for x in s.split("|")]
            if len(parts) < 2:
                continue
            try:
                t = int(float(parts[0]))
                vals = [float(x) for x in parts[1:]]
                rows.append((t, vals))
            except ValueError:
                continue
    return rows

def rate_stats(rows):
    if len(rows) < 3:
        return None
    ts = [x[0] for x in rows]
    diffs = [(b-a)/1e9 for a,b in zip(ts, ts[1:]) if b > a]
    if not diffs:
        return None
    med_dt = statistics.median(diffs)
    rate = 1.0 / med_dt if med_dt > 0 else float("nan")
    return {
        "rows": len(rows),
        "duration_s": (ts[-1]-ts[0])/1e9,
        "median_rate_hz": rate,
        "min_dt_s": min(diffs),
        "max_dt_s": max(diffs),
        "nonmonotonic": sum(1 for a,b in zip(ts,ts[1:]) if b <= a),
        "nonfinite_values": sum(
            1 for _, vals in rows for v in vals if not math.isfinite(v)
        ),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--max-acquisitions", type=int, default=30)
    ap.add_argument("--out", default="reports/aicos_har_sampling_audit_v1")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    all_files = discover_sensor_files(data_dir)
    by_acq = defaultdict(list)
    for r in all_files:
        by_acq[r["acquisition_id"]].append(r)

    # Deterministic, spread-out sample: first N unique acquisitions.
    selected_ids = sorted(by_acq)[:args.max_acquisitions]

    results = []
    for aid in selected_ids:
        for meta in sorted(by_acq[aid], key=lambda x: x["sensor"]):
            rows = read_sensor(meta["path"])
            st = rate_stats(rows)
            if st is None:
                continue
            results.append({
                "acquisition_id": aid,
                "sensor": meta["sensor"],
                "relative_path": meta["relative_path"],
                **st,
            })

    with open(out / "sampling_audit.csv", "w", encoding="utf-8", newline="") as f:
        fields = ["acquisition_id","sensor","relative_path","rows","duration_s",
                  "median_rate_hz","min_dt_s","max_dt_s","nonmonotonic","nonfinite_values"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    by_sensor = defaultdict(list)
    for r in results:
        by_sensor[r["sensor"]].append(r["median_rate_hz"])

    print("AICOS-HAR BOUNDED SAMPLING AUDIT V1")
    print("=" * 40)
    print(f"Acquisitions sampled: {len(selected_ids)}")
    print(f"Sensor files audited: {len(results)}")
    for sensor in sorted(by_sensor):
        vals = by_sensor[sensor]
        print(f"{sensor:>5}: n={len(vals):3d}, "
              f"median={statistics.median(vals):.3f} Hz, "
              f"min={min(vals):.3f}, max={max(vals):.3f}")
    print(f"Report: {out / 'sampling_audit.csv'}")

if __name__ == "__main__":
    main()
