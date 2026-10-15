
"""AICOS-HAR path reconciliation V3.

This fixes the V2 parser defect: AICOS-HAR uses sensor filenames such as
accelerometer.txt/gyroscope.txt/magnetometer.txt/barometer.txt, not necessarily
acc.txt/gyro.txt/mag.txt/baro.txt.

This script reads paths only and never parses sensor samples.
"""
from pathlib import Path
import argparse, csv, json, sys
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_aicos_har_v3 import discover_sensor_files

def read_csv(p):
    with open(p, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", default="reports/aicos_har_path_reconciliation_v3")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    metadata = read_csv(Path(args.metadata))
    inventory = read_csv(Path(args.inventory))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = {r["AcquisitionID"].strip() for r in metadata}
    inv = {r["AcquisitionID"].strip() for r in inventory}

    raw_rows = discover_sensor_files(data_dir)
    raw = defaultdict(list)
    for r in raw_rows:
        raw[r["acquisition_id"]].append(r)

    raw_ids = set(raw)
    missing = sorted(meta - raw_ids)
    raw_only = sorted(raw_ids - meta)

    sensor_counts = Counter(r["sensor"] for r in raw_rows)
    records = []
    for aid in sorted(meta):
        rr = raw.get(aid, [])
        records.append({
            "AcquisitionID": aid,
            "raw_found": int(bool(rr)),
            "sensor_file_count": len(rr),
            "sensors_found": ";".join(sorted({x["sensor"] for x in rr})),
            "raw_paths": ";".join(sorted(x["relative_path"] for x in rr)),
        })

    with open(out / "aicos_har_raw_inventory_v3.csv", "w", encoding="utf-8", newline="") as f:
        fields = list(records[0]) if records else [
            "AcquisitionID","raw_found","sensor_file_count","sensors_found","raw_paths"
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)

    report = {
        "metadata_unique": len(meta),
        "inventory_unique": len(inv),
        "raw_unique_acquisitions": len(raw_ids),
        "raw_sensor_files": len(raw_rows),
        "exact_metadata_to_raw_matches": len(meta & raw_ids),
        "metadata_without_raw": len(meta - raw_ids),
        "raw_only": len(raw_ids - meta),
        "sensor_file_counts": dict(sensor_counts),
        "missing_examples": missing[:50],
        "raw_only_examples": raw_only[:50],
    }
    (out / "aicos_har_path_reconciliation_v3.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("AICOS-HAR PATH RECONCILIATION V3")
    print("=" * 40)
    for k,v in report.items():
        if k.endswith("examples"): continue
        print(f"{k:35}: {v}")
    print("Sensor counts:")
    for k,v in sorted(sensor_counts.items()):
        print(f"  {k:8}: {v}")
    print(f"Report: {out}")

if __name__ == "__main__":
    main()
