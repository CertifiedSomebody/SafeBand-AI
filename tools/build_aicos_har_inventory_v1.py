"""
Second-pass audit for AICOS-HAR after the raw inventory is available.

This does NOT train a model. It builds a conservative acquisition-level table
that can later be used to design a subject-independent benchmark.

Usage:
    python tools/build_aicos_har_inventory_v1.py ^
        --data-dir datasets/raw/AICOS-HAR/AICOS-HAR ^
        --metadata datasets/raw/AICOS-HAR/AICOS-HAR/metadata.csv ^
        --out datasets/processed/aicos_har/aicos_har_acquisition_inventory.csv
"""

from pathlib import Path
import sys
import argparse
import csv
import re
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def norm(x):
    return re.sub(r"\s+", " ", str(x).strip())


def normalize_header(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).strip().lower())


def split_acq(aid):
    p = norm(aid).split("/")
    return {
        "participant_id": p[0] if len(p) > 0 else "",
        "activity_trial_id": p[1] if len(p) > 1 else "",
        "device_position_id": p[2] if len(p) > 2 else "",
    }

def sensor_name(path):
    return path.stem.lower()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    metadata = Path(args.metadata).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with metadata.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise SystemExit("ERROR: metadata.csv contains no records.")

    headers = list(rows[0])
    acq_col = next(
        (c for c in headers if c.strip().lower().replace("_", "") == "acquisitionid"),
        None,
    )
    if acq_col is None:
        raise SystemExit(f"ERROR: no AcquisitionID column. Found: {headers}")

    fold_col = next((c for c in headers if c.lower() == "fold"), "")
    pos_col = next((c for c in headers if "position" in c.lower()), "")

    sensor_map = defaultdict(set)
    file_map = defaultdict(list)

    for p in data_dir.rglob("*.txt"):
        if p.name.lower() in {"readme.txt", "metadata.txt"}:
            continue
        rel = p.relative_to(data_dir)
        if len(rel.parts) >= 4:
            aid = "/".join(rel.parts[:3])
            sensor_map[aid].add(sensor_name(p))
            file_map[aid].append(str(rel))

    output = []
    for r in rows:
        aid = norm(r.get(acq_col, ""))
        parts = split_acq(aid)
        rec = dict(r)
        rec.update(parts)
        rec["sensors_found"] = ";".join(sorted(sensor_map.get(aid, set())))
        rec["sensor_file_count"] = len(file_map.get(aid, []))
        rec["raw_found"] = int(aid in sensor_map)
        output.append(rec)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(f"Inventory written: {out}")
    print(f"Rows: {len(output)}")
    print(f"Participants: {len(set(x['participant_id'] for x in output))}")
    print(f"Activities/trials: {len(set(x['activity_trial_id'] for x in output))}")
    print(f"Acquisitions with raw files: {sum(x['raw_found'] for x in output)}")

if __name__ == "__main__":
    main()
