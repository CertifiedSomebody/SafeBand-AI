"""
BITS-2 -> SafeBand dataset ingestion adapter.

Usage:
    python tools/prepare_bits2.py --zip datasets/raw/BITS-2/full_dataset.zip \
        --out datasets/processed/bits2

Outputs:
    bits2_file_manifest.csv
    bits2_label_summary.csv
    bits2_canonical_long.csv
    bits2_audit.json
    bits2_timestamp_diagnostic.csv

Design:
- Raw BITS-2 files are never modified.
- Sensor type is read from the final CSV field.
- Embedded headers and malformed rows are skipped.
- HR <= 0 is treated as invalid.
- Timestamps are preserved exactly as released; no invented resampling.
- Subject and recording IDs are retained to prevent leakage.
"""

from __future__ import annotations
import argparse, csv, json, re, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ADL_MAP = {
    1: "walking_slowly", 2: "walking_quickly", 3: "jogging", 4: "jumping",
    5: "climbing_up_slowly", 6: "climbing_down_slowly",
    7: "climbing_up_normally", 8: "climbing_down_normally",
    9: "slowly_sitting_on_chair", 10: "rapidly_sitting_on_chair",
    11: "sitting_getting_up", 12: "swinging_hands",
    13: "lying_on_bed", 14: "lying_back_getting_up_slowly",
    15: "lying_back_getting_up_normally", 16: "sideways_to_back_lying",
}
FALL_MAP = {
    1: "forward_fall_knee", 2: "right_fall", 3: "left_fall",
    4: "forward_fall", 5: "bed_to_ground_fall",
    6: "forward_fall_hand_support", 7: "backward_fall_seated",
    8: "grabbing_while_falling",
}
SENSORS = {"acc", "acg", "gyro", "mgm", "hrt"}

def metadata(relpath: str):
    p = Path(relpath)
    parts = [x.lower() for x in p.parts]
    split = "fall" if "fall" in parts else "adl"
    m = re.search(r"user(\d+)", relpath, re.I)
    subject_id = int(m.group(1)) if m else None
    m = re.search(r"adl(\d+)", p.name, re.I)
    if split == "adl" and m:
        label_id = int(m.group(1)); label = ADL_MAP.get(label_id, f"adl_{label_id}")
    else:
        m = re.search(r"fall(\d+)", p.name, re.I)
        label_id = int(m.group(1)) if m else None
        label = FALL_MAP.get(label_id, f"fall_{label_id}") if split == "fall" else "unknown"
    return split, subject_id, label_id, label, int(split == "fall")

def parse_file(zf: zipfile.ZipFile, member: str):
    """Yield cleaned rows: sensor, timestamp, x, y, z."""
    headers = malformed = 0
    with zf.open(member, "r") as raw:
        for bline in raw:
            line = bline.decode("utf-8-sig", errors="replace").strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            nonempty = [x for x in parts if x != ""]
            sensor = nonempty[-1].lower() if nonempty else ""
            if sensor not in SENSORS:
                if parts and parts[0].lower() in {"t", "time", "timestamp"}:
                    headers += 1
                else:
                    malformed += 1
                continue
            try:
                # Drop the sensor token and any trailing empty CSV field.
                sensor_pos = max(i for i, x in enumerate(parts) if x.lower() == sensor)
                vals = [float(x) for i, x in enumerate(parts[:sensor_pos]) if x != ""]
            except ValueError:
                malformed += 1
                continue
            if sensor == "hrt":
                if len(vals) >= 2 and vals[1] > 0:
                    yield sensor, vals[0], vals[1], "", ""
                else:
                    malformed += 1
            elif len(vals) >= 4:
                yield sensor, vals[0], vals[1], vals[2], vals[3]
            else:
                malformed += 1
    return headers, malformed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to untouched BITS-2 ZIP")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    label_counts = defaultdict(lambda: {"recordings": 0, "subjects": set(),
                                        "acc": 0, "acg": 0, "gyro": 0, "mgm": 0, "hrt": 0})
    audit = Counter()
    timestamp_samples = []

    with zipfile.ZipFile(args.zip) as zf:
        members = sorted(x for x in zf.namelist()
                         if x.lower().endswith(".csv") and
                         ("/adl/" in x.lower() or "/fall/" in x.lower()))
        with open(out / "bits2_canonical_long.csv", "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["subject_id","recording_type","label_id","label","is_fall",
                             "source_file","sensor_type","timestamp_raw","x","y","z"])
            for member in members:
                split, subject, lid, label, is_fall = metadata(member)
                counts = Counter()
                first_ts = defaultdict(list)
                last_ts = defaultdict(list)
                valid = 0
                for sensor, t, x, y, z in parse_file(zf, member):
                    writer.writerow([subject, split, lid, label, is_fall,
                                     Path(member).name, sensor, t, x, y, z])
                    counts[sensor] += 1
                    valid += 1
                    if len(first_ts[sensor]) < 20: first_ts[sensor].append(t)
                    if len(last_ts[sensor]) < 20: last_ts[sensor].append(t)
                audit["csv_files"] += 1
                audit["canonical_rows"] += valid
                for s in SENSORS: audit[f"{s}_rows"] += counts[s]
                label_counts[(split,lid,label,is_fall)]["recordings"] += 1
                label_counts[(split,lid,label,is_fall)]["subjects"].add(subject)
                for s in SENSORS: label_counts[(split,lid,label,is_fall)][s] += counts[s]
                manifest_rows.append({
                    "file": member, "recording_type": split, "subject_id": subject,
                    "label_id": lid, "label": label, "is_fall": is_fall,
                    **{f"{s}_valid": counts[s] for s in SENSORS},
                })

    with open(out / "bits2_file_manifest.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(manifest_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(manifest_rows)

    with open(out / "bits2_label_summary.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["recording_type","label_id","label","is_fall","recordings","subjects",
                  "acc_rows","acg_rows","gyro_rows","mgm_rows","hrt_rows"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (split,lid,label,is_fall), d in sorted(label_counts.items()):
            w.writerow({"recording_type":split,"label_id":lid,"label":label,"is_fall":is_fall,
                        "recordings":d["recordings"],"subjects":len(d["subjects"]),
                        "acc_rows":d["acc"],"acg_rows":d["acg"],"gyro_rows":d["gyro"],
                        "mgm_rows":d["mgm"],"hrt_rows":d["hrt"]})

    with open(out / "bits2_audit.json","w",encoding="utf-8") as f:
        json.dump(dict(audit), f, indent=2)

    print(json.dumps(dict(audit), indent=2))
    print(f"Output: {out}")

if __name__ == "__main__":
    main()
