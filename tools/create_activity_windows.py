"""Create leakage-safe fixed-length BITS-2 activity windows.

This stage deliberately uses ACCELEROMETER sample order, not the released
floating-point timestamps. The BITS-2 timestamps lose sub-second precision
and are not reliable enough for cross-sensor resampling.

Default: 2-second windows at the documented ~20 Hz motion rate, 50% overlap.
Only activity labels that can be mapped without semantic invention are kept.
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from ai.feature_extraction import FEATURE_COLUMNS, extract_features

# Safe, semantically defensible first-task mapping.
# Other BITS-2 ADLs are retained in the raw/canonical dataset but excluded
# from this first model rather than assigning them a false class.
LABEL_MAP = {
    "walking_slowly": "WALKING",
    "walking_quickly": "WALKING",
    "jogging": "RUNNING",
    "slowly_sitting_on_chair": "SITTING",
    "rapidly_sitting_on_chair": "SITTING",
    "lying_on_bed": "RESTING",
    "forward_fall_knee": "FALL",
    "right_fall": "FALL",
    "left_fall": "FALL",
    "forward_fall": "FALL",
    "bed_to_ground_fall": "FALL",
    "forward_fall_hand_support": "FALL",
    "backward_fall_seated": "FALL",
    "grabbing_while_falling": "FALL",
}


def _flush_recording(rows: List[Dict], out_rows: List[Dict], window_size: int, stride: int) -> None:
    if not rows:
        return
    label = LABEL_MAP.get(rows[0]["label"])
    if label is None:
        return
    # Use only accelerometer rows. This avoids pretending that the lower-rate
    # HR/magnetometer streams are sample-synchronous with ACC.
    acc = [r for r in rows if r["sensor_type"] == "acc"]
    if len(acc) < window_size:
        return
    subject = rows[0]["subject_id"]
    session = rows[0].get("session_id", rows[0].get("source_file", ""))
    source = rows[0]["source_file"]
    recording_type = rows[0]["recording_type"]
    for start in range(0, len(acc) - window_size + 1, stride):
        w = acc[start:start + window_size]
        samples = [{"acceleration_x": r["x"], "acceleration_y": r["y"], "acceleration_z": r["z"]} for r in w]
        feats = extract_features(samples)
        out_rows.append({
            "subject_id": subject,
            "session_id": session,
            "source_file": source,
            "recording_type": recording_type,
            "activity_label": label,
            "window_start_sample": start,
            "window_end_sample": start + window_size - 1,
            "window_samples": window_size,
            **feats,
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="bits2_canonical_long.csv")
    ap.add_argument("--out", required=True, help="activity_windows.csv")
    ap.add_argument("--window-samples", type=int, default=40)
    ap.add_argument("--stride-samples", type=int, default=20)
    args = ap.parse_args()
    if args.window_samples < 4 or args.stride_samples < 1:
        raise SystemExit("window-samples must be >=4 and stride-samples >=1")

    groups: Dict[str, List[Dict]] = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = r.get("label", "")
            if label not in LABEL_MAP:
                continue
            if r.get("sensor_type") != "acc":
                continue
            try:
                r["x"] = float(r["x"]); r["y"] = float(r["y"]); r["z"] = float(r["z"])
            except (TypeError, ValueError):
                continue
            # source_file is unique within a subject in BITS-2.
            key = f'{r.get("subject_id")}::{r.get("source_file")}'
            groups[key].append(r)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id", "session_id", "source_file", "recording_type", "activity_label",
        "window_start_sample", "window_end_sample", "window_samples", *FEATURE_COLUMNS,
    ]
    total = 0
    label_counts = defaultdict(int)
    subject_counts = defaultdict(set)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for rows in groups.values():
            temp: List[Dict] = []
            _flush_recording(rows, temp, args.window_samples, args.stride_samples)
            for row in temp:
                w.writerow(row); total += 1
                label_counts[row["activity_label"]] += 1
                subject_counts[row["activity_label"]].add(row["subject_id"])

    audit = {
        "windows": total,
        "window_samples": args.window_samples,
        "stride_samples": args.stride_samples,
        "assumed_motion_rate_hz": 20,
        "included_labels": sorted(label_counts),
        "excluded_label_policy": "Exclude semantically ambiguous BITS-2 ADLs from first model; do not relabel them.",
        "windows_by_label": dict(sorted(label_counts.items())),
        "subjects_by_label": {k: len(v) for k, v in sorted(subject_counts.items())},
    }
    audit_path = out_path.with_name(out_path.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
