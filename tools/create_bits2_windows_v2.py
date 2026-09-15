"""
SafeBand AI - BITS-2 Window Builder v2

Creates recording-contained accelerometer windows. Windows never cross
source files, subjects, labels, or recording types.

Default:
  20 Hz nominal acquisition rate (per BITS-2 publication)
  2.0 s windows = 40 samples
  50% overlap

The timestamp field is retained only as provenance; no timestamp-based
resampling or cross-sensor synchronization is performed.
"""

from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.bits2_features import extract_motion_features, FEATURE_COLUMNS


# BITS-2 ADL IDs
ACTIVITY_MAP = {
    1: "WALKING", 2: "WALKING", 3: "RUNNING",
    9: "SITTING", 10: "SITTING", 11: "SITTING",
    13: "RESTING", 14: "RESTING", 15: "RESTING", 16: "RESTING",
}
# All other ADLs are deliberately excluded from this first clean
# four-class activity task rather than forcing ambiguous labels.

def label_for(recording_type: str, label_id: int) -> Tuple[str, str]:
    if recording_type == "fall":
        return "FALL", "fall"
    label = ACTIVITY_MAP.get(label_id)
    return (label, "activity") if label else ("", "")

def flush_recording(
    samples: List[Tuple[float,float,float]],
    meta: Dict[str,str],
    writer: csv.DictWriter,
    window_samples: int,
    stride: int,
) -> int:
    label, task = label_for(meta["recording_type"], int(meta["label_id"]))
    if not label or len(samples) < window_samples:
        return 0

    count = 0
    for start in range(0, len(samples)-window_samples+1, stride):
        window = samples[start:start+window_samples]
        f = extract_motion_features(window)
        row = {
            "window_id": f'{meta["subject_id"]}_{Path(meta["source_file"]).stem}_{start}',
            "subject_id": meta["subject_id"],
            "source_file": meta["source_file"],
            "recording_type": meta["recording_type"],
            "label_id": meta["label_id"],
            "label": label,
            "task": task,
            "start_sample": start,
            "end_sample": start + window_samples - 1,
            "window_samples": window_samples,
            **{k:f[k] for k in FEATURE_COLUMNS},
        }
        writer.writerow(row)
        count += 1
    return count

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-samples", type=int, default=40)
    ap.add_argument("--overlap", type=float, default=0.50)
    args = ap.parse_args()

    if not (0 <= args.overlap < 1):
        raise SystemExit("--overlap must be in [0,1)")
    stride = max(1, int(round(args.window_samples*(1-args.overlap))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_id","subject_id","source_file","recording_type","label_id",
        "label","task","start_sample","end_sample","window_samples",
        *FEATURE_COLUMNS
    ]

    total = 0
    recordings = 0
    current_key = None
    samples: List[Tuple[float,float,float]] = []
    meta: Dict[str,str] = {}

    with open(args.input, newline="", encoding="utf-8") as fin, \
         open(out, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for r in reader:
            if r.get("sensor_type") != "acc":
                continue
            key = r["source_file"]
            if current_key is not None and key != current_key:
                total += flush_recording(samples, meta, writer, args.window_samples, stride)
                recordings += 1
                samples = []
            if key != current_key:
                current_key = key
                meta = {
                    "subject_id": str(r["subject_id"]),
                    "source_file": r["source_file"],
                    "recording_type": r["recording_type"],
                    "label_id": str(r["label_id"]),
                }
            try:
                samples.append((float(r["x"]),float(r["y"]),float(r["z"])))
            except (TypeError, ValueError):
                continue

        if current_key is not None:
            total += flush_recording(samples, meta, writer, args.window_samples, stride)
            recordings += 1

    print(json.dumps({
        "output": str(out),
        "window_samples": args.window_samples,
        "overlap": args.overlap,
        "stride": stride,
        "recordings_seen": recordings,
        "windows_written": total,
    }, indent=2))

if __name__ == "__main__":
    main()
