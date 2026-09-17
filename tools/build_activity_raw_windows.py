"""Build raw 2-second BITS2 ACC windows for deep-learning activity recognition.

Input: datasets/processed/bits2/bits2_canonical_long.csv
Output: datasets/processed/bits2/activity_raw_windows.npz

The same semantic mapping and 40-sample/20-sample window contract used by the
existing ACC activity pipeline are retained. Windows never cross recordings.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABEL_MAP = {
    "walking_slowly": "WALKING", "walking_quickly": "WALKING",
    "jogging": "RUNNING",
    "slowly_sitting_on_chair": "SITTING", "rapidly_sitting_on_chair": "SITTING",
    "lying_on_bed": "RESTING",
}
CLASSES = ["RESTING", "SITTING", "WALKING", "RUNNING"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="datasets/processed/bits2/bits2_canonical_long.csv")
    ap.add_argument("--out", default="datasets/processed/bits2/activity_raw_windows.npz")
    ap.add_argument("--window-samples", type=int, default=40)
    ap.add_argument("--stride-samples", type=int, default=20)
    args = ap.parse_args()
    if args.window_samples < 8 or args.stride_samples < 1:
        raise SystemExit("invalid window/stride")

    groups = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            label = LABEL_MAP.get(r.get("label", ""))
            if label is None or r.get("sensor_type") != "acc":
                continue
            try:
                xyz = [float(r["x"]), float(r["y"]), float(r["z"])]
            except (TypeError, ValueError, KeyError):
                continue
            key = (str(r.get("subject_id", "")), str(r.get("source_file", "")))
            groups[key].append((xyz, label))

    X, y, subjects, sources = [], [], [], []
    for (subject, source), rows in sorted(groups.items()):
        if not rows:
            continue
        labels = [z[1] for z in rows]
        # A source recording is expected to have one activity label. If not,
        # only keep windows whose local majority label is pure.
        for start in range(0, len(rows) - args.window_samples + 1, args.stride_samples):
            w = rows[start:start + args.window_samples]
            counts = Counter(z[1] for z in w)
            label, count = counts.most_common(1)[0]
            if count / args.window_samples < 0.90:
                continue
            arr = np.asarray([z[0] for z in w], dtype=np.float32)
            X.append(arr.T)  # channels x time = 3 x 40
            y.append(CLASSES.index(label)); subjects.append(subject); sources.append(source)

    if not X:
        raise SystemExit("No windows created. Check --input and BITS2 canonical data.")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.stack(X), y=np.asarray(y, np.int64),
                        subjects=np.asarray(subjects), sources=np.asarray(sources),
                        classes=np.asarray(CLASSES))
    audit = {
        "windows": len(y), "subjects": len(set(subjects)),
        "classes": CLASSES, "window_samples": args.window_samples,
        "stride_samples": args.stride_samples, "purity": 0.90,
        "input": str(args.input), "output": str(out)
    }
    Path(str(out) + ".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
