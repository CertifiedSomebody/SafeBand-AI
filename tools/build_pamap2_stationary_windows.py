from pathlib import Path
import sys, re, argparse, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CSV = REPO_ROOT / "datasets" / "raw" / "PAMAP2" / "dataset2.csv"
DEFAULT_OUT = REPO_ROOT / "datasets" / "processed" / "pamap2" / "stationary_accgyro_windows.npz"

TARGETS = {"lying": 0, "sitting": 1, "standing": 2}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def find_col(columns, *patterns):
    n = {norm(c): c for c in columns}
    for p in patterns:
        if norm(p) in n:
            return n[norm(p)]
    for c in columns:
        nc = norm(c)
        if all(norm(p) in nc for p in patterns):
            return c
    return None

def resolve_schema(columns):
    activity = find_col(columns, "activityID") or find_col(columns, "activity")
    people = find_col(columns, "PeopleId") or find_col(columns, "person") or find_col(columns, "subject")
    acc = [find_col(columns, "hand acceleration X"),
           find_col(columns, "hand acceleration Y"),
           find_col(columns, "hand acceleration Z")]
    gyro = [find_col(columns, "hand gyroscope X"),
            find_col(columns, "hand gyroscope Y"),
            find_col(columns, "hand gyroscope Z")]
    if activity is None or people is None or any(x is None for x in acc + gyro):
        raise RuntimeError(
            "Could not resolve required columns. Run audit_bits2/pamap2_schema_audit.py "
            "and inspect the detected schema."
        )
    return activity, people, acc, gyro

def label_key(x):
    s = str(x).strip().lower()
    return s if s in TARGETS else None

def flush_run(rows, person, label, Xs, ys, gs, window, step):
    if label is None or len(rows) < window:
        return
    arr = np.asarray(rows, dtype=np.float32)
    # arr shape: time x 6
    for start in range(0, len(arr) - window + 1, step):
        w = arr[start:start + window]
        if not np.isfinite(w).all():
            continue
        Xs.append(w.T)
        ys.append(TARGETS[label])
        gs.append(str(person))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=200, help="Samples. PAMAP2 is 100 Hz, so 200=2 seconds.")
    ap.add_argument("--step", type=int, default=100, help="Samples. Default 50%% overlap.")
    args = ap.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    if args.window <= 0 or args.step <= 0:
        raise ValueError("window and step must be positive")

    head = pd.read_csv(args.csv, nrows=5)
    activity_col, people_col, acc, gyro = resolve_schema(head.columns)

    Xs, ys, gs = [], [], []
    current_person = None
    current_label = None
    run = []

    def finish():
        nonlocal run
        flush_run(run, current_person, current_label, Xs, ys, gs, args.window, args.step)
        run = []

    for chunk in pd.read_csv(args.csv, chunksize=100000, low_memory=False):
        cols = [people_col, activity_col, *acc, *gyro]
        c = chunk[cols].copy()
        c[activity_col] = c[activity_col].map(label_key)

        for row in c.itertuples(index=False, name=None):
            person = str(row[0])
            label = row[1]
            vals = row[2:8]

            # A new subject or activity creates a hard boundary.
            # This prevents windows from crossing activity transitions.
            if person != current_person or label != current_label:
                finish()
                current_person = person
                current_label = label

            if label is None:
                continue

            try:
                vals = [float(v) for v in vals]
            except (TypeError, ValueError):
                continue

            if not np.isfinite(vals).all():
                continue

            run.append(vals)

    finish()

    if not Xs:
        raise RuntimeError("No valid stationary windows were constructed.")

    X = np.stack(Xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    groups = np.asarray(gs, dtype=str)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y, groups=groups)

    metadata = {
        "source": str(args.csv),
        "output": str(args.out),
        "window_samples": args.window,
        "step_samples": args.step,
        "assumed_source_rate_hz": 100,
        "window_seconds": args.window / 100.0,
        "step_seconds": args.step / 100.0,
        "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
        "classes": ["LYING", "SITTING", "STANDING"],
        "shape": list(X.shape),
        "subjects": sorted(set(groups.tolist())),
        "class_counts": {str(i): int((y == i).sum()) for i in range(3)},
        "note": "Windows never cross subject/activity boundaries; rows with missing/nonfinite required values are skipped."
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("BUILT PAMAP2 STATIONARY DATASET")
    print("X shape:", X.shape)
    print("classes:", metadata["class_counts"])
    print("subjects:", metadata["subjects"])
    print("saved:", args.out)

if __name__ == "__main__":
    main()
