"""Build deterministic 2-second BNO055 V2.1 raw9 windows."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bno055_v2_common import ACTIVITIES, RAW9_CHANNELS


def main():
    p = argparse.ArgumentParser(description="Build BNO055 V2.1 windows.")
    p.add_argument("--input", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2_1.csv")
    p.add_argument("--out", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_windows_v2_1.npz")
    p.add_argument("--window-seconds", type=float, default=2.0)
    p.add_argument("--step-seconds", type=float, default=1.0)
    a = p.parse_args()

    if not a.input.exists():
        raise FileNotFoundError(a.input)

    df = pd.read_csv(a.input)
    missing = [c for c in RAW9_CHANNELS + ["subject_id","session_id","timestamp_s","activity_label"]
               if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    fs = 100.0
    w = int(round(a.window_seconds * fs))
    step = int(round(a.step_seconds * fs))
    if w < 2 or step < 1:
        raise ValueError("Invalid window/step.")
    if step > w:
        raise ValueError("step-seconds cannot exceed window-seconds.")

    class_map = {name: i for i, name in enumerate(ACTIVITIES)}
    Xs, ys, groups, sessions, starts = [], [], [], [], []
    event_start, event_end, event_flags = [], [], []

    for (sub, sess, label), g in df.groupby(
        ["subject_id", "session_id", "activity_label"], sort=False
    ):
        label = str(label)
        if label not in class_map:
            raise ValueError(f"Unknown activity label: {label}")

        g = g.sort_values("timestamp_s").reset_index(drop=True)
        ts = g["timestamp_s"].to_numpy(dtype=float)
        arr = g[RAW9_CHANNELS].to_numpy(dtype=np.float32)

        dt = np.diff(ts)
        if len(dt) and not np.all(np.isfinite(dt)) or (len(dt) and np.any(dt <= 0)):
            raise ValueError(f"Non-increasing timestamps for {sub}/{sess}/{label}.")

        if len(arr) < w:
            continue

        es = float(g["event_start_s"].iloc[0]) if pd.notna(g["event_start_s"].iloc[0]) else np.nan
        ee = float(g["event_end_s"].iloc[0]) if pd.notna(g["event_end_s"].iloc[0]) else np.nan

        for st in range(0, len(arr) - w + 1, step):
            en = st + w
            Xs.append(arr[st:en].T)
            ys.append(class_map[label])
            groups.append(int(sub))
            sessions.append(int(sess))
            starts.append(float(ts[st]))
            event_start.append(es)
            event_end.append(ee)

            # Window overlaps the event interval.
            win_start, win_end = ts[st], ts[en-1]
            is_event = (
                np.isfinite(es) and np.isfinite(ee)
                and win_end >= es and win_start <= ee
            )
            event_flags.append(bool(is_event))

    if not Xs:
        raise ValueError("No windows generated.")

    X = np.stack(Xs)
    y = np.asarray(ys, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    sessions = np.asarray(sessions, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.float64)
    event_start = np.asarray(event_start, dtype=np.float64)
    event_end = np.asarray(event_end, dtype=np.float64)
    event_flags = np.asarray(event_flags, dtype=bool)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        X=X, y=y, groups=groups, sessions=sessions, starts=starts,
        event_start_s=event_start, event_end_s=event_end,
        is_event_window=event_flags,
        channels=np.asarray(RAW9_CHANNELS),
        classes=np.asarray(ACTIVITIES),
        sampling_hz=np.asarray(fs),
        window_seconds=np.asarray(a.window_seconds),
        step_seconds=np.asarray(a.step_seconds),
    )

    print(f"[PASS] Created: {a.out}")
    print(f"[PASS] X shape: {X.shape}")
    print(f"[PASS] Windows: {len(y):,}")
    print(f"[PASS] Subjects: {len(np.unique(groups))}")
    print(f"[PASS] Classes: {len(np.unique(y))}")
    print(f"[PASS] Event-overlap windows: {int(event_flags.sum()):,}")


if __name__ == "__main__":
    main()
