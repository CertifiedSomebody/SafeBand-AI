"""Audit SafeBand BNO055 Synthetic V2.1 for schema, timing and sanity."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bno055_v2_common import ACTIVITIES, REQUIRED_COLUMNS, RAW9_CHANNELS

FS = 100.0


def main():
    p = argparse.ArgumentParser(description="Audit BNO055 Synthetic V2.1.")
    p.add_argument("--input", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2_1.csv")
    a = p.parse_args()

    if not a.input.exists():
        raise FileNotFoundError(a.input)

    df = pd.read_csv(a.input)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    # Event timing is intentionally NaN for non-event activities.
    numeric = [c for c in REQUIRED_COLUMNS if c not in ("activity_label", "event_type", "event_start_s", "event_end_s")]
    bad = {}
    for c in numeric:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        nbad = int((~np.isfinite(vals)).sum())
        if nbad:
            bad[c] = nbad

    counts = df["activity_label"].value_counts().sort_index()
    unknown = sorted(set(df["activity_label"].astype(str)) - set(ACTIVITIES))

    # Timing is evaluated inside each activity recording; timestamps are
    # intentionally re-zeroed for every activity, so global CSV diff is invalid.
    dts = []
    timing_bad = 0
    for _, g in df.groupby(["subject_id", "session_id", "activity_label"], sort=False):
        ts = g["timestamp_s"].to_numpy(dtype=float)
        dt = np.diff(ts)
        if dt.size:
            dts.extend(dt.tolist())
            timing_bad += int(np.sum((dt <= 0) | (np.abs(dt - 1/FS) > 0.002)))

    q = np.sqrt(
        df.quat_w**2 + df.quat_x**2 + df.quat_y**2 + df.quat_z**2
    )

    acc = np.linalg.norm(df[["accel_x_mps2","accel_y_mps2","accel_z_mps2"]].to_numpy(), axis=1)
    gyro = np.linalg.norm(df[["gyro_x_dps","gyro_y_dps","gyro_z_dps"]].to_numpy(), axis=1)
    mag = np.linalg.norm(df[["mag_x_uT","mag_y_uT","mag_z_uT"]].to_numpy(), axis=1)

    print(f"Rows: {len(df):,}")
    print(f"Subjects: {df.subject_id.nunique()}")
    print(f"Subject-session pairs: {df[['subject_id','session_id']].drop_duplicates().shape[0]}")
    print("Activities:")
    print(counts.to_string())
    print(f"Non-finite/non-numeric numeric cells: {sum(bad.values())}")
    print(f"Unknown activity labels: {unknown}")
    if dts:
        print(f"Median dt (within activity recordings): {np.median(dts):.6f}s")
        print(f"Timing samples outside ±2 ms of 0.010s: {timing_bad:,}")
    print(
        "Quaternion norm mean/std/min/max: "
        f"{q.mean():.9f} / {q.std():.9f} / {q.min():.9f} / {q.max():.9f}"
    )

    print("\nPer-activity ACC/GYRO/MAG magnitude:")
    summary = df.assign(acc_mag=acc, gyro_mag=gyro, mag_mag=mag).groupby(
        "activity_label"
    )[["acc_mag","gyro_mag","mag_mag"]].agg(["mean","std","min","max"])
    print(summary.round(3).to_string())

    print("\nPhysical bounds:")
    print(f"ACC magnitude max: {acc.max():.3f} m/s²")
    print(f"GYRO magnitude max: {gyro.max():.3f} °/s")
    print(f"MAG magnitude min/max: {mag.min():.3f} / {mag.max():.3f} µT")

    # Conservative synthetic-domain bounds. The ACC magnitude limit is 30 m/s²
    # (~3.06 g), leaving margin for the modeled fall impact while remaining
    # far below the BNO055 accelerometer component range.
    failures = []
    if missing:
        failures.append("missing required columns")
    if bad:
        failures.append("non-finite/non-numeric values")
    if unknown:
        failures.append("unknown activity labels")
    if not counts.index.tolist() == sorted(ACTIVITIES):
        failures.append("activity set mismatch")
    if not dts or abs(float(np.median(dts)) - 1/FS) > 0.0005:
        failures.append("sampling interval mismatch")
    if timing_bad:
        failures.append("non-100-Hz intervals detected")
    if np.max(np.abs(q - 1)) > 1e-4:
        failures.append("quaternion normalization")
    if not np.isfinite(acc).all() or acc.max() > 30.0:
        failures.append("ACC magnitude outside synthetic bound")
    if not np.isfinite(gyro).all() or gyro.max() > 350.0:
        failures.append("GYRO magnitude outside synthetic bound")
    if not np.isfinite(mag).all() or mag.min() < 20.0 or mag.max() > 75.0:
        failures.append("MAG magnitude outside synthetic bound")

    # Every activity must contain explicit event metadata only for event classes.
    for label in ACTIVITIES:
        g = df[df.activity_label == label]
        if label in ("FALL", "SIT_TO_STAND", "STAND_TO_SIT"):
            if g["event_type"].eq(label).mean() < 0.999:
                failures.append(f"{label} event metadata missing")
            if not (g["event_start_s"].notna().all() and g["event_end_s"].notna().all()):
                failures.append(f"{label} event times missing")
        else:
            if not g["event_type"].fillna("").eq("").all():
                failures.append(f"{label} unexpectedly has event metadata")

    if failures:
        print("\n[FAIL] Audit failed:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\n[PASS] Structural + physical sanity audit passed.")


if __name__ == "__main__":
    main()
