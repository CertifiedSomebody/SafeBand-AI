"""Visualize representative BNO055 Synthetic V2.1 windows."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bno055_v2_common import ACTIVITIES, RAW9_CHANNELS

ACCEL_COLS = RAW9_CHANNELS[:3]
GYRO_COLS = RAW9_CHANNELS[3:6]
MAG_COLS = RAW9_CHANNELS[6:9]


def args():
    p = argparse.ArgumentParser(description="Visualize BNO055 Synthetic V2.1.")
    p.add_argument("--input", type=Path,
                   default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2_1.csv")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT/"reports/figures/bno055_synthetic_v2_1")
    p.add_argument("--subject", type=int, default=None)
    p.add_argument("--session", type=int, default=None)
    p.add_argument("--duration", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def validate(df):
    required = ["subject_id","session_id","timestamp_s","activity_label",
                *RAW9_CHANNELS, "event_start_s","event_end_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    numeric = ["subject_id","session_id","timestamp_s",*RAW9_CHANNELS,
               "event_start_s","event_end_s"]
    for c in numeric:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f"{c} must be numeric.")


def choose_recording(df, requested_subject, requested_session):
    x = df.copy()
    if requested_subject is not None:
        x = x[x.subject_id == requested_subject]
    if requested_session is not None:
        x = x[x.session_id == requested_session]
    if x.empty:
        raise ValueError("No rows match subject/session filters.")

    required = set(ACTIVITIES)
    found = (
        x.groupby(["subject_id","session_id"])["activity_label"]
        .agg(lambda s: required.issubset(set(s.astype(str))))
    )
    complete = found[found]
    if complete.empty:
        raise ValueError("No subject/session contains all nine activities.")
    return tuple(map(int, sorted(complete.index.tolist())[0]))


def select_window(part, duration, rng, event_center=False):
    part = part.sort_values("timestamp_s").reset_index(drop=True)
    ts = part.timestamp_s.to_numpy(float)
    dt = np.diff(ts)
    pos = dt[dt > 0]
    if len(pos) == 0:
        raise ValueError("Invalid timestamps.")
    period = float(np.median(pos))
    w = max(2, int(round(duration / period)))
    if len(part) < w:
        raise ValueError("Activity segment is shorter than requested window.")

    max_start = len(part)-w
    if event_center and pd.notna(part.event_start_s.iloc[0]):
        es = float(part.event_start_s.iloc[0])
        ee = float(part.event_end_s.iloc[0])
        center = (es+ee)/2
        candidates = np.arange(max_start+1)
        centers = ts[candidates] + (ts[np.minimum(candidates+w-1, len(ts)-1)]-ts[candidates])/2
        # Pick the valid start whose window center is closest to the event center,
        # then add a tiny deterministic seed-based offset only if it remains near.
        start = int(candidates[np.argmin(np.abs(centers-center))])
    else:
        start = int(rng.integers(0, max_start+1))

    out = part.iloc[start:start+w].copy()
    out["time_rel_s"] = out.timestamp_s - out.timestamp_s.iloc[0]
    out["acc_mag"] = np.linalg.norm(out[ACCEL_COLS].to_numpy(float), axis=1)
    out["gyro_mag"] = np.linalg.norm(out[GYRO_COLS].to_numpy(float), axis=1)
    out["mag_mag"] = np.linalg.norm(out[MAG_COLS].to_numpy(float), axis=1)
    return out


def plot_sensor_window(w, activity, subject, session, path):
    t = w.time_rel_s.to_numpy(float)
    fig, axes = plt.subplots(3,1,figsize=(13,10),sharex=True)

    axes[0].plot(t,w[ACCEL_COLS[0]],label="X")
    axes[0].plot(t,w[ACCEL_COLS[1]],label="Y")
    axes[0].plot(t,w[ACCEL_COLS[2]],label="Z")
    axes[0].plot(t,w.acc_mag,label="Magnitude",linewidth=2)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].grid(True,alpha=.25); axes[0].legend(ncol=4)

    axes[1].plot(t,w[GYRO_COLS[0]],label="X")
    axes[1].plot(t,w[GYRO_COLS[1]],label="Y")
    axes[1].plot(t,w[GYRO_COLS[2]],label="Z")
    axes[1].plot(t,w.gyro_mag,label="Magnitude",linewidth=2)
    axes[1].set_ylabel("Angular rate (°/s)")
    axes[1].grid(True,alpha=.25); axes[1].legend(ncol=4)

    axes[2].plot(t,w[MAG_COLS[0]],label="X")
    axes[2].plot(t,w[MAG_COLS[1]],label="Y")
    axes[2].plot(t,w[MAG_COLS[2]],label="Z")
    axes[2].plot(t,w.mag_mag,label="Magnitude",linewidth=2)
    axes[2].set_ylabel("Magnetic field (µT)")
    axes[2].set_xlabel("Time from window start (s)")
    axes[2].grid(True,alpha=.25); axes[2].legend(ncol=4)

    if pd.notna(w.event_start_s.iloc[0]):
        es=float(w.event_start_s.iloc[0])-float(w.timestamp_s.iloc[0])
        ee=float(w.event_end_s.iloc[0])-float(w.timestamp_s.iloc[0])
        for ax in axes:
            if ee >= ax.get_xlim()[0] and es <= ax.get_xlim()[1]:
                ax.axvspan(max(es,0), min(ee,t[-1]), alpha=.12)

    fig.suptitle(
        f"SafeBand AI — BNO055 Synthetic V2.1 | {activity} | "
        f"Subject {subject} | Session {session}",
        fontsize=14
    )
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches="tight")
    plt.close(fig)


def main():
    a=args()
    if a.duration <= 0:
        raise ValueError("--duration must be > 0.")
    if not a.input.exists():
        raise FileNotFoundError(a.input)

    a.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(a.input)
    validate(df)
    subject,session=choose_recording(df,a.subject,a.session)
    rng=np.random.default_rng(a.seed)

    rows=[]
    for activity in ACTIVITIES:
        part=df[(df.subject_id==subject)&(df.session_id==session)&
                (df.activity_label==activity)]
        if part.empty:
            raise ValueError(f"Missing activity {activity}.")
        event_mode=activity in ("FALL","SIT_TO_STAND","STAND_TO_SIT")
        w=select_window(part,a.duration,rng,event_center=event_mode)
        path=a.output_dir/f"activity_{activity.lower()}_window.png"
        plot_sensor_window(w,activity,subject,session,path)
        rows.append({
            "activity_label":activity,"subject_id":subject,"session_id":session,
            "start_timestamp_s":float(w.timestamp_s.iloc[0]),
            "end_timestamp_s":float(w.timestamp_s.iloc[-1]),
            "samples":len(w),
            "acc_mag_mean":float(w.acc_mag.mean()),
            "acc_mag_std":float(w.acc_mag.std()),
            "gyro_mag_mean":float(w.gyro_mag.mean()),
            "gyro_mag_std":float(w.gyro_mag.std()),
            "mag_mag_mean":float(w.mag_mag.mean()),
            "mag_mag_std":float(w.mag_mag.std()),
            "event_start_s":float(w.event_start_s.iloc[0]) if pd.notna(w.event_start_s.iloc[0]) else np.nan,
            "event_end_s":float(w.event_end_s.iloc[0]) if pd.notna(w.event_end_s.iloc[0]) else np.nan,
        })
        print(f"[OK] {activity:15s} -> {path.name}")

    summary=a.output_dir/"selected_windows.csv"
    pd.DataFrame(rows).to_csv(summary,index=False)
    print(f"\n[PASS] Visualization completed.")
    print(f"[PASS] Figures : {a.output_dir}")
    print(f"[PASS] Summary : {summary}")


if __name__=="__main__":
    main()
