
from __future__ import annotations
import argparse, csv
import numpy as np
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.bno055_synthetic_common import generate_session, ACTIVITIES, FS

HEADER=[
"subject_id","session_id","timestamp_s","activity_label",
"accel_x_mps2","accel_y_mps2","accel_z_mps2",
"gyro_x_dps","gyro_y_dps","gyro_z_dps",
"mag_x_uT","mag_y_uT","mag_z_uT",
"euler_heading_deg","euler_roll_deg","euler_pitch_deg",
"quat_w","quat_x","quat_y","quat_z",
"linear_accel_x_mps2","linear_accel_y_mps2","linear_accel_z_mps2",
"gravity_x_mps2","gravity_y_mps2","gravity_z_mps2",
"temperature_c"
]

def main():
    ap=argparse.ArgumentParser(description="Generate BNO055-oriented SafeBand synthetic telemetry.")
    ap.add_argument("--out",default="datasets/synthetic/bno055/bno055_synthetic_v1.csv")
    ap.add_argument("--subjects",type=int,default=8)
    ap.add_argument("--sessions",type=int,default=2)
    ap.add_argument("--duration",type=float,default=15.0)
    ap.add_argument("--activities",default=",".join(ACTIVITIES))
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    acts=[x.strip().upper() for x in args.activities.split(",") if x.strip()]
    bad=set(acts)-set(ACTIVITIES)
    if bad: raise SystemExit(f"Unknown activities: {sorted(bad)}")
    out=ROOT/args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    total=0
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(HEADER)
        for s in range(1,args.subjects+1):
            for sess in range(1,args.sessions+1):
                for a_i,a in enumerate(acts):
                    seed=args.seed+s*10000+sess*100+a_i
                    rows=generate_session(str(s),str(sess),a,args.duration,seed)
                    for r in rows:
                        w.writerow([r[0], r[1], f"{r[2]:.4f}", r[3]] + [
                            f"{v:.7f}" if isinstance(v, (float, np.floating)) else v
                            for v in r[4:]
                        ])
                    total+=len(rows)
    print(f"Created {out}")
    print(f"Rows: {total:,} | subjects: {args.subjects} | sessions: {args.sessions} | activities: {len(acts)} | rate: {FS:.0f} Hz")

if __name__=="__main__":
    main()
