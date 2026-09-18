"""Audit BNO055 Synthetic V2 schema, timing, labels and physical sanity."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bno055_v2_common import ACTIVITIES, REQUIRED_COLUMNS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2.csv")
    a=p.parse_args()
    if not a.input.exists(): raise FileNotFoundError(a.input)
    df=pd.read_csv(a.input)
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError("Missing columns: "+", ".join(missing))
    numeric=[c for c in REQUIRED_COLUMNS if c!="activity_label"]
    bad={c:int((~np.isfinite(pd.to_numeric(df[c],errors="coerce"))).sum()) for c in numeric}
    bad={k:v for k,v in bad.items() if v}
    q=np.sqrt(df.quat_w**2+df.quat_x**2+df.quat_y**2+df.quat_z**2)
    print(f"Rows: {len(df):,}")
    print(f"Subjects: {df.subject_id.nunique()}")
    print(f"Subject-session pairs: {df[['subject_id','session_id']].drop_duplicates().shape[0]}")
    print("Activities:")
    print(df.activity_label.value_counts().sort_index().to_string())
    print(f"Non-finite/non-numeric numeric cells: {sum(bad.values())}")
    if bad: print(bad)
    print(f"Median dt: {df.timestamp_s.diff().dropna().median():.6f}s (global CSV ordering)")
    print(f"Quaternion norm mean/std/min/max: {q.mean():.9f} / {q.std():.9f} / {q.min():.9f} / {q.max():.9f}")
    print("\nPer-activity ACC/GYRO magnitude:")
    acc=np.linalg.norm(df[['accel_x_mps2','accel_y_mps2','accel_z_mps2']].to_numpy(),axis=1)
    gyro=np.linalg.norm(df[['gyro_x_dps','gyro_y_dps','gyro_z_dps']].to_numpy(),axis=1)
    x=df.assign(acc_mag=acc,gyro_mag=gyro)
    print(x.groupby("activity_label")[["acc_mag","gyro_mag"]].agg(["mean","std","min","max"]).round(3).to_string())
    if missing or bad or not np.all(np.isfinite(q)) or np.max(np.abs(q-1))>1e-4:
        raise SystemExit("[FAIL] Audit failed.")
    print("\n[PASS] Structural audit passed.")

if __name__=="__main__": main()
