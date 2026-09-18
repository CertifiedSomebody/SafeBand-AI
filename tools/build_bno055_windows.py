from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

RAW_CHANNELS=[
    'accel_x_mps2','accel_y_mps2','accel_z_mps2',
    'gyro_x_dps','gyro_y_dps','gyro_z_dps',
    'mag_x_uT','mag_y_uT','mag_z_uT'
]
FUSED_CHANNELS=[
    'euler_heading_deg','euler_roll_deg','euler_pitch_deg',
    'quat_w','quat_x','quat_y','quat_z',
    'linear_accel_x_mps2','linear_accel_y_mps2','linear_accel_z_mps2',
    'gravity_x_mps2','gravity_y_mps2','gravity_z_mps2'
]


def main():
    ap=argparse.ArgumentParser(description='Build non-overlapping-safe 2 s windows from BNO055 synthetic telemetry.')
    ap.add_argument('--csv',default='datasets/synthetic/bno055/bno055_synthetic_v1.csv')
    ap.add_argument('--out',default='datasets/synthetic/bno055/bno055_windows_v1.npz')
    ap.add_argument('--window-samples',type=int,default=200)
    ap.add_argument('--step-samples',type=int,default=100)
    ap.add_argument('--channels',choices=['raw9','raw9_fused13'],default='raw9')
    args=ap.parse_args()
    p=ROOT/args.csv
    df=pd.read_csv(p)
    required=['subject_id','session_id','timestamp_s','activity_label']+RAW_CHANNELS
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f'Missing required columns: {missing}')
    cols=RAW_CHANNELS if args.channels=='raw9' else RAW_CHANNELS+FUSED_CHANNELS
    X=[]; y=[]; groups=[]; sessions=[]
    for (subject,session,activity), g in df.groupby(['subject_id','session_id','activity_label'],sort=False):
        g=g.sort_values('timestamp_s')
        arr=g[cols].to_numpy(dtype=np.float32)
        # Require an uninterrupted 100 Hz sequence within each activity segment.
        ts=g['timestamp_s'].to_numpy(dtype=float)
        if len(ts)>1:
            d=np.diff(ts)
            if not np.allclose(d,0.01,atol=2e-4):
                raise SystemExit(f'Timestamp contract failed for subject={subject}, session={session}, activity={activity}')
        for start in range(0,len(arr)-args.window_samples+1,args.step_samples):
            w=arr[start:start+args.window_samples]
            if not np.isfinite(w).all():
                continue
            X.append(w.T); y.append(activity); groups.append(str(subject)); sessions.append(str(session))
    if not X: raise SystemExit('No windows generated.')
    X=np.stack(X).astype(np.float32)
    y=np.asarray(y); groups=np.asarray(groups); sessions=np.asarray(sessions)
    out=ROOT/args.out; out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,y=y,groups=groups,sessions=sessions,channels=np.asarray(cols))
    print(f'Created {out}')
    print(f'X shape: {X.shape} | windows: {len(y):,} | subjects: {len(set(groups))} | classes: {sorted(set(y))}')

if __name__=='__main__': main()
