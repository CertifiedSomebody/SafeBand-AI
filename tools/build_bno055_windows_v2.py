"""Build deterministic 2-second BNO055 V2 raw9 windows."""
from pathlib import Path
import argparse, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bno055_v2_common import ACTIVITIES, RAW9_CHANNELS

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"datasets/synthetic/bno055/bno055_synthetic_v2.csv")
    p.add_argument("--out",type=Path,default=ROOT/"datasets/synthetic/bno055/bno055_windows_v2.npz")
    p.add_argument("--window-seconds",type=float,default=2.0)
    p.add_argument("--step-seconds",type=float,default=1.0)
    a=p.parse_args()
    df=pd.read_csv(a.input)
    fs=100.0
    w=int(round(a.window_seconds*fs)); step=int(round(a.step_seconds*fs))
    if w<2 or step<1: raise ValueError("Invalid window/step")
    Xs=[]; ys=[]; groups=[]; sessions=[]; starts=[]
    class_map={v:i for i,v in enumerate(ACTIVITIES)}
    for (sub,sess,label), g in df.groupby(["subject_id","session_id","activity_label"],sort=False):
        g=g.sort_values("timestamp_s")
        arr=g[RAW9_CHANNELS].to_numpy(dtype=np.float32)
        # Only contiguous within one activity segment.
        for st in range(0,len(arr)-w+1,step):
            Xs.append(arr[st:st+w].T)
            ys.append(class_map[str(label)])
            groups.append(int(sub)); sessions.append(int(sess))
            starts.append(float(g["timestamp_s"].iloc[st]))
    if not Xs: raise ValueError("No windows generated.")
    X=np.stack(Xs); y=np.asarray(ys,np.int64)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,X=X,y=y,groups=np.asarray(groups),sessions=np.asarray(sessions),
                        starts=np.asarray(starts),channels=np.asarray(RAW9_CHANNELS),
                        classes=np.asarray(ACTIVITIES))
    print(f"[PASS] Created: {a.out}")
    print(f"[PASS] X shape: {X.shape}")
    print(f"[PASS] Windows: {len(y):,}")
    print(f"[PASS] Subjects: {len(np.unique(groups))}")
    print(f"[PASS] Classes: {len(np.unique(y))}")

if __name__=="__main__": main()
