
"""Build AICOS-HAR ACC temporal windows for activity recognition.

Default:
- ACC only
- 50 Hz resampling
- 2.0 s windows (100 samples)
- 50% overlap
- hard acquisition boundaries
- no window crosses an acquisition
- official metadata Fold is preserved

The output is an NPZ plus a manifest CSV. This is a derived dataset; raw data
is untouched.
"""
from pathlib import Path
import argparse, csv, json, sys
import numpy as np
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_aicos_har_v3 import discover_sensor_files

def read_stream(path):
    t=[]; x=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            p=[z.strip() for z in line.strip().split("|")]
            if len(p)<4: continue
            try:
                t.append(int(float(p[0]))/1e9)
                x.append([float(z) for z in p[1:4]])
            except ValueError: pass
    if len(t)<4: return None,None
    t=np.asarray(t); x=np.asarray(x,dtype=np.float32)
    keep=np.r_[True,np.diff(t)>0]
    return t[keep],x[keep]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True)
    ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",default="datasets/processed/aicos_har/aicos_acc_50hz_2s.npz")
    ap.add_argument("--manifest",default="datasets/processed/aicos_har/aicos_acc_50hz_2s_manifest.csv")
    ap.add_argument("--target-hz",type=float,default=50.0)
    ap.add_argument("--window-s",type=float,default=2.0)
    ap.add_argument("--step-s",type=float,default=1.0)
    args=ap.parse_args()

    meta={}
    with open(args.metadata,"r",encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f): meta[r["AcquisitionID"].strip()]=r

    files=discover_sensor_files(args.data_dir)
    acc={r["acquisition_id"]:r for r in files if r["sensor"]=="acc"}

    n=int(round(args.target_hz*args.window_s))
    step=int(round(args.target_hz*args.step_s))
    X=[]; manifest=[]
    skipped=0

    for aid in sorted(meta):
        r=acc.get(aid)
        if not r: skipped+=1; continue
        t,x=read_stream(r["path"])
        if t is None: skipped+=1; continue
        start,end=float(t[0]),float(t[-1])
        if end-start < args.window_s: continue
        grid=np.arange(start,end-args.window_s+1e-9,1/args.target_hz)
        if len(grid)<n: continue
        y=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(3)]).astype(np.float32)

        for i in range(0,len(y)-n+1,step):
            w=y[i:i+n]
            X.append(w.T)
            manifest.append({
                "acquisition_id":aid,
                "participant_id":meta[aid].get("AcquisitionID","").split("/")[0],
                "activity":meta[aid].get("AcquisitionID","").split("/")[1].rsplit("_",1)[0],
                "fold":meta[aid].get("Fold",""),
                "position":meta[aid].get("Position_FullName",""),
            })

    X=np.asarray(X,dtype=np.float32)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X)
    mf=Path(args.manifest); mf.parent.mkdir(parents=True,exist_ok=True)
    with open(mf,"w",encoding="utf-8",newline="") as f:
        fields=list(manifest[0]) if manifest else ["acquisition_id","participant_id","activity","fold","position"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)

    print("AICOS-HAR WINDOW BUILD")
    print(f"Windows: {len(X)}")
    print(f"Shape: {X.shape}")
    print(f"Acquisitions without usable ACC: {skipped}")
    print(f"NPZ: {out}")
    print(f"Manifest: {mf}")

if __name__=="__main__": main()
