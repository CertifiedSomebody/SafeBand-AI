"""Build alias-safe AICOS-HAR ACC windows while preserving metadata labels."""
from pathlib import Path
import argparse,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_aicos_har_v4 import discover_sensor_files

def read_stream(path):
    t=[];x=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            delimiter="," if "," in line else "|"
            p=[z.strip() for z in next(csv.reader([line],delimiter=delimiter))]
            if len(p)<4:continue
            try:t.append(int(float(p[0]))/1e9);x.append([float(z) for z in p[1:4]])
            except ValueError:pass
    if len(t)<4:return None,None
    t=np.asarray(t);x=np.asarray(x,dtype=np.float32);keep=np.r_[True,np.diff(t)>0]
    return t[keep],x[keep]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",required=True); ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",default="datasets/processed/aicos_har/aicos_acc_50hz_2s.npz"); ap.add_argument("--manifest",default="datasets/processed/aicos_har/aicos_acc_50hz_2s_manifest.csv")
    ap.add_argument("--target-hz",type=float,default=50); ap.add_argument("--window-s",type=float,default=2); ap.add_argument("--step-s",type=float,default=1)
    args=ap.parse_args()
    with open(args.metadata,"r",encoding="utf-8-sig",newline="") as f: meta={r["AcquisitionID"].strip():r for r in csv.DictReader(f)}
    rows=discover_sensor_files(args.data_dir); acc={r["acquisition_id"]:r for r in rows if r["sensor"]=="acc"}
    n=int(round(args.target_hz*args.window_s));step=int(round(args.target_hz*args.step_s));X=[];manifest=[];skipped=0
    for aid in sorted(meta):
        r=acc.get(aid)
        if not r:skipped+=1;continue
        t,x=read_stream(r["path"])
        if t is None or t[-1]-t[0]<args.window_s:skipped+=1;continue
        grid=np.arange(t[0],t[-1]-args.window_s+1e-9,1/args.target_hz)
        if len(grid)<n:continue
        y=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(3)]).astype(np.float32)
        for i in range(0,len(y)-n+1,step):
            X.append(y[i:i+n].T)
            manifest.append({"acquisition_id":aid,"participant_id":aid.split("/")[0],"activity":meta[aid].get("AcquisitionID","").split("/")[1].rsplit("_",1)[0],"fold":meta[aid].get("Fold",""),"position":meta[aid].get("Position_FullName","")})
    X=np.asarray(X,dtype=np.float32);out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,X=X)
    mf=Path(args.manifest);mf.parent.mkdir(parents=True,exist_ok=True)
    fields=["acquisition_id","participant_id","activity","fold","position"]
    with open(mf,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)
    print("AICOS-HAR WINDOW BUILD V4");print(f"Windows: {len(X)}");print(f"Shape: {X.shape}");print(f"Acquisitions without usable ACC: {skipped}");print(f"NPZ: {out}");print(f"Manifest: {mf}")
    if len(X)==0: raise SystemExit("No windows were built.")

if __name__=="__main__":main()
