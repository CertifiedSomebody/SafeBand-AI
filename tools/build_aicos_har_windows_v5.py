"""Build alias-safe AICOS-HAR ACC windows with an explicit .npz output."""
from pathlib import Path
import argparse,csv,sys,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from tools.common_aicos_har_v4 import discover_sensor_files, metadata_activity

def read_stream(path):
    t=[]; x=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            p=[z.strip() for z in line.strip().split("|")]
            if len(p)<4: continue
            try:
                t.append(int(float(p[0]))/1e9); x.append([float(z) for z in p[1:4]])
            except ValueError: pass
    if len(t)<4: return None,None
    t=np.asarray(t,dtype=np.float64); x=np.asarray(x,dtype=np.float32)
    keep=np.r_[True,np.diff(t)>0]
    return t[keep],x[keep]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",required=True); ap.add_argument("--metadata",required=True)
    ap.add_argument("--out",default="datasets/processed/aicos_har/aicos_acc_50hz_2s.npz")
    ap.add_argument("--manifest",default="datasets/processed/aicos_har/aicos_acc_50hz_2s_manifest.csv")
    ap.add_argument("--target-hz",type=float,default=50); ap.add_argument("--window-s",type=float,default=2); ap.add_argument("--step-s",type=float,default=1)
    args=ap.parse_args()
    out=Path(args.out)
    if out.suffix.lower() != ".npz":
        raise SystemExit(f"ERROR: --out must end in .npz (received: {out})")
    if args.target_hz<=0 or args.window_s<=0 or args.step_s<=0: raise SystemExit("ERROR: timing parameters must be positive")
    with open(args.metadata,"r",encoding="utf-8-sig",newline="") as f: meta={r["AcquisitionID"].strip():r for r in csv.DictReader(f)}
    rows=discover_sensor_files(args.data_dir); acc={r["acquisition_id"]:r for r in rows if r["sensor"]=="acc"}
    n=int(round(args.target_hz*args.window_s)); step=int(round(args.target_hz*args.step_s))
    X=[]; manifest=[]; skipped=[]
    for aid in sorted(meta):
        r=acc.get(aid)
        if not r: skipped.append((aid,"no_acc")); continue
        t,x=read_stream(r["path"])
        if t is None or t[-1]-t[0] < args.window_s: skipped.append((aid,"short_or_invalid")); continue
        grid=np.arange(t[0],t[-1]-args.window_s+1e-9,1/args.target_hz)
        if len(grid)<n: skipped.append((aid,"short_after_resample")); continue
        y=np.column_stack([np.interp(grid,t,x[:,j]) for j in range(3)]).astype(np.float32)
        for i in range(0,len(y)-n+1,step):
            X.append(y[i:i+n].T)
            manifest.append({"acquisition_id":aid,"participant_id":aid.split("/")[0],"activity":metadata_activity(aid),"fold":meta[aid].get("Fold",""),"position":meta[aid].get("Position_FullName","")})
    X=np.asarray(X,dtype=np.float32)
    if X.ndim != 3 or X.shape[1:] != (3,n): raise SystemExit(f"ERROR: unexpected window shape {X.shape}")
    if not np.isfinite(X).all(): raise SystemExit("ERROR: non-finite values in windows")
    out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,X=X)
    mf=Path(args.manifest); mf.parent.mkdir(parents=True,exist_ok=True)
    fields=["acquisition_id","participant_id","activity","fold","position"]
    with open(mf,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(manifest)
    report=out.with_name(out.stem+"_build_report.json")
    import json
    report.write_text(json.dumps({"windows":int(len(X)),"shape":list(X.shape),"target_hz":args.target_hz,"window_s":args.window_s,"step_s":args.step_s,"metadata_acquisitions":len(meta),"acc_acquisitions":len(acc),"acquisitions_with_windows":len(set(r["acquisition_id"] for r in manifest)),"skipped":len(skipped),"skipped_examples":skipped[:20]},indent=2),encoding="utf-8")
    print("AICOS-HAR WINDOW BUILD V5")
    print(f"Windows: {len(X)}"); print(f"Shape: {X.shape}"); print(f"Metadata acquisitions: {len(meta)}"); print(f"ACC acquisitions discovered: {len(acc)}"); print(f"Acquisitions with windows: {len(set(r['acquisition_id'] for r in manifest))}"); print(f"Skipped acquisitions: {len(skipped)}"); print(f"NPZ: {out}"); print(f"Manifest: {mf}"); print(f"Build report: {report}")

if __name__=="__main__": main()
