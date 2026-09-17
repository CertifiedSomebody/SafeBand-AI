from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

LABELS=["RESTING","SITTING","WALKING","RUNNING"]
LABEL_TO_ID={x:i for i,x in enumerate(LABELS)}

def read_inputs(canonical, activity):
    c=pd.read_csv(canonical)
    w=pd.read_csv(activity)
    c_req=["subject_id","source_file","sensor_type","timestamp_raw","x","y","z"]
    w_req=["subject_id","source_file","activity_label","window_start_sample","window_end_sample","window_samples"]
    for col in c_req:
        if col not in c.columns: raise ValueError(f"canonical_long missing required column: {col}")
    for col in w_req:
        if col not in w.columns: raise ValueError(f"activity_windows missing required column: {col}")
    c["sensor_type"]=c["sensor_type"].astype(str).str.lower()
    c["source_file"]=c["source_file"].astype(str)
    c["timestamp_raw"]=pd.to_numeric(c["timestamp_raw"],errors="coerce")
    for col in ["x","y","z"]: c[col]=pd.to_numeric(c[col],errors="coerce")
    w["source_file"]=w["source_file"].astype(str)
    w["activity_label"]=w["activity_label"].astype(str).str.upper()
    w["window_start_sample"]=pd.to_numeric(w["window_start_sample"],errors="coerce").astype("Int64")
    w["window_end_sample"]=pd.to_numeric(w["window_end_sample"],errors="coerce").astype("Int64")
    w["window_samples"]=pd.to_numeric(w["window_samples"],errors="coerce").astype("Int64")
    w=w[w.activity_label.isin(LABELS)].copy()
    return c,w

def resample_gyro_to_acc(acc,gyro):
    # Match gyro to the ACC timestamps. No invented clock is used.
    at=acc[:,0]
    gt=gyro[:,0]
    if len(at)==0 or len(gt)==0: return None
    order=np.argsort(gt)
    gt=gt[order]; gv=gyro[order,1:4]
    # Collapse duplicate gyro timestamps by keeping the first sample.
    keep=np.r_[True,np.diff(gt)!=0]
    gt=gt[keep]; gv=gv[keep]
    if at.min()<gt.min() or at.max()>gt.max():
        return None
    out=np.column_stack([np.interp(at,gt,gv[:,j]) for j in range(3)])
    return out.astype(np.float32)

def main():
    ap=argparse.ArgumentParser(description="Build subject-grouped BITS2 ACC+GYRO raw windows.")
    ap.add_argument("--canonical",type=Path,default=ROOT/"datasets/processed/bits2/bits2_canonical_long.csv")
    ap.add_argument("--activity",type=Path,default=ROOT/"datasets/processed/bits2/activity_windows.csv")
    ap.add_argument("--out",type=Path,default=ROOT/"datasets/processed/bits2/activity_accgyro_windows.npz")
    ap.add_argument("--min-coverage",type=float,default=0.98)
    a=ap.parse_args()

    c,w=read_inputs(a.canonical,a.activity)
    records={}
    for fname,grp in c[c.sensor_type.isin(["acc","gyro"])].groupby("source_file",sort=False):
        rec={}
        for sensor,sg in grp.groupby("sensor_type",sort=False):
            arr=sg[["timestamp_raw","x","y","z"]].dropna().to_numpy(np.float64)
            # Preserve file order for diagnostics, then sort by timestamp.
            arr=arr[np.argsort(arr[:,0])]
            rec[sensor]=arr
        if "acc" in rec and "gyro" in rec: records[fname]=rec

    X=[]; y=[]; groups=[]; subjects=[]; sources=[]; starts=[]; coverage=[]
    skipped={"missing_record":0,"short_window":0,"bad_bounds":0,"gyro_range":0,"low_coverage":0}
    expected_len=None

    for row in w.itertuples(index=False):
        fname=str(row.source_file)
        if fname not in records:
            skipped["missing_record"]+=1; continue
        acc=records[fname]["acc"]; gyro=records[fname]["gyro"]
        start=int(row.window_start_sample); end=int(row.window_end_sample); n=int(row.window_samples)
        if start<0 or end<start or n<=0 or end-start+1!=n:
            skipped["bad_bounds"]+=1; continue
        if expected_len is None: expected_len=n
        if n!=expected_len:
            skipped["bad_bounds"]+=1; continue

        # Window indices are defined by the existing ACC activity artifact.
        # They are applied to the sorted ACC stream of the same source file.
        ae=acc[start:end+1]
        if len(ae)!=n:
            skipped["short_window"]+=1; continue

        # Find gyro samples over the same timestamp interval and interpolate
        # onto ACC timestamps. This handles minor timestamp-rate differences.
        lo,hi=ae[0,0],ae[-1,0]
        mask=(gyro[:,0]>=lo)&(gyro[:,0]<=hi)
        ge=gyro[mask]
        if len(ge)<2:
            skipped["gyro_range"]+=1; continue
        gmatch=resample_gyro_to_acc(ae,ge)
        if gmatch is None or len(gmatch)!=n:
            skipped["gyro_range"]+=1; continue

        # Coverage is based on whether the interpolation used actual gyro data
        # across the complete ACC time interval.
        cov=float(min(1.0,(ge[-1,0]-ge[0,0])/(hi-lo))) if hi>lo else 1.0
        if cov<a.min_coverage:
            skipped["low_coverage"]+=1; continue

        sig=np.vstack([ae[:,1:4].T,gmatch.T]).astype(np.float32)
        X.append(sig); y.append(LABEL_TO_ID[str(row.activity_label)])
        groups.append(str(row.subject_id)); subjects.append(str(row.subject_id))
        sources.append(fname); starts.append(start); coverage.append(cov)

    if not X: raise RuntimeError("No valid ACC+GYRO windows were constructed.")

    X=np.stack(X); y=np.asarray(y,np.int64); groups=np.asarray(groups)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,X=X,y=y,groups=groups,subjects=groups,source_file=np.asarray(sources),window_start=np.asarray(starts),coverage=np.asarray(coverage,np.float32))

    audit={
        "output":str(a.out),"shape":list(X.shape),"classes":LABELS,
        "samples":int(len(X)),"subjects":int(len(np.unique(groups))),
        "window_samples":int(X.shape[-1]),"channels":["ax","ay","az","gx","gy","gz"],
        "skipped":skipped,
        "coverage_min":float(np.min(coverage)),"coverage_mean":float(np.mean(coverage)),
        "source_canonical":str(a.canonical),"source_activity_metadata":str(a.activity)
    }
    audit_path=a.out.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))

if __name__=="__main__": main()
