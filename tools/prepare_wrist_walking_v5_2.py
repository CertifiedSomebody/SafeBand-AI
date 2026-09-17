#!/usr/bin/env python3
"""Prepare walking records for SafeBand PPG V5.2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,wfdb
from ai.walking_features_v5_2 import extract_features

WINDOW_SEC=8.0
SHIFT_SEC=2.0
FS=256.0
META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def walk_records(root):
    return [x.strip() for x in (Path(root)/"RECORDS").read_text().splitlines() if x.strip().endswith("_walk")]

def hr_target(peaks,start,end,fs):
    p=np.asarray(peaks,dtype=int); p=p[(p>=start)&(p<end)]
    if len(p)<2:return np.nan
    rr=np.diff(p)/fs; rr=rr[(rr>=60/220)&(rr<=60/30)]
    if len(rr)==0:return np.nan
    return float(np.mean(60/rr))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); ap.add_argument("--summary-out",required=True)
    args=ap.parse_args(); root=Path(args.root); rows=[]; audit=[]
    for rec in walk_records(root):
        path=str(root/rec); r=wfdb.rdrecord(path,physical=True); a=wfdb.rdann(path,"atr")
        names=list(r.sig_name); idx={n:i for i,n in enumerate(names)}
        required=["wrist_ppg","wrist_gyro_x","wrist_gyro_y","wrist_gyro_z",
                  "wrist_low_noise_accelerometer_x","wrist_low_noise_accelerometer_y","wrist_low_noise_accelerometer_z",
                  "wrist_wide_range_accelerometer_x","wrist_wide_range_accelerometer_y","wrist_wide_range_accelerometer_z",
                  "wrist_mag_x","wrist_mag_y","wrist_mag_z"]
        miss=[x for x in required if x not in idx]
        if miss: raise ValueError(f"{rec}: missing {miss}")
        if abs(float(r.fs)-FS)>1e-6: raise ValueError(f"{rec}: expected {FS} Hz, got {r.fs}")
        ppg=r.p_signal[:,idx["wrist_ppg"]]
        gyro=r.p_signal[:,[idx[f"wrist_gyro_{a}"] for a in "xyz"]]
        acc=r.p_signal[:,[idx[f"wrist_low_noise_accelerometer_{a}"] for a in "xyz"]]
        wacc=r.p_signal[:,[idx[f"wrist_wide_range_accelerometer_{a}"] for a in "xyz"]]
        mag=r.p_signal[:,[idx[f"wrist_mag_{a}"] for a in "xyz"]]
        peaks=np.asarray(a.sample,dtype=int)
        nwin=max(0,int(np.floor((len(ppg)-WINDOW_SEC*r.fs)/(SHIFT_SEC*r.fs)))+1); valid=0
        for k in range(nwin):
            start=int(round(k*SHIFT_SEC*r.fs)); end=start+int(round(WINDOW_SEC*r.fs))
            target=hr_target(peaks,start,end,float(r.fs))
            if not np.isfinite(target): continue
            feat=extract_features(ppg[start:end],acc[start:end],gyro[start:end],wacc[start:end],mag[start:end],float(r.fs))
            row={"subject":rec.split("_")[0],"activity":"walk","record":rec,"start_sample":start,
                 "start_time":start/float(r.fs),"hr_bpm":target}; row.update(feat); rows.append(row); valid+=1
        audit.append({"record":rec,"subject":rec.split("_")[0],"fs_hz":float(r.fs),"samples":int(len(ppg)),
                      "duration_sec":float(len(ppg)/r.fs),"annotation_count":int(len(peaks)),
                      "windows_total":nwin,"windows_valid":valid,"channel_count":len(names)})
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No valid walking windows produced")
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.summary_out).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.out,index=False)
    feats=[c for c in df.columns if c not in META]
    summary={"version":"PPG WALKING V5.2 PREPARATION","window_sec":WINDOW_SEC,"shift_sec":SHIFT_SEC,"fs_hz":FS,
             "records":audit,"rows":len(df),"subjects":sorted(df.subject.unique().tolist()),"features":len(feats),
             "target_definition":"Mean instantaneous HR from ECG annotation RR intervals inside each 8 s window; ECG waveform is not an input."}
    Path(args.summary_out).write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
