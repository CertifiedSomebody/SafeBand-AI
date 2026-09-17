#!/usr/bin/env python3
"""Prepare PhysioNet PTT/MAX30101-domain data for SafeBand PPG V6."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.ppg_v6_features import extract_features

FS_HZ=500.0
PPG_COLS=["pleth_1","pleth_2","pleth_3","pleth_4","pleth_5","pleth_6"]
ACC_COLS=["a_x","a_y","a_z"]
GYRO_COLS=["g_x","g_y","g_z"]
REQUIRED=["time","peaks",*PPG_COLS,*ACC_COLS,*GYRO_COLS]
RECORD_RE=re.compile(r"^s(?P<subject>\d+)_(?P<activity>run|sit|walk)$",re.I)

def parse_record_name(path:Path):
    m=RECORD_RE.fullmatch(path.stem)
    if not m: raise ValueError(f"Unexpected record filename '{path.name}'")
    return f"s{int(m.group('subject'))}",m.group('activity').lower()

def ecg_hr_from_peaks(peaks,fs):
    x=np.asarray(peaks,float)
    idx=np.flatnonzero(x>0)
    if idx.size<3:return None
    rr=np.diff(idx)/float(fs)
    rr=rr[(rr>=60/220)&(rr<=60/30)]
    if rr.size<2:return None
    hr=60/float(np.median(rr))
    return hr if 30<=hr<=220 else None

def process_record(path,window_sec,shift_sec):
    df=pd.read_csv(path)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing:raise ValueError(f"{path.name}: missing required columns: {missing}")
    subject,activity=parse_record_name(path)
    numeric=df[["peaks",*PPG_COLS,*ACC_COLS,*GYRO_COLS]].apply(pd.to_numeric,errors="coerce")
    if numeric.isna().any().any():
        bad=numeric.isna().sum(); bad=bad[bad>0].to_dict()
        raise ValueError(f"{path.name}: NaN/non-numeric values in {bad}")
    w=int(round(window_sec*FS_HZ)); step=int(round(shift_sec*FS_HZ))
    if w<=0 or step<=0 or step>w:raise ValueError("Require 0 < shift_sec <= window_sec")
    rows=[]; skipped=0
    for start in range(0,len(df)-w+1,step):
        stop=start+w; p=df.iloc[start:stop]
        hr=ecg_hr_from_peaks(p.peaks.to_numpy(),FS_HZ)
        if hr is None: skipped+=1; continue
        f=extract_features(p[PPG_COLS].to_numpy(float),p[ACC_COLS].to_numpy(float),p[GYRO_COLS].to_numpy(float),FS_HZ,FS_HZ)
        f.update(subject=subject,activity=activity,record=path.stem,start_sample=int(start),start_time=str(p.time.iloc[0]),hr_bpm=float(hr))
        rows.append(f)
    return rows,{"record":path.stem,"subject":subject,"activity":activity,"raw_samples":int(len(df)),"possible_windows":int(max(0,(len(df)-w)//step+1)),"valid_hr_windows":len(rows),"skipped_no_hr":skipped}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ptt-root",required=True); ap.add_argument("--out",required=True); ap.add_argument("--summary-out",required=True); ap.add_argument("--window-sec",type=float,default=8.0); ap.add_argument("--shift-sec",type=float,default=2.0)
    a=ap.parse_args(); root=Path(a.ptt_root); csv_dir=root/"csv"
    if not csv_dir.is_dir():csv_dir=root/"CSV"
    if not csv_dir.is_dir():raise FileNotFoundError(f"Neither csv nor CSV exists under {root}")
    records=[]
    for p in csv_dir.glob("s*.csv"):
        try:parse_record_name(p);records.append(p)
        except ValueError:pass
    records.sort(key=lambda p:(parse_record_name(p)[0][1:].zfill(6),parse_record_name(p)[1]))
    if not records:raise FileNotFoundError("No sXX_(run|sit|walk).csv records found")
    rows=[]; summaries=[]
    for i,p in enumerate(records,1):
        print(f"[{i:02d}/{len(records):02d}] {p.name}"); r,s=process_record(p,a.window_sec,a.shift_sec); rows.extend(r);summaries.append(s)
    if not rows:raise RuntimeError("No valid windows produced")
    df=pd.DataFrame(rows); meta=["subject","activity","record","start_sample","start_time","hr_bpm"]; feats=sorted(c for c in df.columns if c not in meta)
    if not np.isfinite(df[feats+["hr_bpm"]].to_numpy(float)).all():raise RuntimeError("Prepared table contains NaN/Inf")
    out=Path(a.out);summary=Path(a.summary_out);out.parent.mkdir(parents=True,exist_ok=True);summary.parent.mkdir(parents=True,exist_ok=True)
    df[meta+feats].to_csv(out,index=False)
    summary.write_text(json.dumps({"version":"PPG V6","dataset":"PhysioNet Pulse Transit Time PPG Dataset","sensor_domain":"MAX30101-family reference","records_found":len(records),"records_processed":len(summaries),"windows":len(df),"subjects":int(df.subject.nunique()),"features":len(feats),"activities":sorted(df.activity.unique()),"window_sec":a.window_sec,"shift_sec":a.shift_sec,"sample_rate_hz":FS_HZ,"feature_columns":feats,"records":summaries,"ecg_waveform_used_as_feature":False,"ecg_peaks_used_as_target_source":True,"spo2_used_as_target":False},indent=2),encoding="utf-8")
    print(json.dumps({"records":len(records),"subjects":int(df.subject.nunique()),"windows":len(df),"features":len(feats),"out":str(out)},indent=2))
if __name__=="__main__":main()
