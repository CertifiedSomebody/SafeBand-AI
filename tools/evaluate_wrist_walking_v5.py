#!/usr/bin/env python3
"""Evaluate a walking HR model on an untouched subject split."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def skey(s):
    s=str(s); return int(s[1:]) if s.startswith("s") and s[1:].isdigit() else 10**9

def scores(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=np.abs(y-p)
    return {"n":int(len(y)),"mae_bpm":float(mean_absolute_error(y,p)),"rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)),"within_3_bpm_pct":float(np.mean(e<=3)*100),
            "within_5_bpm_pct":float(np.mean(e<=5)*100),"within_10_bpm_pct":float(np.mean(e<=10)*100),
            "bias_bpm":float(np.mean(p-y)),"median_abs_error_bpm":float(np.median(e)),
            "p90_abs_error_bpm":float(np.percentile(e,90)),"p95_abs_error_bpm":float(np.percentile(e,95)),
            "max_abs_error_bpm":float(np.max(e))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    df=pd.read_csv(args.input); bundle=joblib.load(args.model)
    feats=bundle["feature_names"]
    missing=[c for c in feats if c not in df.columns]
    if missing: raise ValueError(f"Missing model features: {missing}")
    X=df[feats].to_numpy(float); y=df.hr_bpm.to_numpy(float); pred=bundle["model"].predict(X)
    cal=bundle.get("calibration",{})
    if cal.get("enabled"): pred=float(cal["slope"])*pred+float(cal["intercept"])
    out=df[["subject","activity","record","start_sample","start_time","hr_bpm"]].copy(); out["prediction_bpm"]=pred; out["error_bpm"]=pred-y; out["abs_error_bpm"]=np.abs(pred-y)
    def grouped(col): return {str(k):scores(g.hr_bpm,g.prediction_bpm) for k,g in out.groupby(col,sort=True)}
    report={"version":"PPG WALKING V5.0 EVALUATION","model":args.model,"overall":scores(y,pred),"by_subject":grouped("subject"),"by_record":grouped("record"),"worst_subject":min(grouped("subject"),key=lambda k:grouped("subject")[k]["mae_bpm"]*-1),"worst_record":max(grouped("record"),key=lambda k:grouped("record")[k]["mae_bpm"])}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
