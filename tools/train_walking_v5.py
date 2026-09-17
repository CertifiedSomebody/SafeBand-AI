#!/usr/bin/env python3
"""Walking-focused V5 model.

Keeps the established SafeBand principles: 8 s windows, 2 s shift, subject-level
split, Ridge/RF/ExtraTrees/HGB comparison, validation-MAE selection, validation-only
affine calibration, untouched test subjects, and full diagnostics.

The external Wrist dataset is a separate domain. This script trains only on the
walking dataset supplied to it; it does not contaminate the frozen V4.3 benchmark.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def skey(s):
    s=str(s); return int(s[1:]) if s.startswith("s") and s[1:].isdigit() else 10**9

def split(ids):
    ids=sorted(set(map(str,ids)),key=skey); n=len(ids); a=round(n*.60); b=round(n*.20)
    if a+b>=n: b=n-a-1
    return ids[:a],ids[a:a+b],ids[a+b:]

def scores(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=np.abs(y-p)
    return {"n":int(len(y)),"mae_bpm":float(mean_absolute_error(y,p)),"rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p)),"within_3_bpm_pct":float(np.mean(e<=3)*100),"within_5_bpm_pct":float(np.mean(e<=5)*100),"within_10_bpm_pct":float(np.mean(e<=10)*100),"bias_bpm":float(np.mean(p-y)),"median_abs_error_bpm":float(np.median(e)),"p90_abs_error_bpm":float(np.percentile(e,90)),"p95_abs_error_bpm":float(np.percentile(e,95)),"max_abs_error_bpm":float(np.max(e))}

def pipe(k):
    if k=="ridge": m=Ridge(alpha=10)
    elif k=="rf": m=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.75,random_state=42,n_jobs=-1)
    elif k=="extra_trees": m=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.80,random_state=42,n_jobs=-1)
    elif k=="hgb": m=HistGradientBoostingRegressor(max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=42)
    else: raise ValueError(k)
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",m)])

def affine(y,p):
    A=np.column_stack([p,np.ones(len(p))]); c,*_=np.linalg.lstsq(A,y,rcond=None); return float(c[0]),float(c[1])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--model-out",required=True); ap.add_argument("--report-out",required=True); ap.add_argument("--predictions-out",required=True); args=ap.parse_args()
    df=pd.read_csv(args.input); feat=[c for c in df.columns if c not in META]; X=df[feat].to_numpy(float); y=df.hr_bpm.to_numpy(float); ids=df.subject.astype(str).to_numpy(); tr_s,va_s,te_s=split(ids)
    tr=np.isin(ids,tr_s); va=np.isin(ids,va_s); te=np.isin(ids,te_s)
    if set(tr_s)&set(va_s) or set(tr_s)&set(te_s) or set(va_s)&set(te_s): raise RuntimeError("Subject leakage detected")
    baseline=float(y[tr].mean()); val_models={}
    for k in ["ridge","rf","extra_trees","hgb"]:
        m=pipe(k); m.fit(X[tr],y[tr]); val_models[k]=scores(y[va],m.predict(X[va]))
    selected=min(val_models,key=lambda k:val_models[k]["mae_bpm"])
    sel=pipe(selected); sel.fit(X[tr],y[tr]); pv=sel.predict(X[va]); slope,intercept=affine(y[va],pv); pcal=slope*pv+intercept
    raw=float(mean_absolute_error(y[va],pv)); cal=float(mean_absolute_error(y[va],pcal)); enabled=cal<raw
    final=pipe(selected); final.fit(X[tr|va],y[tr|va]); pt=final.predict(X[te]);
    if enabled: pt=slope*pt+intercept
    pred=df.loc[te,["subject","activity","record","start_sample","start_time","hr_bpm"]].copy(); pred["prediction_bpm"]=pt; pred["error_bpm"]=pt-pred.hr_bpm; pred["abs_error_bpm"]=np.abs(pred.error_bpm)
    def grouped(col): return {str(k):scores(g.hr_bpm,g.prediction_bpm) for k,g in pred.groupby(col,sort=True)}
    report={"version":"PPG WALKING V5.0","dataset":{"windows":len(df),"subjects":int(df.subject.nunique()),"features":len(feat),"activity":"walk"},"split":{"method":"numeric_subject_60_20_20","train_subjects":tr_s,"validation_subjects":va_s,"test_subjects":te_s,"subject_overlap":False,"train_windows":int(tr.sum()),"validation_windows":int(va.sum()),"test_windows":int(te.sum())},"baseline":{"constant_train_mean_bpm":baseline,"validation":scores(y[va],np.full(va.sum(),baseline)),"test":scores(y[te],np.full(te.sum(),baseline))},"validation_models":val_models,"selected_model":selected,"validation_calibration":{"enabled":enabled,"slope":slope,"intercept":intercept,"raw_mae_bpm":raw,"calibrated_mae_bpm":cal},"held_out_test":scores(y[te],pt),"held_out_test_by_subject":grouped("subject"),"held_out_test_by_record":grouped("record"),"notes":["Walking-focused external dataset experiment; not a replacement for the frozen V4.3 MAX30101 benchmark.","ECG R-peak annotations provide HR target; ECG waveform is never a model feature.","8 s windows and 2 s shift are retained from the established SafeBand protocol.","Model selection uses validation MAE only; held-out test subjects remain untouched."]}
    Path(args.model_out).parent.mkdir(parents=True,exist_ok=True); Path(args.report_out).parent.mkdir(parents=True,exist_ok=True); Path(args.predictions_out).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":final,"feature_names":feat,"model_type":selected,"calibration":{"enabled":enabled,"slope":slope,"intercept":intercept},"version":"PPG WALKING V5.0"},args.model_out); pred.to_csv(args.predictions_out,index=False); Path(args.report_out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
