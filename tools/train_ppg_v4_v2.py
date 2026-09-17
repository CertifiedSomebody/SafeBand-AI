#!/usr/bin/env python3
"""Leakage-safe PPG V4.2 model comparison + baseline diagnostics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def scores(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); e=np.abs(y-p)
    return {
        "n":int(len(y)),
        "mae_bpm":float(mean_absolute_error(y,p)),
        "rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
        "r2":float(r2_score(y,p)),
        "within_3_bpm_pct":float(np.mean(e<=3)*100),
        "within_5_bpm_pct":float(np.mean(e<=5)*100),
        "within_10_bpm_pct":float(np.mean(e<=10)*100),
        "bias_bpm":float(np.mean(p-y)),
        "median_abs_error_bpm":float(np.median(e)),
        "p90_abs_error_bpm":float(np.percentile(e,90)),
        "max_abs_error_bpm":float(np.max(e)),
    }

def subject_key(s):
    s=str(s); return int(s[1:]) if s.startswith("s") and s[1:].isdigit() else s

def split(ids):
    ids=sorted(set(map(str,ids)),key=subject_key)
    n=len(ids); a=round(n*.60); b=round(n*.20)
    a=max(1,a); b=max(1,b)
    if a+b>=n: b=n-a-1
    return ids[:a],ids[a:a+b],ids[a+b:]

def models():
    return {
        "ridge":Pipeline([("impute",SimpleImputer(strategy="median")),("model",Ridge(alpha=10))]),
        "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestRegressor(
            n_estimators=700,min_samples_leaf=2,max_features=.75,random_state=42,n_jobs=-1))]),
        "extra_trees":Pipeline([("impute",SimpleImputer(strategy="median")),("model",ExtraTreesRegressor(
            n_estimators=700,min_samples_leaf=2,max_features=.80,random_state=42,n_jobs=-1))]),
        "hgb":Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(
            max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=42))]),
    }

def affine(y,p):
    A=np.column_stack([p,np.ones(len(p))])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    return float(c[0]),float(c[1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True); ap.add_argument("--model-out",required=True)
    ap.add_argument("--report-out",required=True); ap.add_argument("--predictions-out",required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    missing=META-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    feats=[c for c in df.columns if c not in META]
    X=df[feats].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    y=pd.to_numeric(df.hr_bpm,errors="coerce").to_numpy(float)
    subjects=df.subject.astype(str).to_numpy()
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError("NaN/Inf in input")
    tr_s,va_s,te_s=split(subjects)
    tr=np.isin(subjects,tr_s); va=np.isin(subjects,va_s); te=np.isin(subjects,te_s)
    if set(tr_s)&set(va_s) or set(tr_s)&set(te_s) or set(va_s)&set(te_s): raise RuntimeError("Subject leakage")
    print("=== PPG V4.2 ===")
    print(f"Rows={len(df)} Features={len(feats)} Subjects={len(set(subjects))}")
    print(f"TRAIN {tr_s} windows={tr.sum()}")
    print(f"VAL   {va_s} windows={va.sum()}")
    print(f"TEST  {te_s} windows={te.sum()}")
    print("\n--- CONSTANT TRAIN-MEAN BASELINE ---")
    mean=float(np.mean(y[tr]))
    for name,mask in [("VAL",va),("TEST",te)]:
        s=scores(y[mask],np.full(mask.sum(),mean))
        print(f"{name}: MAE={s['mae_bpm']:.3f} RMSE={s['rmse_bpm']:.3f} R2={s['r2']:.3f} ±5={s['within_5_bpm_pct']:.2f}%")
    ms=models(); val={}; vp={}
    for name,m in ms.items():
        print(f"\nTraining {name}...")
        m.fit(X[tr],y[tr]); p=m.predict(X[va]); vp[name]=p; val[name]=scores(y[va],p)
        s=val[name]
        print(f"VAL {name}: MAE={s['mae_bpm']:.3f} RMSE={s['rmse_bpm']:.3f} R2={s['r2']:.3f} ±3={s['within_3_bpm_pct']:.2f}% ±5={s['within_5_bpm_pct']:.2f}%")
    selected=min(val,key=lambda k:val[k]["mae_bpm"])
    slope,intercept=affine(y[va],vp[selected])
    calp=slope*vp[selected]+intercept
    raw_mae=mean_absolute_error(y[va],vp[selected]); cal_mae=mean_absolute_error(y[va],calp)
    enabled=bool(cal_mae<raw_mae)
    print(f"\nSelected: {selected}")
    print(f"Calibration validation MAE: raw={raw_mae:.3f}, calibrated={cal_mae:.3f}, enabled={enabled}")
    final=ms[selected]; final.fit(X[tr|va],y[tr|va])
    pr=final.predict(X[te]); p= slope*pr+intercept if enabled else pr
    test=scores(y[te],p)
    pred=df.loc[te,["subject","activity","record","start_sample","start_time","hr_bpm"]].copy()
    pred["prediction_bpm"]=p; pred["error_bpm"]=p-pred.hr_bpm; pred["abs_error_bpm"]=np.abs(pred.error_bpm)
    Path(args.model_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.predictions_out).parent.mkdir(parents=True,exist_ok=True)
    artifact={"model":final,"feature_names":feats,"model_type":selected,"version":"PPG V4.2",
              "calibration":{"enabled":enabled,"slope":slope,"intercept":intercept},
              "train_subjects":tr_s,"validation_subjects":va_s,"test_subjects":te_s,
              "domain_warning":"MAX30101-domain reference; not MAX30102 validation."}
    joblib.dump(artifact,args.model_out); pred.to_csv(args.predictions_out,index=False)
    report={"version":"PPG V4.2","dataset":{"windows":len(df),"subjects":len(set(subjects)),"features":len(feats),
             "activities":sorted(df.activity.unique().tolist())},
            "split":{"method":"numeric_subject_60_20_20","train_subjects":tr_s,"validation_subjects":va_s,
                     "test_subjects":te_s,"subject_overlap":False,"train_windows":int(tr.sum()),
                     "validation_windows":int(va.sum()),"test_windows":int(te.sum())},
            "baseline":{"train_mean_bpm":mean,"validation":scores(y[va],np.full(va.sum(),mean)),
                        "test":scores(y[te],np.full(te.sum(),mean))},
            "validation":val,"selected_model":selected,
            "validation_calibration":{"enabled":enabled,"slope":slope,"intercept":intercept,
                                      "raw_mae_bpm":float(raw_mae),"calibrated_mae_bpm":float(cal_mae)},
            "held_out_test":test,"artifact":str(args.model_out),"predictions":str(args.predictions_out),
            "notes":["Regression task: classification accuracy is not the primary metric.",
                     "ECG waveform is not a feature; ECG peaks provide HR target.",
                     "SpO2 is not used as a continuous training target.",
                     "Model selection uses validation MAE only; test remains untouched."]}
    Path(args.report_out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("\n=== FINAL HELD-OUT TEST ===")
    for k,v in test.items(): print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")

if __name__=="__main__": main()
