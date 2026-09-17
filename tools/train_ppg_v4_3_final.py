#!/usr/bin/env python3
"""SafeBand PPG V4.3 FINAL: controlled ablation + leakage-safe HR model.

V4.3 is the final MAX30101-domain reference experiment for the current PTT work.
It does NOT invent new labels or change the dataset split. It compares:
  1) constant train-mean baseline
  2) spectral/peak HR baseline
  3) PPG-only HGB
  4) PPG + motion HGB
and retains the established Ridge/RF/ExtraTrees/HGB full-feature comparison.

Model selection is by validation MAE only. Test subjects are never used for
selection, calibration, or feature selection.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

META={"subject","activity","record","start_sample","start_time","hr_bpm"}

def skey(s):
    s=str(s)
    return int(s[1:]) if s.startswith("s") and s[1:].isdigit() else 10**9

def split(ids):
    ids=sorted(set(map(str,ids)),key=skey)
    n=len(ids); a=round(n*.60); b=round(n*.20)
    if a+b>=n: b=n-a-1
    return ids[:a],ids[a:a+b],ids[a+b:]

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

def pipe(kind):
    if kind=="ridge":
        model=Ridge(alpha=10)
    elif kind=="rf":
        model=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.75,random_state=42,n_jobs=-1)
    elif kind=="extra_trees":
        model=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.80,random_state=42,n_jobs=-1)
    elif kind=="hgb":
        model=HistGradientBoostingRegressor(max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=42)
    else:
        raise ValueError(kind)
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",model)])

def affine(y,p):
    A=np.column_stack([p,np.ones(len(p))])
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    return float(c[0]),float(c[1])

def activity_hr_baseline(df):
    # Baseline based only on window PPG spectral/peak columns, no target-derived
    # feature is introduced. It is deliberately simple and transparent.
    dom=[c for c in df.columns if c.endswith("_dom_bpm")]
    peak=[c for c in df.columns if c.endswith("_peak_hr")]
    cols=dom+peak
    if not cols:
        return None
    x=df[cols].to_numpy(float)
    vals=[]
    for row in x:
        v=row[np.isfinite(row)]
        v=v[(v>=30)&(v<=220)]
        vals.append(float(np.median(v)) if len(v) else np.nan)
    return np.asarray(vals)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--model-out",required=True)
    ap.add_argument("--report-out",required=True)
    ap.add_argument("--predictions-out",required=True)
    args=ap.parse_args()

    df=pd.read_csv(args.input)
    missing=META-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    feat=[c for c in df.columns if c not in META]
    X=df[feat].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    y=pd.to_numeric(df.hr_bpm,errors="coerce").to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Input contains NaN/Inf in model matrix or target.")

    ids=df.subject.astype(str).to_numpy()
    tr_s,va_s,te_s=split(ids)
    tr=np.isin(ids,tr_s); va=np.isin(ids,va_s); te=np.isin(ids,te_s)
    if set(tr_s)&set(va_s) or set(tr_s)&set(te_s) or set(va_s)&set(te_s):
        raise RuntimeError("Subject leakage detected.")

    ppg=[c for c in feat if not (c.startswith("acc_") or c.startswith("gyro_") or "_accmag" in c or "_gyromag" in c)]
    motion=[c for c in feat if c.startswith("acc_") or c.startswith("gyro_") or "_accmag" in c or "_gyromag" in c]
    if not ppg or not motion: raise RuntimeError("Expected both PPG and motion feature groups.")

    print("=== SafeBand PPG V4.3 FINAL ===")
    print(f"Rows={len(df)} Features={len(feat)} PPG={len(ppg)} Motion={len(motion)}")
    print(f"Train={tr_s} ({tr.sum()})")
    print(f"Val  ={va_s} ({va.sum()})")
    print(f"Test ={te_s} ({te.sum()})")

    mean=float(y[tr].mean())
    baseline={}
    for name,mask in [("validation",va),("test",te)]:
        baseline[name]=scores(y[mask],np.full(mask.sum(),mean))
    hrb=activity_hr_baseline(df)
    hr_baseline={}
    if hrb is not None:
        for name,mask in [("validation",va),("test",te)]:
            good=np.isfinite(hrb[mask])
            hr_baseline[name]=scores(y[mask][good],hrb[mask][good])

    val_models={}
    test_models={}
    for name in ["ridge","rf","extra_trees","hgb"]:
        print(f"Training full {name}...")
        m=pipe(name); m.fit(X[tr],y[tr]); val_models[name]=scores(y[va],m.predict(X[va]))
    selected=min(val_models,key=lambda k:val_models[k]["mae_bpm"])
    print(f"Selected full model: {selected}")

    # Controlled ablation: PPG-only vs PPG+motion, both selected with validation MAE.
    ablation={}
    for label,cols in [("ppg_only",ppg),("ppg_plus_motion",feat)]:
        m=pipe("hgb")
        m.fit(df.loc[tr,cols].to_numpy(float),y[tr])
        pv=m.predict(df.loc[va,cols].to_numpy(float))
        ablation[label]=scores(y[va],pv)
        print(f"{label}: VAL MAE={ablation[label]['mae_bpm']:.3f}")

    # Calibration for selected full model, validation only.
    sel=pipe(selected); sel.fit(X[tr],y[tr]); pv=sel.predict(X[va])
    slope,intercept=affine(y[va],pv)
    pcal=slope*pv+intercept
    raw_mae=mean_absolute_error(y[va],pv); cal_mae=mean_absolute_error(y[va],pcal)
    cal_enabled=bool(cal_mae<raw_mae)
    print(f"Calibration: raw={raw_mae:.3f}, calibrated={cal_mae:.3f}, enabled={cal_enabled}")

    final=pipe(selected); final.fit(X[tr|va],y[tr|va])
    pt=final.predict(X[te])
    if cal_enabled: pt=slope*pt+intercept
    test=scores(y[te],pt)
    pred=df.loc[te,["subject","activity","record","start_sample","start_time","hr_bpm"]].copy()
    pred["prediction_bpm"]=pt; pred["error_bpm"]=pt-pred.hr_bpm; pred["abs_error_bpm"]=np.abs(pred.error_bpm)

    # Activity + subject diagnostics in the same final report.
    def grouped(col):
        return {str(k):scores(g.hr_bpm,g.prediction_bpm) for k,g in pred.groupby(col,sort=True)}
    report={
        "version":"PPG V4.3 FINAL",
        "dataset":{"windows":len(df),"subjects":int(df.subject.nunique()),"features":len(feat),
                   "ppg_features":len(ppg),"motion_features":len(motion),
                   "activities":sorted(df.activity.unique().tolist())},
        "split":{"method":"numeric_subject_60_20_20","train_subjects":tr_s,"validation_subjects":va_s,
                 "test_subjects":te_s,"subject_overlap":False,"train_windows":int(tr.sum()),
                 "validation_windows":int(va.sum()),"test_windows":int(te.sum())},
        "baselines":{"constant_train_mean_bpm":mean,"constant":baseline,"ppg_hr_baseline":hr_baseline},
        "validation_models":val_models,
        "selected_model":selected,
        "ablation_validation":ablation,
        "validation_calibration":{"enabled":cal_enabled,"slope":slope,"intercept":intercept,
                                  "raw_mae_bpm":float(raw_mae),"calibrated_mae_bpm":float(cal_mae)},
        "held_out_test":test,
        "held_out_test_by_activity":grouped("activity"),
        "held_out_test_by_subject":grouped("subject"),
        "artifact":str(args.model_out),
        "predictions":str(args.predictions_out),
        "notes":[
            "Regression task: classification accuracy is not the primary metric.",
            "ECG waveform is not a feature; ECG peaks provide HR target.",
            "SpO2 is not used as a continuous training target.",
            "Model selection uses validation MAE only; test remains untouched.",
            "V4.3 keeps the established 159-feature representation; the new experiment is controlled PPG-only vs PPG+motion ablation and final diagnostics.",
            "This is a MAX30101-domain reference experiment and is not a MAX30102 hardware validation result."
        ]
    }
    Path(args.model_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report_out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.predictions_out).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":final,"feature_names":feat,"model_type":selected,
                 "calibration":{"enabled":cal_enabled,"slope":slope,"intercept":intercept},
                 "version":"PPG V4.3 FINAL","domain_warning":"MAX30101 reference; validate on MAX30102 hardware."},
                args.model_out)
    pred.to_csv(args.predictions_out,index=False)
    Path(args.report_out).write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("\n=== HELD-OUT TEST ===")
    for k,v in test.items(): print(f"{k}: {v:.4f}" if isinstance(v,float) else f"{k}: {v}")
    print("\n=== ACTIVITY ===")
    for k,v in report["held_out_test_by_activity"].items():
        print(f"{k}: MAE={v['mae_bpm']:.3f} ±5={v['within_5_bpm_pct']:.2f}% Bias={v['bias_bpm']:+.3f}")
    print("\n=== SUBJECT ===")
    for k,v in report["held_out_test_by_subject"].items():
        print(f"{k}: MAE={v['mae_bpm']:.3f} ±5={v['within_5_bpm_pct']:.2f}% Bias={v['bias_bpm']:+.3f}")

if __name__=="__main__": main()
