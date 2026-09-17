#!/usr/bin/env python3
"""Robust walking HR experiment for SafeBand PPG V5.2.

Key protocol:
- S9 is a completely untouched external test subject.
- Development subjects are S1,S2,S3,S6,S8.
- GroupKFold over development subjects supplies five out-of-fold folds.
- Variant/model selection uses mean OOF MAE, not one validation subject.
- Calibration is bounded and optional; it is learned only from OOF predictions.
- Final model is fitted on all development subjects, then evaluated once on S9.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

META={"subject","activity","record","start_sample","start_time","hr_bpm"}
DEV_SUBJECTS=["s1","s2","s3","s6","s8"]
TEST_SUBJECT="s9"
RANDOM_STATE=42

def scores(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=np.abs(y-p)
    return {"n":int(len(y)),"mae_bpm":float(mean_absolute_error(y,p)),
      "rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p)),
      "within_3_bpm_pct":float(np.mean(e<=3)*100),"within_5_bpm_pct":float(np.mean(e<=5)*100),
      "within_10_bpm_pct":float(np.mean(e<=10)*100),"bias_bpm":float(np.mean(p-y)),
      "median_abs_error_bpm":float(np.median(e)),"p90_abs_error_bpm":float(np.percentile(e,90)),
      "p95_abs_error_bpm":float(np.percentile(e,95)),"max_abs_error_bpm":float(np.max(e))}

def make_model(k):
    if k=="ridge":m=Ridge(alpha=10)
    elif k=="rf":m=RandomForestRegressor(n_estimators=700,min_samples_leaf=2,max_features=.75,random_state=RANDOM_STATE,n_jobs=-1)
    elif k=="extra_trees":m=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=.80,random_state=RANDOM_STATE,n_jobs=-1)
    elif k=="hgb":m=HistGradientBoostingRegressor(max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=RANDOM_STATE)
    else:raise ValueError(k)
    return Pipeline([("impute",SimpleImputer(strategy="median")),("model",m)])

def features_for(allf,v):
    if v=="ppg_only": return [c for c in allf if not (c.startswith(("acc_","gyro_","wacc_","mag_")) or c.startswith("corr_ppg_"))]
    if v=="ppg_plus_gyro": return [c for c in allf if not (c.startswith(("acc_","wacc_","mag_")) or c=="corr_ppg_accmag" or c=="corr_ppg_waccmag" or c=="corr_ppg_magmag")]
    if v=="ppg_plus_acc_gyro": return [c for c in allf if not (c.startswith(("wacc_","mag_")) or c in ("corr_ppg_waccmag","corr_ppg_magmag"))]
    if v=="all_motion": return list(allf)
    raise ValueError(v)

def bounded_affine_fit(y,p,lo=.80,hi=1.20):
    A=np.column_stack([p,np.ones(len(p))]); c=np.linalg.lstsq(A,y,rcond=None)[0]
    slope=float(np.clip(c[0],lo,hi)); intercept=float(np.mean(y-slope*p))
    return slope,intercept

def evaluate_oof(X,y,groups,variant,model_name,calibration):
    gkf=GroupKFold(n_splits=len(np.unique(groups))); oof=np.full(len(y),np.nan)
    fold=[]
    for fold_id,(tr,va) in enumerate(gkf.split(X,y,groups),1):
        m=make_model(model_name);m.fit(X[tr],y[tr]);p=m.predict(X[va]);oof[va]=p
        fold.append({"fold":fold_id,"validation_subjects":sorted(set(groups[va])),"metrics":scores(y[va],p)})
    if calibration:
        slope,intercept=bounded_affine_fit(y,oof)
        pc=slope*oof+intercept
        cal=scores(y,pc)
    else:
        slope,intercept=1.0,0.0;pc=oof;cal=scores(y,oof)
    return oof,fold,slope,intercept,cal

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True);ap.add_argument("--model-out",required=True)
    ap.add_argument("--report-out",required=True);ap.add_argument("--predictions-out",required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.input); allf=[c for c in df.columns if c not in META]
    ids=df.subject.astype(str)
    if set(ids)!=set(DEV_SUBJECTS+[TEST_SUBJECT]): raise RuntimeError(f"Expected subjects {DEV_SUBJECTS+[TEST_SUBJECT]}, got {sorted(set(ids))}")
    dev=df.subject.isin(DEV_SUBJECTS).to_numpy(); test=(df.subject==TEST_SUBJECT).to_numpy()
    y=df.hr_bpm.to_numpy(float)
    variants=["ppg_only","ppg_plus_gyro","ppg_plus_acc_gyro","all_motion"]; models=["ridge","rf","extra_trees","hgb"]
    results={}
    best=None
    for v in variants:
        feats=features_for(allf,v); X=df[feats].to_numpy(float)
        results[v]={"feature_count":len(feats),"models":{}}
        for mn in models:
            for cal in (False,True):
                _,fold,slope,intercept,met=evaluate_oof(X[dev],y[dev],df.loc[dev,"subject"].to_numpy(),v,mn,cal)
                key=mn+("_bounded_cal" if cal else "_raw")
                results[v]["models"][key]={"metrics":met,"calibration":{"enabled":cal,"slope":slope,"intercept":intercept},"folds":fold}
                if best is None or met["mae_bpm"]<best["mae"]:
                    best={"mae":met["mae_bpm"],"variant":v,"model":mn,"calibration":cal,"features":feats,
                          "slope":slope,"intercept":intercept}
    X=df[best["features"]].to_numpy(float)
    final=make_model(best["model"]); final.fit(X[dev],y[dev]); pred=final.predict(X[test])
    if best["calibration"]: pred=best["slope"]*pred+best["intercept"]
    out=df.loc[test,["subject","activity","record","start_sample","start_time","hr_bpm"]].copy()
    out["prediction_bpm"]=pred;out["error_bpm"]=pred-out.hr_bpm;out["abs_error_bpm"]=np.abs(out.error_bpm)
    report={"version":"PPG WALKING V5.2 ROBUST",
      "objective":"Improve walking HR generalization while keeping S9 completely untouched during development.",
      "dataset":{"windows":len(df),"subjects":sorted(set(ids)),"development_subjects":DEV_SUBJECTS,"test_subject":TEST_SUBJECT,"features_total":len(allf)},
      "parameters":{"window_sec":8.0,"shift_sec":2.0,"fs_hz":256.0,"ppg_band_hz":[0.5,5.0],
                    "models":models,"random_state":RANDOM_STATE,"cv":"5-fold GroupKFold over five development subjects",
                    "selection_metric":"mean out-of-fold MAE","calibration":"bounded affine slope [0.80,1.20], learned from development OOF predictions only"},
      "validation_results":results,
      "selected":{k:v for k,v in best.items() if k not in ("features","mae")},
      "selected_feature_count":len(best["features"]),
      "held_out_test":scores(y[test],pred),
      "held_out_test_by_record":{},
      "notes":[
        "S9 is never used for feature/model/calibration selection.",
        "Unlike V5.1, no single validation subject determines the calibration.",
        "Unrestricted negative-slope calibration is intentionally prohibited.",
        "Wide-range accelerometer and magnetometer are included only as a controlled all-motion variant.",
        "ECG annotations are used only to derive the HR target; ECG waveform is never a model feature.",
        "This remains an external walking stress experiment, not a population-level clinical validation."
      ]}
    for rec,g in out.groupby("record"): report["held_out_test_by_record"][str(rec)]=scores(g.hr_bpm,g.prediction_bpm)
    for p in [args.model_out,args.report_out,args.predictions_out]:Path(p).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":final,"feature_names":best["features"],"variant":best["variant"],"model_type":best["model"],
                 "calibration":{"enabled":best["calibration"],"slope":best["slope"],"intercept":best["intercept"]},
                 "version":"PPG WALKING V5.2 ROBUST"},args.model_out)
    out.to_csv(args.predictions_out,index=False);Path(args.report_out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
