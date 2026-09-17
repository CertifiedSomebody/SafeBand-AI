from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import argparse, json, joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

from ai.hr_estimator import make_features


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    m=np.isfinite(y)&np.isfinite(p); y=y[m]; p=p[m]
    e=np.abs(y-p)
    return {
        "n":int(len(y)),
        "mae_bpm":float(mean_absolute_error(y,p)),
        "rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
        "r2":float(r2_score(y,p)),
        "within_3_bpm_pct":float(np.mean(e<=3)*100),
        "within_5_bpm_pct":float(np.mean(e<=5)*100),
        "within_10_bpm_pct":float(np.mean(e<=10)*100),
        "bias_bpm":float(np.mean(p-y)),
    }


def split(subjects):
    ids=np.array(sorted(set(subjects.tolist())))
    if len(ids)<3: raise ValueError("Need >=3 subjects.")
    nt=round(.60*len(ids)); nv=round(.20*len(ids))
    return ids[:nt],ids[nt:nt+nv],ids[nt+nv:]


def build(z):
    bvp,acc=z["bvp"],z["acc"]
    fs1,fs2=float(z["bvp_fs"]),float(z["acc_fs"])
    names=None; X=np.empty((len(bvp),0),np.float32)
    rows=[]
    for i in range(len(bvp)):
        d=make_features(bvp[i],acc[i],fs1,fs2)
        rows.append(d)
        if names is None: names=sorted(d)
    X=np.asarray([[d[k] for k in names] for d in rows],np.float32)
    return X,names


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--model-out",required=True)
    ap.add_argument("--report-out",required=True)
    args=ap.parse_args()

    z=np.load(args.input,allow_pickle=True)
    required={"bvp","acc","hr","subject","bvp_fs","acc_fs"}
    missing=required-set(z.files)
    if missing: raise ValueError(f"Missing arrays: {sorted(missing)}")

    bvp,acc=z["bvp"],z["acc"]; y=z["hr"].astype(float)
    subjects=z["subject"].astype(str)
    if not(len(bvp)==len(acc)==len(y)==len(subjects)): raise ValueError("Length mismatch")
    if not(np.isfinite(bvp).all() and np.isfinite(acc).all() and np.isfinite(y).all()):
        raise ValueError("NaN/Inf in input")

    ts,vs,es=split(subjects)
    tr=np.isin(subjects,ts); va=np.isin(subjects,vs); te=np.isin(subjects,es)
    if set(ts)&set(vs) or set(ts)&set(es) or set(vs)&set(es):
        raise RuntimeError("Subject leakage detected")

    print(f"Windows={len(y)} subjects={len(set(subjects))}")
    print(f"Train={tr.sum()} Val={va.sum()} Test={te.sum()}")
    print("Extracting V3 features...")
    X,names=build(z)

    models={
      "ridge":Pipeline([("impute",SimpleImputer(strategy="median")),("model",Ridge(alpha=10))]),
      "rf":Pipeline([("impute",SimpleImputer(strategy="median")),("model",RandomForestRegressor(
          n_estimators=700,min_samples_leaf=2,max_features=.75,random_state=42,n_jobs=-1))]),
      "extra_trees":Pipeline([("impute",SimpleImputer(strategy="median")),("model",ExtraTreesRegressor(
          n_estimators=700,min_samples_leaf=2,max_features=.8,random_state=42,n_jobs=-1))]),
      "hgb":Pipeline([("impute",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(
          max_iter=450,learning_rate=.04,max_leaf_nodes=31,l2_regularization=1.5,random_state=42))])
    }

    val={}
    val_pred={}
    for n,m in models.items():
        print("Training",n)
        m.fit(X[tr],y[tr])
        pv=m.predict(X[va]); val_pred[n]=pv; val[n]=metrics(y[va],pv)
        print(f"  validation MAE={val[n]['mae_bpm']:.3f}")

    selected=min(val,key=lambda k:val[k]["mae_bpm"])

    # Conservative affine calibration is selected only if validation MAE improves.
    # It is fitted on validation predictions, then applied to the final test.
    raw=val_pred[selected]
    A=np.column_stack([raw,np.ones(len(raw))])
    coef=np.linalg.lstsq(A,y[va],rcond=None)[0]
    cal=A@coef
    raw_mae=mean_absolute_error(y[va],raw)
    cal_mae=mean_absolute_error(y[va],cal)
    use_cal=cal_mae < raw_mae
    calibration={"enabled":bool(use_cal),"slope":float(coef[0]),"intercept":float(coef[1]),
                 "validation_raw_mae":float(raw_mae),"validation_calibrated_mae":float(cal_mae)}

    final=models[selected]
    final.fit(X[tr|va],y[tr|va])
    test_raw=final.predict(X[te])
    test_pred=coef[0]*test_raw+coef[1] if use_cal else test_raw
    test=metrics(y[te],test_pred)

    artifact={"model":final,"feature_names":names,
              "bvp_fs_hz":float(z["bvp_fs"]),"acc_fs_hz":float(z["acc_fs"]),
              "window_sec":8.0,"shift_sec":2.0,
              "source_sensor":"Empatica E4 wrist BVP + wrist ACC",
              "target":"ECG-derived HR BPM","model_type":selected,
              "calibration":calibration,
              "training_subjects":ts.tolist(),"validation_subjects":vs.tolist(),
              "test_subjects":es.tolist(),
              "domain_warning":"Reference model only; not MAX30102 validated."}

    mo=Path(args.model_out); mo.parent.mkdir(parents=True,exist_ok=True); joblib.dump(artifact,mo)
    report={"dataset":{"windows":len(y),"subjects":len(set(subjects))},
            "split":{"method":"deterministic_subject_level_60_20_20",
                     "train_subjects":ts.tolist(),"validation_subjects":vs.tolist(),
                     "test_subjects":es.tolist(),"train_windows":int(tr.sum()),
                     "validation_windows":int(va.sum()),"test_windows":int(te.sum()),
                     "subject_overlap":False},
            "validation":val,"selected_model":selected,
            "validation_calibration":calibration,
            "held_out_test":test,"model_artifact":str(mo),
            "feature_names":names,
            "notes":["V3 adds conservative temporal/physiological continuity support for deployment.",
                     "Affine calibration is validation-selected only.",
                     "Final estimator is refit on train+validation.",
                     "Held-out test subjects are never used for selection.",
                     "E4 BVP is a reference sensor; MAX30102 transfer is not claimed."]}
    ro=Path(args.report_out); ro.parent.mkdir(parents=True,exist_ok=True)
    ro.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"\nSelected={selected}")
    print(f"Test MAE={test['mae_bpm']:.3f} | RMSE={test['rmse_bpm']:.3f} | R2={test['r2']:.3f}")
    print(f"±5={test['within_5_bpm_pct']:.2f}% | ±10={test['within_10_bpm_pct']:.2f}%")
    print("Artifact:",mo)
    print("Report:",ro)


if __name__=="__main__": main()
