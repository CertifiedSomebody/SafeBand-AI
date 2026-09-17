"""Train the SafeBand BITS-2 motion activity reference model.

This is a Phase-1 benchmark, not the final hardware model. BITS-2 does not
provide a clean STANDING class, so the learned activity set is:
RESTING, SITTING, WALKING, RUNNING. FALL remains a separate binary detector.

All splits are subject-disjoint. The held-out test subjects are untouched
until model selection is complete.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Dict,Tuple
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
from ai.feature_extraction import FEATURE_COLUMNS

def split_by_subject(df,seed):
    labels=set(df.target)
    for attempt in range(200):
        s=seed+attempt
        a=GroupShuffleSplit(1,test_size=.20,random_state=s)
        trv_i,te_i=next(a.split(df,groups=df.subject_id)); trv,te=df.iloc[trv_i],df.iloc[te_i]
        b=GroupShuffleSplit(1,test_size=.25,random_state=s+1000)
        tr_i,va_i=next(b.split(trv,groups=trv.subject_id)); tr,va=trv.iloc[tr_i],trv.iloc[va_i]
        if all(set(x.target)==labels for x in (tr,va,te)): return tr,va,te
    raise RuntimeError("Could not construct a subject-disjoint split containing every class.")

def metrics(y,p,labels):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_precision":float(pr),"macro_recall":float(rc),"macro_f1":float(f1),
            "confusion_matrix":confusion_matrix(y,p,labels=labels).tolist()}

def models(seed):
    return {
      "logistic":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=seed))]),
      "rf":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=500,min_samples_leaf=2,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=seed))]),
      "extra_trees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",n_jobs=-1,random_state=seed))]),
      "hgb":Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=350,learning_rate=.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=seed))])
    }

def train_task(df,seed,fall):
    tr,va,te=split_by_subject(df,seed); labs=sorted(df.target.unique()); res={}; ms=models(seed)
    for name,m in ms.items():
        m.fit(tr[FEATURE_COLUMNS],tr.target); res[name]=metrics(va.target,m.predict(va[FEATURE_COLUMNS]),labs)
        if fall:
            from sklearn.metrics import recall_score
            res[name]["fall_recall"]=float(recall_score(va.target,m.predict(va[FEATURE_COLUMNS]),pos_label="FALL",zero_division=0))
    key=(lambda n:(res[n]["fall_recall"],res[n]["macro_f1"],res[n]["balanced_accuracy"])) if fall else (lambda n:(res[n]["macro_f1"],res[n]["balanced_accuracy"]))
    selected=max(res,key=key); model=ms[selected]
    dev=pd.concat([tr,va],ignore_index=True); model.fit(dev[FEATURE_COLUMNS],dev.target)
    pred=model.predict(te[FEATURE_COLUMNS]); test=metrics(te.target,pred,labs)
    if fall:
        from sklearn.metrics import recall_score
        test["fall_recall"]=float(recall_score(te.target,pred,pos_label="FALL",zero_division=0))
    return model,selected,res,test,labs,confusion_matrix(te.target,pred,labels=labs).tolist(),{
        "train_subjects":sorted(tr.subject_id.unique()),"validation_subjects":sorted(va.subject_id.unique()),"test_subjects":sorted(te.subject_id.unique()),
        "train_windows":len(tr),"validation_windows":len(va),"test_windows":len(te)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",required=True); ap.add_argument("--model-dir",default=str(ROOT/"models"/"activity_bits2_v2")); ap.add_argument("--report",default=str(ROOT/"models"/"activity_bits2_v2"/"report.json")); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); df=pd.read_csv(a.input)
    need=set(FEATURE_COLUMNS)|{"subject_id","activity_label"}
    miss=sorted(need-set(df.columns))
    if miss: raise ValueError(f"Missing required columns: {miss}")
    df=df.dropna(subset=FEATURE_COLUMNS+["subject_id","activity_label"]).copy(); df.subject_id=df.subject_id.astype(str)
    activity=df[df.activity_label.isin(["RESTING","SITTING","WALKING","RUNNING"])].copy(); activity["target"]=activity.activity_label
    fall=df.copy(); fall["target"]=np.where(fall.activity_label.eq("FALL"),"FALL","NON_FALL")
    out=Path(a.model_dir); out.mkdir(parents=True,exist_ok=True)
    report={"version":"SafeBand BITS-2 Motion V2","input":str(Path(a.input)),"feature_count":len(FEATURE_COLUMNS),
            "features":FEATURE_COLUMNS,"window_samples":int(df.window_samples.iloc[0]) if "window_samples" in df else None,
            "assumed_sample_rate_hz":20,"notes":["BITS-2 has no clean STANDING class.","Activity and fall tasks are trained separately."]}
    for name,task,fall_task,file in [("activity",activity,False,"activity_model.joblib"),("fall_detector",fall,True,"fall_detector.joblib")]:
        model,selected,val,test,labs,cm,split=train_task(task,a.seed,fall_task)
        payload={"model":model,"feature_columns":FEATURE_COLUMNS,"model_name":"SafeBand BITS-2 Motion Activity Model" if name=="activity" else "SafeBand BITS-2 Fall Detector",
                 "model_version":"bits2_motion_v2","label_classes":labs,"task":name,"window_samples":int(df.window_samples.iloc[0]) if "window_samples" in df else None,
                 "sample_rate_hz":20}
        joblib.dump(payload,out/file)
        report[name]={"selected_model":selected,"validation":val,"held_out_test":test,"labels":labs,"confusion_matrix":cm,"split":split}
        print(f"{name}: selected={selected} | test={test}")
    rp=Path(a.report); rp.parent.mkdir(parents=True,exist_ok=True); rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Saved report: {rp}")
if __name__=="__main__":main()
