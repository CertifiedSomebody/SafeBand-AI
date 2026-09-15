"""
SafeBand AI - BITS-2 Model Training v2

Trains two subject-independent models:
  activity_model.joblib : RESTING/SITTING/WALKING/RUNNING
  fall_detector.joblib  : FALL/NON_FALL

Key methodological rule:
  splitting happens by subject BEFORE model fitting.

Validation is used for model selection and fall-threshold selection.
The test set is touched only once for final reporting.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


from ai.bits2_features import FEATURE_COLUMNS

SEED = 42

def split_subjects(subjects):
    subjects = sorted(set(str(s) for s in subjects), key=lambda x: int(x) if x.isdigit() else x)
    rng = np.random.default_rng(SEED)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, round(n*0.20))
    n_val = max(1, round(n*0.20))
    test = sorted(subjects[:n_test], key=lambda x:int(x) if x.isdigit() else x)
    val = sorted(subjects[n_test:n_test+n_val], key=lambda x:int(x) if x.isdigit() else x)
    train = sorted(subjects[n_test+n_val:], key=lambda x:int(x) if x.isdigit() else x)
    return train,val,test

def models():
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=150, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))
        ]),

    }

def evaluate(model, X, y, labels):
    pred = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro")),
        "classification_report": classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),
        "confusion_matrix": confusion_matrix(y,pred,labels=labels).tolist(),
    }

def fall_threshold_search(model, X, y):
    if not hasattr(model, "predict_proba"):
        return 0.50
    proba = model.predict_proba(X)
    classes = [str(c).upper() for c in model.classes_]
    idx = classes.index("FALL")
    p = proba[:,idx]
    best = None
    for threshold in np.arange(0.30,0.901,0.01):
        pred = np.where(p >= threshold, "FALL", "NON_FALL")
        rec = recall_score(y,pred,pos_label="FALL",zero_division=0)
        prec = precision_score(y,pred,pos_label="FALL",zero_division=0)
        f1 = f1_score(y,pred,pos_label="FALL",zero_division=0)
        # Prefer recall >= 0.90; among those maximize F1.
        eligible = rec >= 0.90
        score = (1 if eligible else 0, f1 if eligible else rec, prec)
        if best is None or score > best[0]:
            best = (score,float(threshold),float(prec),float(rec),float(f1))
    return best[1]

def evaluate_fall(model, X, y, threshold):
    classes = [str(c).upper() for c in model.classes_]
    p = model.predict_proba(X)[:,classes.index("FALL")]
    pred = np.where(p >= threshold, "FALL", "NON_FALL")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "macro_f1": float(f1_score(y,pred,average="macro")),
        "fall_precision": float(precision_score(y,pred,pos_label="FALL",zero_division=0)),
        "fall_recall": float(recall_score(y,pred,pos_label="FALL",zero_division=0)),
        "fall_f1": float(f1_score(y,pred,pos_label="FALL",zero_division=0)),
        "roc_auc": float(roc_auc_score((y=="FALL").astype(int),p)),
        "classification_report": classification_report(y,pred,labels=["FALL","NON_FALL"],output_dict=True,zero_division=0),
        "confusion_matrix": confusion_matrix(y,pred,labels=["FALL","NON_FALL"]).tolist(),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", help="Legacy: use the same window file for both tasks.")
    ap.add_argument("--activity-input", help="Window file for activity model.")
    ap.add_argument("--fall-input", help="Window file for fall model.")
    ap.add_argument("--model-dir",default="models")
    ap.add_argument("--report",default="models/bits2_training_report_v2.json")
    args=ap.parse_args()

    activity_input=args.activity_input or args.input
    fall_input=args.fall_input or args.input
    if not activity_input or not fall_input:
        raise SystemExit("Provide --input, or both --activity-input and --fall-input.")

    adf=pd.read_csv(activity_input)
    fdf=pd.read_csv(fall_input)
    required=set(FEATURE_COLUMNS)|{"subject_id","label","task","source_file"}
    for name,d in [("activity",adf),("fall",fdf)]:
        missing=required-set(d.columns)
        if missing:
            raise SystemExit(f"{name} input missing columns: {sorted(missing)}")

    # A single fixed subject split is derived from the union of both files,
    # so the activity and fall experiments use identical held-out people.
    all_subjects=set(adf.subject_id.astype(str))|set(fdf.subject_id.astype(str))
    train_s,val_s,test_s=split_subjects(all_subjects)
    def make_splits(d):
        return {
            "train":d[d.subject_id.astype(str).isin(train_s)],
            "validation":d[d.subject_id.astype(str).isin(val_s)],
            "test":d[d.subject_id.astype(str).isin(test_s)],
        }
    splits=make_splits(adf)
    fall_splits=make_splits(fdf)

    report={"activity_dataset":activity_input,"fall_dataset":fall_input,
            "feature_columns":FEATURE_COLUMNS,"seed":SEED,
            "window_samples":{"activity":int(adf.window_samples.iloc[0]),
                              "fall":int(fdf.window_samples.iloc[0])},
            "subjects":{"train":train_s,"validation":val_s,"test":test_s},
            "window_counts":{"activity":{k:int(len(v)) for k,v in splits.items()},
                             "fall":{k:int(len(v)) for k,v in fall_splits.items()}}}

    # Activity
    mask_tr=splits["train"].task.eq("activity")
    mask_v=splits["validation"].task.eq("activity")
    mask_t=splits["test"].task.eq("activity")
    ytr=splits["train"].loc[mask_tr,"label"]; yv=splits["validation"].loc[mask_v,"label"]; yt=splits["test"].loc[mask_t,"label"]
    Xa_tr=splits["train"].loc[mask_tr,FEATURE_COLUMNS]; Xa_v=splits["validation"].loc[mask_v,FEATURE_COLUMNS]; Xa_t=splits["test"].loc[mask_t,FEATURE_COLUMNS]
    candidates={}
    for name,m in models().items():
        m.fit(Xa_tr,ytr)
        candidates[name]=m
    val_scores={n:evaluate(m,Xa_v,yv,["RESTING","SITTING","WALKING","RUNNING"]) for n,m in candidates.items()}
    best_name=max(val_scores,key=lambda n:(val_scores[n]["macro_f1"],val_scores[n]["balanced_accuracy"]))
    best=candidates[best_name]
    # Refit on train+validation only after model selection.
    av=pd.concat([splits["train"],splits["validation"]])
    av=av[av.task.eq("activity")]
    best.fit(av[FEATURE_COLUMNS],av.label)
    activity_test=evaluate(best,Xa_t,yt,["RESTING","SITTING","WALKING","RUNNING"])
    report["activity"]={"selected_model":best_name,"validation":{k:{
        "accuracy":v["accuracy"],"balanced_accuracy":v["balanced_accuracy"],"macro_f1":v["macro_f1"]
    } for k,v in val_scores.items()},"test":activity_test}

    # Fall: ADL windows become NON_FALL. Use only fall recordings plus the
    # clean activity windows as negatives; excluded ambiguous ADLs are not
    # silently treated as non-fall.
    def fall_frame(s):
        out=s.copy()
        out["fall_label"]=np.where(out.recording_type.eq("fall"),"FALL",
                                   np.where(out.task.eq("activity"),"NON_FALL",""))
        return out[out.fall_label.ne("")]
    ftr,fv,ft=[fall_frame(s) for s in fall_splits.values()]
    ytr=ftr.fall_label; yv=fv.fall_label; yt=ft.fall_label
    candidates={}
    for name,m in models().items():
        m.fit(ftr[FEATURE_COLUMNS],ytr)
        candidates[name]=m
    fall_val={}
    for n,m in candidates.items():
        th=fall_threshold_search(m,fv[FEATURE_COLUMNS],yv)
        ev=evaluate_fall(m,fv[FEATURE_COLUMNS],yv,th)
        fall_val[n]=ev
    # Select primarily for recall at/above 90%, then F1.
    best_name=max(fall_val,key=lambda n:(
        1 if fall_val[n]["fall_recall"]>=0.90 else 0,
        fall_val[n]["fall_f1"], fall_val[n]["fall_precision"]
    ))
    best=candidates[best_name]
    av=pd.concat([ftr,fv])
    best.fit(av[FEATURE_COLUMNS],av.fall_label)
    threshold=fall_val[best_name]["threshold"]
    fall_test=evaluate_fall(best,ft[FEATURE_COLUMNS],yt,threshold)
    report["fall_detector"]={"selected_model":best_name,"validation":{
        k:{x:v[x] for x in ["threshold","accuracy","balanced_accuracy","macro_f1","fall_precision","fall_recall","fall_f1","roc_auc"]}
        for k,v in fall_val.items()
    },"test":fall_test}

    model_dir=Path(args.model_dir); model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":best,"feature_columns":FEATURE_COLUMNS,
                 "model_name":"SafeBand Activity Model v2",
                 "model_version":"bits2-v2-activity"},model_dir/"activity_model.joblib")
    joblib.dump({"model":candidates[best_name],"feature_columns":FEATURE_COLUMNS,
                 "model_name":"SafeBand Fall Detector v2",
                 "model_version":"bits2-v2-fall",
                 "fall_threshold":threshold},model_dir/"fall_detector.joblib")

    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({
        "activity_selected":best_name,
        "activity_test_macro_f1":activity_test["macro_f1"],
        "activity_test_balanced_accuracy":activity_test["balanced_accuracy"],
        "fall_selected":best_name,
        "fall_test_recall":fall_test["fall_recall"],
        "fall_test_precision":fall_test["fall_precision"],
        "fall_threshold":threshold,
        "report":args.report,
    },indent=2))

if __name__=="__main__":
    main()
