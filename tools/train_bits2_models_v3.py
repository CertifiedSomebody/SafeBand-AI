"""
SafeBand AI — BITS-2 Training v3.

Activity:
  compares RF/ExtraTrees/Logistic and reports per-class performance.

Fall:
  compares models and evaluates thresholds over a recall/false-positive
  operating range. No threshold is declared safe automatically.
"""

from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,classification_report,confusion_matrix,precision_score,recall_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ai.bits2_features import FEATURE_COLUMNS
from ai.stationary_features import STATIONARY_FEATURES
from ai.fall_event_features import FALL_EVENT_FEATURES

SEED=42
def split(sub):
    s=sorted(set(str(x) for x in sub),key=lambda x:int(x))
    rng=np.random.default_rng(SEED); rng.shuffle(s)
    nt=max(1,round(len(s)*.2)); nv=max(1,round(len(s)*.2))
    return sorted(s[nt+nv:],key=int),sorted(s[nt:nt+nv],key=int),sorted(s[:nt],key=int)

def models():
    return {
      "extra_trees":ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=SEED,n_jobs=-1),
      "random_forest":RandomForestClassifier(n_estimators=250,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=SEED,n_jobs=-1),
      "logistic_regression":Pipeline([("scale",StandardScaler()),("clf",LogisticRegression(max_iter=2500,class_weight="balanced",random_state=SEED))])
    }

def ev(m,X,y,labels):
    p=m.predict(X)
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro")),"report":classification_report(y,p,labels=labels,output_dict=True,zero_division=0),
            "confusion_matrix":confusion_matrix(y,p,labels=labels).tolist()}

def fall_operating_points(m,X,y):
    classes=[str(c).upper() for c in m.classes_]; idx=classes.index("FALL")
    p=m.predict_proba(X)[:,idx]; out=[]
    for th in np.arange(.25,.86,.01):
        pred=np.where(p>=th,"FALL","NON_FALL")
        out.append({"threshold":round(float(th),2),
                    "fall_recall":float(recall_score(y,pred,pos_label="FALL")),
                    "fall_precision":float(precision_score(y,pred,pos_label="FALL")),
                    "nonfall_recall":float(recall_score(y,pred,pos_label="NON_FALL")),
                    "macro_f1":float(f1_score(y,pred,average="macro")),
                    "false_positive_rate":float(((y=="NON_FALL")&(pred=="FALL")).sum()/max(1,(y=="NON_FALL").sum()))})
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--activity-input",required=True); ap.add_argument("--fall-input",required=True)
    ap.add_argument("--model-dir",default="models"); ap.add_argument("--report",default="models/bits2_training_report_v3.json")
    a=ap.parse_args()
    adf=pd.read_csv(a.activity_input); fdf=pd.read_csv(a.fall_input)
    subjects=set(adf.subject_id.astype(str))|set(fdf.subject_id.astype(str))
    tr,va,te=split(subjects)
    def S(d): return {k:d[d.subject_id.astype(str).isin(v)].copy() for k,v in [("train",tr),("validation",va),("test",te)]}
    A=S(adf); F=S(fdf)
    AF=FEATURE_COLUMNS+STATIONARY_FEATURES
    FF=FEATURE_COLUMNS+FALL_EVENT_FEATURES
    report={"version":"v3","seed":SEED,"activity_dataset":a.activity_input,"fall_dataset":a.fall_input,
            "subjects":{"train":tr,"validation":va,"test":te},"feature_counts":{"activity":len(AF),"fall":len(FF)}}

    # Activity model selection.
    c=models()
    scores={}
    for n,m in c.items():
        m.fit(A["train"][AF],A["train"].label)
        scores[n]=ev(m,A["validation"][AF],A["validation"].label,["RESTING","SITTING","WALKING","RUNNING"])
    best=max(scores,key=lambda n:(scores[n]["macro_f1"],scores[n]["balanced_accuracy"]))
    m=c[best]; fit=pd.concat([A["train"],A["validation"]]); m.fit(fit[AF],fit.label)
    report["activity"]={"selected_model":best,"validation":scores[best],
                        "test":ev(m,A["test"][AF],A["test"].label,["RESTING","SITTING","WALKING","RUNNING"])}

    # Fall model. Require both fall and clean mapped ADL controls.
    c=models(); fs={}
    for n,m in c.items():
        m.fit(F["train"][FF],F["train"].label)
        fs[n]=fall_operating_points(m,F["validation"][FF],F["validation"].label)
    # Model selection is based on best validation macro-F1, not raw recall.
    best=max(c,key=lambda n:max(x["macro_f1"] for x in fs[n]))
    # choose validation threshold with explicit preference for recall >= .90,
    # otherwise maximize macro-F1. The full curve is retained for review.
    eligible=[x for x in fs[best] if x["fall_recall"]>=.90]
    point=max(eligible or fs[best],key=lambda x:(x["macro_f1"],x["fall_precision"]))
    m=c[best]; fit=pd.concat([F["train"],F["validation"]]); m.fit(fit[FF],fit.label)
    classes=[str(x).upper() for x in m.classes_]; p=m.predict_proba(F["test"][FF])[:,classes.index("FALL")]
    pred=np.where(p>=point["threshold"],"FALL","NON_FALL")
    report["fall_detector"]={"selected_model":best,"validation_operating_point":point,
      "validation_curve":fs[best],"test":{"threshold":point["threshold"],
      "accuracy":float(accuracy_score(F["test"].label,pred)),
      "balanced_accuracy":float(balanced_accuracy_score(F["test"].label,pred)),
      "macro_f1":float(f1_score(F["test"].label,pred,average="macro")),
      "fall_precision":float(precision_score(F["test"].label,pred,pos_label="FALL")),
      "fall_recall":float(recall_score(F["test"].label,pred,pos_label="FALL")),
      "nonfall_recall":float(recall_score(F["test"].label,pred,pos_label="NON_FALL")),
      "false_positive_rate":float(((F["test"].label=="NON_FALL")&(pred=="FALL")).sum()/max(1,(F["test"].label=="NON_FALL").sum())),
      "roc_auc":float(roc_auc_score((F["test"].label=="FALL").astype(int),p)),
      "confusion_matrix":confusion_matrix(F["test"].label,pred,labels=["FALL","NON_FALL"]).tolist()}}

    md=Path(a.model_dir); md.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":m,"feature_columns":AF,"model_version":"bits2-v3-activity"},md/"activity_model_v3.joblib")
    joblib.dump({"model":m,"feature_columns":FF,"model_version":"bits2-v3-fall","fall_threshold":point["threshold"]},md/"fall_detector_v3.joblib")
    Path(a.report).parent.mkdir(parents=True,exist_ok=True); Path(a.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"activity_test":report["activity"]["test"],"fall_test":report["fall_detector"]["test"],"report":a.report},indent=2))

if __name__=="__main__": main()
