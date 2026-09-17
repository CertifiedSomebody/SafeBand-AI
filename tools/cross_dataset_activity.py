"""Leakage-safe cross-dataset activity transfer using the common ACC contract.

Shared labels only: SITTING and WALKING.
Source model selection is performed by grouped 5-fold CV on SOURCE ONLY.
The TARGET dataset is never used to select the model.
"""
from __future__ import annotations
from pathlib import Path
import argparse, sys, json
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.common_acc_features import FEATURES

SHARED=["SITTING","WALKING"]
MODELS={
 "extra_trees":ExtraTreesClassifier(n_estimators=700,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1),
 "random_forest":RandomForestClassifier(n_estimators=700,max_features="sqrt",class_weight="balanced",random_state=42,n_jobs=-1),
 "rbf_svm":Pipeline([("scale",StandardScaler()),("svc",SVC(C=10,gamma="scale",class_weight="balanced"))])
}

def score(y,p):
    return {"accuracy":float(accuracy_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro")),
            "confusion_matrix":confusion_matrix(y,p,labels=SHARED).tolist(),
            "classification_report":classification_report(y,p,labels=SHARED,target_names=SHARED,output_dict=True,zero_division=0)}

def select_source(df):
    X=df[FEATURES].to_numpy(); y=df.activity_label.to_numpy(); g=df.subject_id.to_numpy()
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]
    for name,proto in MODELS.items():
        vals=[]
        for tr,va in cv.split(X,y,g):
            m=clone(proto)
            m.fit(X[tr],y[tr]); p=m.predict(X[va])
            vals.append([accuracy_score(y[va],p),balanced_accuracy_score(y[va],p),f1_score(y[va],p,average="macro")])
        a=np.array(vals)
        rows.append({"model":name,"accuracy_mean":float(a[:,0].mean()),
                     "balanced_accuracy_mean":float(a[:,1].mean()),
                     "macro_f1_mean":float(a[:,2].mean())})
    return sorted(rows,key=lambda r:(r["macro_f1_mean"],r["balanced_accuracy_mean"],r["accuracy_mean"]),reverse=True)


def validate_domain(d, path):
    classes = sorted(d.activity_label.astype(str).unique().tolist())
    if classes != SHARED:
        raise ValueError(f"{path}: expected exactly shared classes {SHARED}; got {classes}")
    if d.subject_id.nunique() < 5:
        raise ValueError(f"{path}: need at least 5 subjects for grouped source selection; found {d.subject_id.nunique()}")


def load_common(path):
    path_obj=ROOT/path
    audit_path=path_obj.with_suffix(".audit.json")
    if not audit_path.exists(): raise ValueError(f"FORTH-TRACE audit missing: {audit_path}. Rebuild common ACC artifact first.")
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("contract") != "SafeBand common ACC V1.2": raise ValueError(f"Stale/older FORTH-TRACE common artifact: {audit.get('contract')!r}. Rebuild first.")
    d=pd.read_csv(path_obj)
    d=d[d.activity_label.isin(SHARED)].copy()
    missing=[c for c in FEATURES if c not in d.columns]
    if missing: raise ValueError(f"{path}: missing common features: {missing}")
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["activity_label","subject_id"])
    validate_domain(d, path)
    return d

def load_bits(path):
    d=pd.read_csv(ROOT/path)
    # BITS-2 labels are activity_label in the actual project artifact.
    d=d[d.activity_label.isin(SHARED)].copy()
    missing=[c for c in FEATURES if c not in d.columns]
    if missing: raise ValueError(f"BITS-2 artifact missing common features: {missing}")
    d=d.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["activity_label","subject_id"])
    d["source"]="BITS2"
    validate_domain(d, path)
    return d

def run_transfer(src,tgt,src_name,tgt_name,out):
    ranking=select_source(src)
    selected=ranking[0]["model"]
    model=MODELS[selected]
    Xs=src[FEATURES].to_numpy(); ys=src.activity_label.to_numpy()
    Xt=tgt[FEATURES].to_numpy(); yt=tgt.activity_label.to_numpy()
    model=clone(MODELS[selected])
    model.fit(Xs,ys); pt=model.predict(Xt)
    target_metrics=score(yt,pt)
    target_counts=pd.Series(yt).value_counts()
    majority_class=str(target_counts.index[0])
    majority_accuracy=float(target_counts.iloc[0]/len(yt))
    predicted_counts=pd.Series(pt).value_counts()
    degenerate_prediction=(len(predicted_counts) == 1)
    target_metrics.update({
        "majority_class": majority_class,
        "majority_baseline_accuracy": majority_accuracy,
        "accuracy_delta_vs_majority_baseline": float(target_metrics["accuracy"] - majority_accuracy),
        "chance_balanced_accuracy": 0.5,
        "degenerate_single_class_prediction": degenerate_prediction,
        "predicted_class_counts": {str(k): int(v) for k,v in predicted_counts.items()},
    })
    result={"direction":f"{src_name}_TO_{tgt_name}","source_rows":len(src),
            "target_rows":len(tgt),"source_subjects":sorted(src.subject_id.astype(str).unique().tolist()),
            "target_subjects":sorted(tgt.subject_id.astype(str).unique().tolist()),
            "classes":SHARED,"source_model_selection":ranking,
            "selected_model":selected,"target_metrics":target_metrics,
            "interpretation_note":"Cross-domain metrics are evaluated on untouched target data. Accuracy must be read with balanced accuracy and macro-F1; majority-class and degenerate-prediction diagnostics are included to expose domain-shift collapse."}
    (out/f"{src_name.lower()}_to_{tgt_name.lower()}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"\n{src_name} -> {tgt_name}: {selected}")
    print(result["target_metrics"])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--forth",default="datasets/processed/activity_common_acc/forth_trace_common_acc.csv")
    ap.add_argument("--bits2",default="datasets/processed/bits2/activity_windows.csv")
    ap.add_argument("--out-dir",default="models/activity_cross_dataset")
    args=ap.parse_args()
    forth=load_common(args.forth); bits=load_bits(args.bits2)
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    run_transfer(forth,bits,"FORTH_TRACE","BITS2",out)
    run_transfer(bits,forth,"BITS2","FORTH_TRACE",out)
    manifest={"shared_classes":SHARED,"forbidden_mappings":["RESTING->STANDING","RUNNING->STAIRS"],
              "feature_contract":"ACC-only, 37 features, common 20 Hz / 40-sample windows; FORTH source resampled from 51.2 Hz using ~2 s raw windows",
              "source_selection":"5-fold StratifiedGroupKFold on source only","seed":42}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\nSaved cross-dataset reports to",out)

if __name__=="__main__": main()
