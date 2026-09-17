"""Subject-grouped 5-fold CV for the FORTH-TRACE common ACC contract.

This is a stability study. No test set is used for model selection.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.common_acc_features import FEATURES

MODELS={
 "extra_trees":ExtraTreesClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=42,n_jobs=-1),
 "random_forest":RandomForestClassifier(n_estimators=500,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=42,n_jobs=-1),
 "rbf_svm":Pipeline([("scale",StandardScaler()),("svc",SVC(C=10,gamma="scale",class_weight="balanced"))]),
 "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=300,learning_rate=.05,max_leaf_nodes=31,l2_regularization=.1,random_state=42)
}

def metrics(y,p):
    return {"accuracy":accuracy_score(y,p),"balanced_accuracy":balanced_accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro")}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default="datasets/processed/activity_common_acc/forth_trace_common_acc.csv")
    ap.add_argument("--out-dir",default="models/forth_trace_activity_cv")
    args=ap.parse_args()
    input_path=ROOT/args.input
    audit_path=input_path.with_suffix(".audit.json")
    if not audit_path.exists():
        raise SystemExit(f"Missing audit file: {audit_path}. Rebuild the common-ACC artifact first.")
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("contract") != "SafeBand common ACC V1.2":
        raise SystemExit(f"Stale/older common-ACC artifact detected ({audit.get('contract')!r}). Rebuild with V1.2 first.")
    df=pd.read_csv(input_path)
    labels=sorted(df.activity_label.unique())
    if len(df) == 0 or len(labels) < 2:
        raise SystemExit("Common-ACC artifact is empty or has fewer than two classes. Rebuild it first.")
    if df.subject_id.nunique() < 5:
        raise SystemExit(f"Need at least 5 subjects for 5-fold grouped CV; found {df.subject_id.nunique()}")
    X=df[FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy()
    y=df.activity_label.to_numpy()
    g=df.subject_id.to_numpy()
    splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
    fold_rows=[]; details={}
    for name,model0 in MODELS.items():
        vals=[]
        for fold,(tr,va) in enumerate(splitter.split(X,y,g),1):
            model=clone(model0)
            model.fit(X[tr],y[tr])
            p=model.predict(X[va])
            m=metrics(y[va],p); m["fold"]=fold; m["model"]=name
            vals.append(m)
        fold_rows.extend(vals)
        details[name]=vals
        print(f"{name:22s} acc={np.mean([v['accuracy'] for v in vals]):.4f}±{np.std([v['accuracy'] for v in vals]):.4f} "
              f"bal={np.mean([v['balanced_accuracy'] for v in vals]):.4f}±{np.std([v['balanced_accuracy'] for v in vals]):.4f} "
              f"macroF1={np.mean([v['macro_f1'] for v in vals]):.4f}±{np.std([v['macro_f1'] for v in vals]):.4f}")
    summary=[]
    for name,vals in details.items():
        summary.append({"model":name,
                        "accuracy_mean":float(np.mean([v["accuracy"] for v in vals])),
                        "accuracy_std":float(np.std([v["accuracy"] for v in vals])),
                        "balanced_accuracy_mean":float(np.mean([v["balanced_accuracy"] for v in vals])),
                        "balanced_accuracy_std":float(np.std([v["balanced_accuracy"] for v in vals])),
                        "macro_f1_mean":float(np.mean([v["macro_f1"] for v in vals])),
                        "macro_f1_std":float(np.std([v["macro_f1"] for v in vals]))})
    # Primary selection criterion: macro-F1, then balanced accuracy, then accuracy.
    selected=sorted(summary,key=lambda r:(r["macro_f1_mean"],r["balanced_accuracy_mean"],r["accuracy_mean"]),reverse=True)[0]
    out=ROOT/args.out_dir; out.mkdir(parents=True,exist_ok=True)
    report={"dataset":args.input,"contract":"common ACC V1.1","subjects":sorted(set(map(str,g))),
            "classes":labels,"folds":5,"seed":42,"selection_metric":"macro_f1",
            "selected_model":selected,"model_summary":summary,"fold_metrics":fold_rows}
    (out/"cv_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame(summary).to_csv(out/"cv_model_summary.csv",index=False)
    print("\nSELECTED:",selected["model"])
    print("Saved:",out/"cv_report.json")

if __name__=="__main__": main()
