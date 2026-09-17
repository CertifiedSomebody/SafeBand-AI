"""Train/freeze SafeBand BME680 AIRWISE V1 with a chronological hold-out."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from ai.bme680_features import feature_columns

DEFAULT=ROOT/"datasets"/"processed"/"airwise_bme680"/"airwise_bme680_features.csv"
OUT=ROOT/"models"/"bme680_airwise_v1"

def metrics(y,p,labels=None):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=labels,average="macro",zero_division=0)
    return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_precision":float(pr),"macro_recall":float(rc),"macro_f1":float(f1),
            "confusion_matrix":confusion_matrix(y,p,labels=labels).tolist()}

def temporal_split(df):
    parts=[]
    for loc,g in df.groupby("location",sort=True):
        g=g.sort_values("Date").reset_index(drop=True); n=len(g); a=int(n*.60); b=int(n*.80)
        parts += [g.iloc[:a].assign(_split="train"),g.iloc[a:b].assign(_split="val"),g.iloc[b:].assign(_split="test")]
    return pd.concat(parts,ignore_index=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=str(DEFAULT)); ap.add_argument("--out",default=str(OUT)); ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    data=Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Prepared AIRWISE file not found: {data}. Run python tools\\prepare_airwise_bme680.py first.")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data); df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df=df.dropna(subset=["Date","location","IAQ_class"]).copy()
    feats=feature_columns(df); labels=sorted(df.IAQ_class.astype(str).unique()); sp=temporal_split(df)
    tr,va,te=[sp[sp._split==s].copy() for s in ("train","val","test")]
    models={
      "logistic":Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),("m",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=args.seed))]),
      "rf":Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=500,random_state=args.seed,n_jobs=-1,class_weight="balanced_subsample",min_samples_leaf=2))]),
      "extra_trees":Pipeline([("imp",SimpleImputer(strategy="median")),("m",ExtraTreesClassifier(n_estimators=500,random_state=args.seed,n_jobs=-1,class_weight="balanced",min_samples_leaf=2))]),
      "hgb":Pipeline([("imp",SimpleImputer(strategy="median")),("m",HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_leaf_nodes=31,l2_regularization=1.0,random_state=args.seed))])
    }
    results={}
    for name,m in models.items():
        if name=="hgb":
            m.fit(tr[feats],tr.IAQ_class,m__sample_weight=compute_sample_weight("balanced",tr.IAQ_class))
        else: m.fit(tr[feats],tr.IAQ_class)
        results[name]=metrics(va.IAQ_class,m.predict(va[feats]),labels=labels)
    selected=max(results,key=lambda k:results[k]["balanced_accuracy"])
    dev=pd.concat([tr,va],ignore_index=True); model=models[selected]
    if selected=="hgb":
        model.fit(dev[feats],dev.IAQ_class,m__sample_weight=compute_sample_weight("balanced",dev.IAQ_class))
    else: model.fit(dev[feats],dev.IAQ_class)
    pte=model.predict(te[feats]); final=metrics(te.IAQ_class,pte,labels=labels)
    joblib.dump({"model":model,"feature_columns":feats,"model_name":"SafeBand BME680 AIRWISE V1",
                 "model_version":"bme680_airwise_v1","label_classes":labels,
                 "dataset_domain":"AIRWISE BME680 indoor minute data"},out/"bme680_airwise_v1.joblib")
    pd.DataFrame({"Date":te.Date,"location":te.location,"y_true":te.IAQ_class,"y_pred":pte}).to_csv(out/"test_predictions.csv",index=False)
    report={"version":"SafeBand BME680 AIRWISE V1","features":feats,"feature_count":len(feats),
            "split":"60/20/20 chronological within each location","rows":{"train":len(tr),"val":len(va),"test":len(te)},
            "validation":results,"selected_model":selected,"held_out_test":final}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
