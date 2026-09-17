"""Evaluate a saved SafeBand audio-event prediction CSV."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    df=pd.read_csv(a.predictions)
    need={"label","prediction"}
    if not need.issubset(df): raise ValueError(f"Missing columns: {sorted(need-set(df))}")
    labels=sorted(set(df.label.astype(str))|set(df.prediction.astype(str)))
    pr,rc,f1,_=precision_recall_fscore_support(df.label,df.prediction,labels=labels,average="macro",zero_division=0)
    r={"n":len(df),"accuracy":float(accuracy_score(df.label,df.prediction)),
       "balanced_accuracy":float(balanced_accuracy_score(df.label,df.prediction)),
       "macro_precision":float(pr),"macro_recall":float(rc),"macro_f1":float(f1),
       "labels":labels,"confusion_matrix":confusion_matrix(df.label,df.prediction,labels=labels).tolist()}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(r,indent=2),encoding="utf-8"); print(json.dumps(r,indent=2))
if __name__=="__main__": main()
