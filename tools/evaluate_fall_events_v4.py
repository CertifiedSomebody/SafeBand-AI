"""
SafeBand AI — Event-level fall evaluation v4.

Turns per-window fall probabilities into event predictions. This prevents
one physical fall from being counted as many independent successful windows
and supports a precision/recall/false-alarm tradeoff at the recording level.

A recording is considered a FALL event if the recording contains at least
one confirmed temporal cluster of high-probability windows.

This is an evaluation/experiment layer only; it does not send alerts.
"""

from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

def load_predictions(model_path, window_csv):
    payload=joblib.load(model_path)
    df=pd.read_csv(window_csv)
    cols=payload["feature_columns"]
    model=payload["model"]
    p=model.predict_proba(df[cols])
    classes=[str(c).upper() for c in model.classes_]
    idx=classes.index("FALL")
    df["fall_probability"]=p[:,idx]
    return df

def aggregate_recording(g, threshold, min_hits, max_gap):
    g=g.sort_values("start_sample")
    hits=g["fall_probability"].to_numpy() >= threshold
    starts=g["start_sample"].to_numpy()
    clusters=[]
    current=[]
    for i,hit in enumerate(hits):
        if not hit:
            continue
        if not current or starts[i]-starts[current[-1]] <= max_gap:
            current.append(i)
        else:
            clusters.append(current); current=[i]
    if current: clusters.append(current)
    confirmed=[c for c in clusters if len(c)>=min_hits]
    return bool(confirmed), (max((g.iloc[c]["fall_probability"].max() for c in confirmed),default=0.0)), len(confirmed)

def evaluate(df, threshold, min_hits, max_gap):
    rows=[]
    for key,g in df.groupby(["subject_id","source_file"],sort=False):
        is_fall=str(g["recording_type"].iloc[0]).lower()=="fall"
        pred,peak,count=aggregate_recording(g,threshold,min_hits,max_gap)
        rows.append({"subject_id":key[0],"source_file":key[1],
                     "truth":"FALL" if is_fall else "NON_FALL",
                     "prediction":"FALL" if pred else "NON_FALL",
                     "peak_probability":peak,"confirmed_clusters":count})
    r=pd.DataFrame(rows)
    y=r.truth; p=r.prediction
    tn,fp,fn,tp=confusion_matrix(y,p,labels=["NON_FALL","FALL"]).ravel()
    return {
        "recordings":int(len(r)),
        "fall_events":int((y=="FALL").sum()),
        "nonfall_recordings":int((y=="NON_FALL").sum()),
        "fall_recall":float(recall_score(y,p,pos_label="FALL",zero_division=0)),
        "fall_precision":float(precision_score(y,p,pos_label="FALL",zero_division=0)),
        "fall_f1":float(f1_score(y,p,pos_label="FALL",zero_division=0)),
        "false_positive_rate":float(fp/max(1,fp+tn)),
        "false_alarms":int(fp),
        "missed_falls":int(fn),
        "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],
    },r

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--input",required=True)
    ap.add_argument("--out",default="models/fall_event_report_v4.json")
    ap.add_argument("--thresholds",default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    ap.add_argument("--min-hits",default="1,2,3",help="Comma-separated consecutive/high-probability hit counts")
    ap.add_argument("--max-gap",type=int,default=30,help="Maximum sample gap between hit windows")
    args=ap.parse_args()

    df=load_predictions(args.model,args.input)
    results=[]
    for th in map(float,args.thresholds.split(",")):
        for mh in map(int,args.min_hits.split(",")):
            metrics,_=evaluate(df,th,mh,args.max_gap)
            results.append({"threshold":th,"min_hits":mh,**metrics})
    result_df=pd.DataFrame(results)
    # Select operating point conservatively: prefer >=90% recall, then lowest
    # false-positive rate, then highest F1. If impossible, report the best F1.
    eligible=result_df[result_df.fall_recall>=.90]
    pool=eligible if len(eligible) else result_df
    best=pool.sort_values(["false_positive_rate","fall_f1"],ascending=[True,False]).iloc[0].to_dict()
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "window_input":args.input,"model":args.model,
        "max_gap_samples":args.max_gap,
        "operating_points":results,"selected_operating_point":best
    },indent=2),encoding="utf-8")
    print(json.dumps({"selected_operating_point":best,"report":args.out},indent=2))

if __name__=="__main__":
    main()
