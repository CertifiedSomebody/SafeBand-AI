#!/usr/bin/env python3
"""Independent held-out evaluation for V5.2."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import joblib,numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def scores(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float);e=np.abs(y-p)
    return {"n":len(y),"mae_bpm":float(mean_absolute_error(y,p)),"rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)),"within_3_bpm_pct":float(np.mean(e<=3)*100),
            "within_5_bpm_pct":float(np.mean(e<=5)*100),"within_10_bpm_pct":float(np.mean(e<=10)*100),
            "bias_bpm":float(np.mean(p-y)),"median_abs_error_bpm":float(np.median(e)),
            "p90_abs_error_bpm":float(np.percentile(e,90)),"p95_abs_error_bpm":float(np.percentile(e,95)),
            "max_abs_error_bpm":float(np.max(e))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--predictions",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();p=pd.read_csv(a.predictions)
    if "prediction_bpm" not in p:raise ValueError("prediction_bpm missing")
    report={"version":"PPG WALKING V5.2 INDEPENDENT EVALUATION","overall":scores(p.hr_bpm,p.prediction_bpm)}
    report["by_subject"]={str(k):scores(g.hr_bpm,g.prediction_bpm) for k,g in p.groupby("subject")}
    report["by_record"]={str(k):scores(g.hr_bpm,g.prediction_bpm) for k,g in p.groupby("record")}
    report["worst_record"]=max(report["by_record"],key=lambda k:report["by_record"][k]["mae_bpm"])
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
