#!/usr/bin/env python3
"""Read V4.3 final predictions and produce interpretable diagnostics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np,pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def scores(g):
    y=g.hr_bpm.to_numpy(float); p=g.prediction_bpm.to_numpy(float); e=np.abs(y-p)
    return {"n":len(g),"mae_bpm":float(mean_absolute_error(y,p)),
            "rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p)),
            "within_3_bpm_pct":float(np.mean(e<=3)*100),
            "within_5_bpm_pct":float(np.mean(e<=5)*100),
            "within_10_bpm_pct":float(np.mean(e<=10)*100),
            "bias_bpm":float(np.mean(p-y)),
            "median_abs_error_bpm":float(np.median(e)),
            "p90_abs_error_bpm":float(np.percentile(e,90)),
            "p95_abs_error_bpm":float(np.percentile(e,95)),
            "max_abs_error_bpm":float(np.max(e))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--report-out",required=True)
    a=ap.parse_args(); df=pd.read_csv(a.predictions)
    need={"subject","activity","record","hr_bpm","prediction_bpm"}
    if not need.issubset(df): raise ValueError(f"Missing columns: {sorted(need-set(df))}")
    if not np.isfinite(df[["hr_bpm","prediction_bpm"]].to_numpy(float)).all(): raise ValueError("NaN/Inf detected")
    report={"overall":scores(df),
            "by_activity":{str(k):scores(g) for k,g in df.groupby("activity",sort=True)},
            "by_subject":{str(k):scores(g) for k,g in df.groupby("subject",sort=True)},
            "by_record":{str(k):scores(g) for k,g in df.groupby("record",sort=True)}}
    report["worst_activity"]=max(report["by_activity"],key=lambda k:report["by_activity"][k]["mae_bpm"])
    report["worst_subject"]=max(report["by_subject"],key=lambda k:report["by_subject"][k]["mae_bpm"])
    report["worst_record"]=max(report["by_record"],key=lambda k:report["by_record"][k]["mae_bpm"])
    out=Path(a.report_out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    o=report["overall"]
    print("=== V4.3 FINAL DIAGNOSTICS ===")
    print(f"MAE={o['mae_bpm']:.3f} BPM | RMSE={o['rmse_bpm']:.3f} | R2={o['r2']:.3f}")
    print(f"±3={o['within_3_bpm_pct']:.2f}% | ±5={o['within_5_bpm_pct']:.2f}% | ±10={o['within_10_bpm_pct']:.2f}%")
    print(f"Bias={o['bias_bpm']:+.3f} | Median={o['median_abs_error_bpm']:.3f} | P90={o['p90_abs_error_bpm']:.3f} | P95={o['p95_abs_error_bpm']:.3f}")
    print("Worst activity:",report["worst_activity"]," Worst subject:",report["worst_subject"]," Worst record:",report["worst_record"])
    print("Saved:",out)

if __name__=="__main__": main()
