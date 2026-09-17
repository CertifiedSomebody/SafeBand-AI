#!/usr/bin/env python3
"""V4.2 independent diagnostics: activity, subject, record and error analysis."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def m(g):
    y=g.hr_bpm.to_numpy(float); p=g.prediction_bpm.to_numpy(float); e=np.abs(y-p)
    return {"n":len(g),"mae_bpm":float(mean_absolute_error(y,p)),
            "rmse_bpm":float(np.sqrt(mean_squared_error(y,p))),
            "r2":float(r2_score(y,p)) if len(g)>1 else None,
            "within_3_bpm_pct":float(np.mean(e<=3)*100),
            "within_5_bpm_pct":float(np.mean(e<=5)*100),
            "within_10_bpm_pct":float(np.mean(e<=10)*100),
            "bias_bpm":float(np.mean(p-y)),
            "median_abs_error_bpm":float(np.median(e)),
            "p90_abs_error_bpm":float(np.percentile(e,90)),
            "max_abs_error_bpm":float(np.max(e))}

def groups(df,col):
    return {str(k):m(g) for k,g in df.groupby(col,sort=True)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--predictions",required=True); ap.add_argument("--report-out",required=True)
    a=ap.parse_args(); df=pd.read_csv(a.predictions)
    need={"subject","activity","record","hr_bpm","prediction_bpm"}
    if not need.issubset(df.columns): raise ValueError(f"Missing: {sorted(need-set(df.columns))}")
    df.hr_bpm=pd.to_numeric(df.hr_bpm,errors="coerce"); df.prediction_bpm=pd.to_numeric(df.prediction_bpm,errors="coerce")
    if not np.isfinite(df[["hr_bpm","prediction_bpm"]].to_numpy(float)).all(): raise ValueError("NaN/Inf")
    report={"overall":m(df),"by_activity":groups(df,"activity"),"by_subject":groups(df,"subject"),
            "by_record":groups(df,"record")}
    e=np.abs(df.prediction_bpm-df.hr_bpm)
    report["error_quantiles_bpm"]={k:float(np.percentile(e,q)) for k,q in
        [("p50",50),("p75",75),("p90",90),("p95",95),("max",100)]}
    # Flags are diagnostic, not automatic judgments.
    report["diagnostic_flags"]={
        "largest_activity_mae":max(report["by_activity"],key=lambda k:report["by_activity"][k]["mae_bpm"]),
        "largest_subject_mae":max(report["by_subject"],key=lambda k:report["by_subject"][k]["mae_bpm"]),
        "largest_record_mae":max(report["by_record"],key=lambda k:report["by_record"][k]["mae_bpm"]),
    }
    out=Path(a.report_out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("=== V4.2 DIAGNOSTICS ===")
    o=report["overall"]; print(f"N={o['n']} MAE={o['mae_bpm']:.3f} RMSE={o['rmse_bpm']:.3f} R2={o['r2']:.3f}")
    print(f"±3={o['within_3_bpm_pct']:.2f}% ±5={o['within_5_bpm_pct']:.2f}% ±10={o['within_10_bpm_pct']:.2f}% Bias={o['bias_bpm']:+.3f}")
    print("\nACTIVITY")
    for k,v in report["by_activity"].items(): print(f"{k:>5}: MAE={v['mae_bpm']:.3f} ±5={v['within_5_bpm_pct']:.2f}% Bias={v['bias_bpm']:+.3f}")
    print("\nSUBJECT")
    for k,v in report["by_subject"].items(): print(f"{k:>3}: MAE={v['mae_bpm']:.3f} ±5={v['within_5_bpm_pct']:.2f}% Bias={v['bias_bpm']:+.3f}")
    print("\nLargest activity/subject/record errors:")
    print(report["diagnostic_flags"])
    print(f"\nSaved: {out}")
if __name__=="__main__": main()
