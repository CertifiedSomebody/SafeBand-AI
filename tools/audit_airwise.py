"""Audit AIRWISE v1.0.0 without assuming archive extraction depth."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.repo_utils import discover_airwise_dir, discover_airwise_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="", help="Optional AIRWISE sensor_1min directory.")
    ap.add_argument("--out", default=str(ROOT/"datasets"/"processed"/"airwise_bme680"/"airwise_audit.json"))
    args = ap.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else discover_airwise_dir()
    paths = discover_airwise_files(data_dir)
    frames=[]; report={"dataset_dir":str(data_dir.resolve()),"files":[],"total_rows":0}
    for p in paths:
        df=pd.read_csv(p); frames.append(df)
        item={"file":p.name,"rows":len(df),"columns":list(df.columns),
              "missing":{k:int(v) for k,v in df.isna().sum().items() if v},
              "locations":sorted(df["location"].dropna().astype(str).unique().tolist()) if "location" in df else [],
              "class_counts":df["IAQ_class"].value_counts(dropna=False).to_dict() if "IAQ_class" in df else {}}
        if "Date" in df:
            dt=pd.to_datetime(df["Date"],errors="coerce")
            item.update({"date_min":str(dt.min()),"date_max":str(dt.max()),"date_invalid":int(dt.isna().sum())})
        report["files"].append(item); report["total_rows"] += len(df)
    all_df=pd.concat(frames,ignore_index=True)
    required=["temperature_c","pressure_hpa","humidity_rh","gas_resistance_ohms","IAQ_proxy"]
    report["combined_class_counts"]={str(k):int(v) for k,v in all_df["IAQ_class"].value_counts(dropna=False).items()}
    report["numeric_summary"]=all_df[required].describe().to_dict()
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print(json.dumps(report,indent=2,default=str))
if __name__=="__main__": main()
