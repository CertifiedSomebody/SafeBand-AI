"""Prepare AIRWISE BME680 minute data with robust dataset discovery."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.bme680_features import add_features, feature_columns
from tools.repo_utils import discover_airwise_dir, discover_airwise_files

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-dir",default="",help="Optional AIRWISE sensor_1min directory.")
    ap.add_argument("--out",default=str(ROOT/"datasets"/"processed"/"airwise_bme680"/"airwise_bme680_features.csv"))
    args=ap.parse_args()
    data_dir=Path(args.data_dir) if args.data_dir else discover_airwise_dir()
    paths=discover_airwise_files(data_dir)
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    required=["Date","location","temperature_c","pressure_hpa","humidity_rh","gas_resistance_ohms","IAQ_class"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df=df.dropna(subset=required).copy()
    df["IAQ_class"]=df["IAQ_class"].astype(str)
    df=add_features(df)
    feats=feature_columns(df)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    df[["Date","location","IAQ_class",*feats]].to_csv(out,index=False)
    print(f"Dataset directory: {data_dir.resolve()}")
    print(f"Input files: {len(paths)}")
    print(f"Prepared rows: {len(df)}")
    print(f"Features: {len(feats)}")
    print(df["IAQ_class"].value_counts().to_string())
    print(f"Saved: {out.resolve()}")
if __name__=="__main__": main()
