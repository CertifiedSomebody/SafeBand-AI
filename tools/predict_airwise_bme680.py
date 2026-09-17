"""Run the frozen AIRWISE BME680 V1 model on new minute-level CSV data."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import joblib,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from ai.bme680_features import add_features
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default=str(ROOT/"models"/"bme680_airwise_v1"/"bme680_airwise_v1.joblib"))
    ap.add_argument("--csv",required=True); ap.add_argument("--out",default="")
    args=ap.parse_args()
    df=pd.read_csv(args.csv); df=add_features(df)
    payload=joblib.load(args.model); model=payload["model"] if isinstance(payload,dict) and "model" in payload else payload
    feats=payload.get("feature_columns",[]) if isinstance(payload,dict) else []
    if not feats: feats=[c for c in df.columns if c not in {"Date","location","IAQ_class","IAQ_proxy","temperature_c_zscore","pressure_hpa_zscore","humidity_rh_zscore","gas_resistance_ohms_zscore","IAQ_proxy_zscore","Z-score","samples_per_hour"} and pd.api.types.is_numeric_dtype(df[c])]
    missing=[c for c in feats if c not in df.columns]
    if missing: raise ValueError(f"Missing model features: {missing}")
    df["IAQ_prediction"]=model.predict(df[feats])
    out=Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem+"_predicted.csv")
    out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False); print(f"Saved: {out.resolve()}")
if __name__=="__main__": main()
