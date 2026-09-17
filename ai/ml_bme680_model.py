"""Runtime adapter for the optional AIRWISE BME680 IAQ classifier.

The model is a benchmark adapter. It must not be interpreted as a clinical
or personal-health classifier. Runtime prediction requires the same BME680
feature contract used during training.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any,Dict
import pandas as pd
import joblib
from ai.bme680_features import add_features

class BME680IAQModel:
    def __init__(self,model_path:Path,max_history:int=5):
        self.model_path=Path(model_path); self.max_history=max(5,int(max_history))
        self.model=None; self.feature_columns=[]; self.model_name="SafeBand BME680 AIRWISE V1"
        self.model_version="unknown"; self.load_error=None; self.history=deque(maxlen=self.max_history)
        self._load()
    def _load(self):
        if not self.model_path.exists():
            self.load_error="BME680 IAQ model file not found."; return
        try:
            payload=joblib.load(self.model_path)
            if isinstance(payload,dict) and "model" in payload:
                self.model=payload["model"]; self.feature_columns=list(payload.get("feature_columns",[]))
                self.model_name=str(payload.get("model_name",self.model_name)); self.model_version=str(payload.get("model_version",self.model_version))
            else: self.model=payload
        except Exception as exc: self.load_error=str(exc)
    @property
    def available(self): return self.model is not None
    def reset(self): self.history.clear()
    def predict(self,sample:Dict[str,Any])->Dict[str,Any]:
        if not self.available: return {"iaq_class":"UNKNOWN","confidence":0.0,"source":"UNAVAILABLE","error":self.load_error}
        row={"Date":sample.get("timestamp",sample.get("Date")),"location":sample.get("location","wearable"),
             "temperature_c":sample.get("temperature"),"pressure_hpa":sample.get("pressure"),
             "humidity_rh":sample.get("humidity"),"gas_resistance_ohms":sample.get("gas_resistance_ohms"),
             "heat_stable":sample.get("heat_stable",0)}
        self.history.append(row)
        df=add_features(pd.DataFrame(list(self.history)))
        if not self.feature_columns:
            self.feature_columns=[c for c in df.columns if c not in {"Date","location","IAQ_class","IAQ_proxy","temperature_c_zscore","pressure_hpa_zscore","humidity_rh_zscore","gas_resistance_ohms_zscore","IAQ_proxy_zscore","Z-score","samples_per_hour"} and pd.api.types.is_numeric_dtype(df[c])]
        x=df.iloc[[-1]][self.feature_columns]
        pred=str(self.model.predict(x)[0])
        confidence=0.0
        if hasattr(self.model,"predict_proba"):
            p=self.model.predict_proba(x)[0]; confidence=float(max(p))
        return {"iaq_class":pred,"confidence":confidence,"source":"ML","model_name":self.model_name,"model_version":self.model_version}
