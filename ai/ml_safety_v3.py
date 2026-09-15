"""
SafeBand AI — ML v3 runtime adapter.

This adapter is intentionally standalone. Existing rule-based inference remains
the authoritative fallback until v3 is validated on SafeBand hardware.
"""
from pathlib import Path
import joblib, pandas as pd
from ai.bits2_features import extract_motion_features
from ai.stationary_features import stationary_features
from ai.fall_event_features import fall_event_features

class MLSafetyV3:
    def __init__(self,model_dir="models"):
        d=Path(model_dir)
        self.activity=joblib.load(d/"activity_model_v3.joblib")
        self.fall=joblib.load(d/"fall_detector_v3.joblib")
    def _x(self,w,payload,extra):
        f=extract_motion_features(w); f.update(extra(w))
        cols=payload["feature_columns"]
        return pd.DataFrame([{c:float(f.get(c,0.0)) for c in cols}],columns=cols)
    def activity_predict(self,w):
        p=self.activity["model"]; x=self._x(w,self.activity,stationary_features)
        probs={str(c):float(v) for c,v in zip(p.classes_,p.predict_proba(x)[0])}
        label=max(probs,key=probs.get)
        return label,probs[label],probs
    def fall_predict(self,w):
        p=self.fall["model"]; x=self._x(w,self.fall,fall_event_features)
        probs={str(c):float(v) for c,v in zip(p.classes_,p.predict_proba(x)[0])}
        pf=probs.get("FALL",0.0); th=float(self.fall.get("fall_threshold",.5))
        return ("FALL" if pf>=th else "NON_FALL"),pf,probs
