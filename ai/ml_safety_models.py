"""
SafeBand AI - ML Safety Model Runtime Adapter

Loads the validated activity and fall joblib artifacts produced by
tools/train_bits2_models.py. This is opt-in and independent of the
existing rule-based safety engine.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
from ai.bits2_features import extract_motion_features, FEATURE_COLUMNS

class MLSafetyModels:
    def __init__(self, model_dir: Path = Path("models")):
        self.model_dir=Path(model_dir)
        self.activity=self._load(self.model_dir/"activity_model.joblib")
        self.fall=self._load(self.model_dir/"fall_detector.joblib")

    @staticmethod
    def _load(path):
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception:
            return None

    @property
    def available(self):
        return self.activity is not None or self.fall is not None

    def _X(self, window):
        f=extract_motion_features(window)
        import pandas as pd
        cols=(self.activity or self.fall).get("feature_columns",FEATURE_COLUMNS)
        return pd.DataFrame([{c:float(f.get(c,0.0)) for c in cols}],columns=cols)

    def predict_activity(self,window)->Tuple[str,float,Dict[str,float]]:
        if self.activity is None:
            raise RuntimeError("Activity model unavailable")
        X=self._X(window); m=self.activity["model"]
        label=str(m.predict(X)[0]).upper()
        probs={}
        if hasattr(m,"predict_proba"):
            for c,p in zip(m.classes_,m.predict_proba(X)[0]):
                probs[str(c).upper()]=float(p)
        return label,probs.get(label,0.0),probs

    def predict_fall(self,window)->Tuple[str,float,Dict[str,float]]:
        if self.fall is None:
            raise RuntimeError("Fall model unavailable")
        X=self._X(window); payload=self.fall; m=payload["model"]
        probs={str(c).upper():float(p) for c,p in zip(m.classes_,m.predict_proba(X)[0])}
        p=probs.get("FALL",0.0)
        threshold=float(payload.get("fall_threshold",0.5))
        return ("FALL" if p>=threshold else "NON_FALL"),p,probs
