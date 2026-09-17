"""Runtime adapter for the optional SafeBand audio-event classifier."""
from __future__ import annotations
from pathlib import Path
from typing import Any,Dict,Sequence,Tuple
import pandas as pd
from ai.audio_features import FEATURE_COLUMNS,extract_audio_features

class MLAudioModel:
    def __init__(self, model_path: Path, min_samples: int = 16000):
        self.model_path=Path(model_path); self.min_samples=int(min_samples)
        self.model=None; self.feature_columns=list(FEATURE_COLUMNS)
        self.model_name="SafeBand Audio Event Model"; self.model_version="unknown"
        self.load_error=None; self._load()
    def _load(self):
        if not self.model_path.exists():
            self.load_error="Audio model file not found."; return
        try:
            import joblib
            payload=joblib.load(self.model_path)
            if isinstance(payload,dict) and "model" in payload:
                self.model=payload["model"]; self.feature_columns=list(payload.get("feature_columns",FEATURE_COLUMNS))
                self.model_name=str(payload.get("model_name",self.model_name))
                self.model_version=str(payload.get("model_version",self.model_version))
            else: self.model=payload
        except Exception as exc:
            self.load_error=str(exc); self.model=None
    @property
    def available(self): return self.model is not None
    def predict(self,samples:Sequence[float],sample_rate_hz:float)->Tuple[str,float,Dict[str,float]]:
        if not self.available: raise RuntimeError(self.load_error or "Audio model unavailable")
        if len(samples)<self.min_samples: raise ValueError(f"Audio window too short: {len(samples)} < {self.min_samples}")
        f=extract_audio_features(samples,sample_rate_hz)
        X=pd.DataFrame([{c:float(f.get(c,0.0)) for c in self.feature_columns}],columns=self.feature_columns)
        label=str(self.model.predict(X)[0]).upper()
        probs={}
        if hasattr(self.model,"predict_proba"):
            p=self.model.predict_proba(X)[0]; classes=getattr(self.model,"classes_",[])
            probs={str(c).upper():float(v) for c,v in zip(classes,p)}
        return label,float(probs.get(label,0.0)),probs
