"""SafeBand V5 runtime adapter for the dataset-balanced temporal fall model.

This is deliberately separate from the existing runtime so V5 can be tested
without changing the working prototype. The returned probability is intended
to be fed into ai.fall_event_engine.FallEventConfirmator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

from ai.bits2_features import extract_motion_features


class MLSafetyV5:
    def __init__(self, model_path: str | Path = "models/multidataset_v5_event/multidataset_v5_event_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.payload = joblib.load(self.model_path) if self.model_path.exists() else None

    @property
    def available(self) -> bool:
        return isinstance(self.payload, dict) and self.payload.get("model") is not None

    def predict_fall_probability(self, window) -> float:
        if not self.available:
            raise RuntimeError(f"V5 model unavailable: {self.model_path}")
        features = extract_motion_features(window)
        columns = self.payload["feature_columns"]
        x = pd.DataFrame([{c: float(features.get(c, 0.0)) for c in columns}], columns=columns)
        model = self.payload["model"]
        classes = [str(c).upper() for c in model.classes_]
        probs = model.predict_proba(x)[0]
        try:
            return float(probs[classes.index("FALL")])
        except ValueError:
            raise RuntimeError("V5 artifact does not contain a FALL class")

    def predict_fall(self, window) -> Tuple[str, float, Dict[str, float]]:
        p = self.predict_fall_probability(window)
        threshold = float(self.payload.get("event_config", {}).get("threshold", 0.75))
        return ("FALL" if p >= threshold else "NON_FALL"), p, {
            "FALL": p,
            "NON_FALL": 1.0 - p,
        }
