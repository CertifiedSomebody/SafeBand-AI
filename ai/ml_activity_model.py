"""Runtime adapters for the validated SafeBand motion models.

The first BITS-2 training phase uses two separate estimators:
- activity model: RESTING / SITTING / WALKING / RUNNING
- fall detector: FALL / NON_FALL

Separating them prevents the many overlapping fall windows from dominating
ordinary activity classification and gives the safety layer an explicit fall
probability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.feature_extraction import FEATURE_COLUMNS, extract_features
from config.settings import AI_FALL_MODEL_PATH, AI_MODEL_PATH


class MLActivityModel:
    """Load and execute SafeBand activity and fall models."""

    def __init__(self, model_path: Path = AI_MODEL_PATH, fall_model_path: Path = AI_FALL_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.fall_model_path = Path(fall_model_path)
        self.model: Any = None
        self.fall_model: Any = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.model_name = "SafeBand Activity Model"
        self.model_version = "unknown"
        self.fall_model_name = "SafeBand Fall Detector"
        self.fall_model_version = "unknown"
        self.load_error: Optional[str] = None
        self.fall_load_error: Optional[str] = None
        self._load_activity()
        self._load_fall()

    @staticmethod
    def _load_payload(path: Path):
        import joblib
        return joblib.load(path)

    def _load_activity(self) -> None:
        if not self.model_path.exists():
            self.load_error = "Activity model file not found."
            return
        try:
            payload = self._load_payload(self.model_path)
            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.feature_columns = list(payload.get("feature_columns", FEATURE_COLUMNS))
                self.model_name = str(payload.get("model_name", self.model_name))
                self.model_version = str(payload.get("model_version", self.model_version))
            else:
                self.model = payload
        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    def _load_fall(self) -> None:
        if not self.fall_model_path.exists():
            self.fall_load_error = "Fall detector file not found."
            return
        try:
            payload = self._load_payload(self.fall_model_path)
            if isinstance(payload, dict) and "model" in payload:
                self.fall_model = payload["model"]
                self.fall_model_name = str(payload.get("model_name", self.fall_model_name))
                self.fall_model_version = str(payload.get("model_version", self.fall_model_version))
            else:
                self.fall_model = payload
        except Exception as exc:
            self.fall_load_error = str(exc)
            self.fall_model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @property
    def fall_available(self) -> bool:
        return self.fall_model is not None

    def _frame(self, window: List[Dict[str, Any]]):
        import pandas as pd
        features = extract_features(window)
        row = {name: float(features.get(name, 0.0)) for name in self.feature_columns}
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict(self, window: List[Dict[str, Any]]) -> Tuple[str, float, Dict[str, float]]:
        if not self.available:
            raise RuntimeError(self.load_error or "ML activity model unavailable.")
        X = self._frame(window)
        prediction = self.model.predict(X)[0]
        label = str(prediction).upper()
        probabilities: Dict[str, float] = {}
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = getattr(self.model, "classes_", [])
            for cls, probability in zip(classes, proba):
                probabilities[str(cls).upper()] = float(probability)
            confidence = probabilities.get(label, max(probabilities.values(), default=0.0))
        return label, float(confidence), probabilities

    def predict_fall(self, window: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """Return P(FALL) and the complete binary probability map."""
        if not self.fall_available:
            raise RuntimeError(self.fall_load_error or "ML fall detector unavailable.")
        X = self._frame(window)
        probabilities: Dict[str, float] = {}
        if hasattr(self.fall_model, "predict_proba"):
            proba = self.fall_model.predict_proba(X)[0]
            classes = getattr(self.fall_model, "classes_", [])
            for cls, probability in zip(classes, proba):
                probabilities[str(cls).upper()] = float(probability)
            return float(probabilities.get("FALL", 0.0)), probabilities
        prediction = str(self.fall_model.predict(X)[0]).upper()
        return (1.0 if prediction == "FALL" else 0.0), {prediction: 1.0}


__all__ = ["MLActivityModel"]
