"""
SAFEBAND AI - ML Activity Model Adapter

Runtime adapter for a trained scikit-learn/joblib activity model.
No model is shipped yet. Dataset research and training will produce
models/activity_model.joblib later.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.feature_extraction import FEATURE_COLUMNS, extract_features
from config.settings import AI_MODEL_PATH


class MLActivityModel:
    """Load and execute a validated SafeBand activity model."""

    def __init__(self, model_path: Path = AI_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model: Any = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.model_name = "SafeBand Activity Model"
        self.model_version = "unknown"
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = "Model file not found."
            return

        try:
            import joblib

            payload = joblib.load(self.model_path)

            if isinstance(payload, dict) and "model" in payload:
                self.model = payload["model"]
                self.feature_columns = list(
                    payload.get("feature_columns", FEATURE_COLUMNS)
                )
                self.model_name = str(
                    payload.get("model_name", self.model_name)
                )
                self.model_version = str(
                    payload.get("model_version", self.model_version)
                )
            else:
                self.model = payload

        except Exception as exc:
            self.load_error = str(exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(
        self,
        window: List[Dict[str, Any]],
    ) -> Tuple[str, float, Dict[str, float]]:
        """Predict activity from a SafeBand sensor window."""
        if not self.available:
            raise RuntimeError(self.load_error or "ML model unavailable.")

        features = extract_features(window)
        row = {name: float(features.get(name, 0.0)) for name in self.feature_columns}

        # Most SafeBand training pipelines will use a pandas DataFrame
        # so sklearn estimators with feature-name validation remain happy.
        try:
            import pandas as pd
            X = pd.DataFrame([row], columns=self.feature_columns)
        except ImportError:
            X = [[row[name] for name in self.feature_columns]]

        prediction = self.model.predict(X)[0]
        label = str(prediction).upper()
        confidence = 0.0
        probabilities: Dict[str, float] = {}

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = getattr(self.model, "classes_", [])
            for cls, probability in zip(classes, proba):
                probabilities[str(cls).upper()] = float(probability)
            if probabilities:
                confidence = probabilities.get(label, max(probabilities.values()))

        return label, confidence, probabilities


__all__ = ["MLActivityModel"]
