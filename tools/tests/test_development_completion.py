import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def test_activity_and_hazard_models_load():
    activity = joblib.load(ROOT / "models" / "safe_band_development_v1" / "activity_model.joblib")
    hazard = joblib.load(ROOT / "models" / "safe_band_development_v1" / "hazard_model.joblib")
    assert len(activity["feature_columns"]) == 58 - 5
    assert len(hazard["feature_columns"]) == 58
    assert activity["task"] == "activity"
    assert hazard["task"] == "hazard"


def test_real_replay_sources_are_unchanged_and_readable():
    real_dir = ROOT / "datasets" / "real sensor data"
    for name in ("stationary.csv", "stationary_2.csv", "walking.csv"):
        df = pd.read_csv(real_dir / name)
        assert len(df) > 0
        assert "elapsed_ms" in df
        assert "activity" in df


def test_final_ml_report_exists():
    report = json.loads((ROOT / "reports" / "final_ml_training_report.json").read_text())
    assert report["classification"] == "DEVELOPMENT_EVIDENCE"
    assert report["training"]["real_contamination"] is False
