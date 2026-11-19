import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]


def test_optimization_report_and_model_card():
    report = json.loads((ROOT / "reports" / "ml_optimization_report.json").read_text())
    assert report["classification"] == "DEVELOPMENT_EVIDENCE"
    assert report["leakage"]["result"] == "PASS"
    assert report["inference_parity"]["status"] == "PASS"
    assert report["2_12"]["status"] == "BLOCKED"
    assert "model_card" in report


def test_frozen_models_match_feature_contract():
    activity = joblib.load(ROOT / "models" / "safe_band_optimized_v1" / "activity_model.joblib")
    hazard = joblib.load(ROOT / "models" / "safe_band_optimized_v1" / "hazard_model.joblib")
    assert len(activity["feature_columns"]) == 58
    assert hazard["feature_columns"] == activity["feature_columns"]
    assert hazard["threshold"] == 0.05
