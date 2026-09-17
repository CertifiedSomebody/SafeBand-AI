"""SafeBand Phase-1 software preflight.

Run from repository root: python tools\\preflight_phase1.py
Checks imports, dataset presence, and discovers expected model artifacts.
Missing optional artifacts are reported as [--] but do not invalidate imports.
"""
from __future__ import annotations
import importlib, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

REQUIRED = [
    "ai.activity_recognition", "ai.feature_extraction", "ai.ml_activity_model",
    "ai.audio_features", "ai.ml_audio_model", "ai.audio_event_recognition",
    "ai.bme680_features", "ai.ml_bme680_model", "ai.ppg_v4_features",
    "ai.walking_features_v5_3", "sensors.inmp441", "sensors.bme680",
    "ai.sensor_fusion", "ai.risk_engine", "config.settings",
]

def first_existing(label, candidates):
    for p in candidates:
        if p.exists():
            print(f"[OK] {label}: {p}")
            return True
    print(f"[--] {label}: no artifact found")
    for p in candidates:
        print(f"     expected: {p}")
    return False

def main():
    failures = []
    for name in REQUIRED:
        try:
            importlib.import_module(name); print(f"[OK] import {name}")
        except Exception as exc:
            failures.append((name, str(exc))); print(f"[FAIL] import {name}: {exc}")

    print("\nArtifacts:")
    first_existing("activity model", [ROOT/"models/activity_bits2_v2/activity_model.joblib", ROOT/"models/activity_model.joblib"])
    first_existing("fall model", [ROOT/"models/activity_bits2_v2/fall_detector.joblib", ROOT/"models/fall_detector.joblib"])
    first_existing("BME680 V1 model", [ROOT/"models/bme680_airwise_v1/bme680_airwise_v1.joblib"])
    first_existing("PPG V4.3 model", [ROOT/"models/ppg_v4_3/ppg_v4_3_max30101.joblib", ROOT/"models/ppg_v4/ppg_v4_max30101.joblib"])
    first_existing("audio model", [ROOT/"models/audio_event_v1/audio_event_model.joblib"])

    print("\nRaw datasets:")
    for name, p in [("BITS-2",ROOT/"datasets/raw/BITS-2"),("PTT",ROOT/"datasets/raw/PTT"),("Wrist PPG Exercise",ROOT/"datasets/raw/WristPPGExercise"),("AIRWISE",ROOT/"datasets/raw/AIRWISE")]:
        status = "OK" if p.exists() else "--"
        print(f"[{status}] {name}: {p}")

    if failures:
        raise SystemExit(f"{len(failures)} import check(s) failed.")
    print("\nPHASE-1 PREFLIGHT: PASS")

if __name__ == "__main__": main()
