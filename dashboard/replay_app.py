from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.feature_engineering.feature_extraction import FEATURE_COLUMNS, extract_features

ACTIVITY_MODEL = ROOT / "models" / "safe_band_development_v1" / "activity_model.joblib"
HAZARD_MODEL = ROOT / "models" / "safe_band_development_v1" / "hazard_model.joblib"
REAL_DIR = ROOT / "datasets" / "real sensor data"
SYNTH_DIR = ROOT / "datasets" / "realistic_synthetic_v2"


def load_payload(path: Path) -> dict:
    return joblib.load(path)


def risk_assessment(activity: str, confidence: float, row: pd.Series) -> dict:
    factors = {}
    if activity == "FALL_OR_HAZARD":
        factors["motion/hazard"] = 65.0
    elif activity == "HIGH_MOTION":
        factors["motion/hazard"] = 30.0
    elif activity == "RUNNING":
        factors["motion/hazard"] = 10.0
    elif activity == "TRANSITION":
        factors["motion/hazard"] = 5.0
    else:
        factors["motion/hazard"] = 0.0
    hr = pd.to_numeric(pd.Series([row.get("final_hr", -999)]), errors="coerce").iloc[0]
    if hr >= 130:
        factors["physiology"] = 15.0
    elif hr >= 110:
        factors["physiology"] = 8.0
    elif 0 < hr < 50:
        factors["physiology"] = 12.0
    else:
        factors["physiology"] = 0.0
    spo2 = pd.to_numeric(pd.Series([row.get("final_spo2", -999)]), errors="coerce").iloc[0]
    if 0 < spo2 < 90:
        factors["SpO2"] = 20.0
    elif 0 < spo2 < 94:
        factors["SpO2"] = 10.0
    else:
        factors["SpO2"] = 0.0
    temperature = pd.to_numeric(pd.Series([row.get("temperature_C", np.nan)]), errors="coerce").iloc[0]
    factors["environment"] = 10.0 if pd.notna(temperature) and (temperature > 45 or temperature < 5) else 0.0
    elapsed_delta = row.get("_delta_ms", 0.0)
    factors["sensor-health"] = 20.0 if elapsed_delta > 1000 else 0.0
    score = float(np.clip(sum(factors.values()), 0.0, 100.0))
    level = "LOW" if score < 30 else "MODERATE" if score < 60 else "HIGH" if score < 80 else "CRITICAL"
    return {"score": score, "level": level, "factors": factors}


st.set_page_config(page_title="SafeBand AI Replay", layout="wide")
st.markdown("# SafeBand AI")
st.caption("Development replay and sensor-fusion demonstration")

source_options = sorted([p.name for p in REAL_DIR.glob("*.csv")] + [f"SYNTH:{p.name}" for p in SYNTH_DIR.glob("*.csv")])
selected = st.sidebar.selectbox("Data source", source_options)
speed = st.sidebar.selectbox("Replay speed", [1, 2, 5, 10], index=1)
play = st.sidebar.button("Play")
paused = st.sidebar.button("Pause")

if selected.startswith("SYNTH:"):
    csv_path = SYNTH_DIR / selected.split(":", 1)[1]
else:
    csv_path = REAL_DIR / selected
df = pd.read_csv(csv_path)
if "elapsed_ms" in df:
    elapsed = pd.to_numeric(df["elapsed_ms"], errors="coerce")
    df["_delta_ms"] = elapsed.diff().fillna(0.0)
else:
    df["_delta_ms"] = 0.0

if "replay_index" not in st.session_state:
    st.session_state.replay_index = 0
if "replay_playing" not in st.session_state:
    st.session_state.replay_playing = True
if play:
    st.session_state.replay_playing = True
if paused:
    st.session_state.replay_playing = False

index = st.sidebar.slider("Current sample", 0, len(df) - 1, st.session_state.replay_index)
st.session_state.replay_index = index
row = df.iloc[index]

activity_payload = load_payload(ACTIVITY_MODEL)
hazard_payload = load_payload(HAZARD_MODEL)
features = extract_features(df.iloc[: index + 1])
feature_row = features.iloc[[-1]][activity_payload["feature_columns"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
activity = str(activity_payload["model"].predict(feature_row)[0]).upper()
activity_confidence = 0.0
if hasattr(activity_payload["model"], "predict_proba"):
    activity_confidence = float(np.max(activity_payload["model"].predict_proba(feature_row)[0]))

hazard_feature_row = features.iloc[[-1]][hazard_payload["feature_columns"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
hazard_probability = 0.0
if hasattr(hazard_payload["model"], "predict_proba"):
    classes = list(hazard_payload["model"].classes_)
    probabilities = hazard_payload["model"].predict_proba(hazard_feature_row)[0]
    if 1 in classes:
        hazard_probability = float(probabilities[classes.index(1)])
hazard = hazard_probability >= float(hazard_payload["threshold"])

risk = risk_assessment(activity, activity_confidence, row)

st.subheader("System Status")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Data Source", "Replay")
c2.metric("Sample", f"{index + 1}/{len(df)}")
c3.metric("Elapsed", f"{row.get('elapsed_ms', 0):.0f} ms")
c4.metric("Stall", "YES" if float(row.get("_delta_ms", 0)) > 1000 else "NO")

st.subheader("Activity and Safety")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Activity AI", activity)
a2.metric("Confidence", f"{activity_confidence:.1%}")
a3.metric("Hazard AI", "HAZARD" if hazard else "NORMAL")
a4.metric("Hazard Probability", f"{hazard_probability:.1%}")

st.progress(min(risk["score"] / 100.0, 1.0))
st.markdown(f"**Development Risk:** {risk['level']} ({risk['score']:.1f}/100)")
st.caption("Contributing factors: " + ", ".join(f"{k}={v:.1f}" for k, v in risk["factors"].items()))

st.subheader("Sensor Telemetry")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Accel Magnitude", f"{np.linalg.norm([row.get('accel_x_m_s2', 0), row.get('accel_y_m_s2', 0), row.get('accel_z_m_s2', 0)]):.2f} m/s²")
s2.metric("Gyro Magnitude", f"{np.linalg.norm([row.get('gyro_x_rad_s', 0), row.get('gyro_y_rad_s', 0), row.get('gyro_z_rad_s', 0)]):.2f} rad/s")
s3.metric("Final HR", "Unavailable" if pd.to_numeric(pd.Series([row.get("final_hr", -999)]), errors="coerce").iloc[0] <= 0 or row.get("final_hr_valid", 0) != 1 else f"{row.get('final_hr'):.0f} bpm")
s4.metric("Final SpO2", "Unavailable" if row.get("final_spo2_valid", 0) != 1 else f"{row.get('final_spo2'):.1f}%")

e1, e2, e3, e4 = st.columns(4)
e1.metric("Temperature", f"{row.get('temperature_C', np.nan):.2f} °C")
e2.metric("Humidity", f"{row.get('humidity_pct', np.nan):.1f}%")
e3.metric("Pressure", f"{row.get('pressure_hPa', np.nan):.2f} hPa")
e4.metric("Battery", f"{row.get('battery_percent', np.nan):.0f}%")

st.subheader("Replay History")
history_start = max(0, index - 150)
history = df.iloc[history_start:index + 1]
chart_df = pd.DataFrame({
    "Elapsed ms": history["elapsed_ms"] if "elapsed_ms" in history else history.index,
    "Acceleration magnitude": np.sqrt(history["accel_x_m_s2"] ** 2 + history["accel_y_m_s2"] ** 2 + history["accel_z_m_s2"] ** 2),
})
st.line_chart(chart_df.set_index("Elapsed ms"))

st.caption("Development demonstration only. Not clinically or independently hardware validated.")

if st.session_state.replay_playing:
    next_index = min(len(df) - 1, index + speed)
    if next_index != index:
        st.session_state.replay_index = next_index
        time.sleep(0.05)
        st.rerun()
