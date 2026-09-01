"""
SAFEBAND AI - Main Application

Main entry point for the SAFEBAND AI prototype.

Pipeline:
    Simulated Sensors
            ↓
    Activity Recognition
            ↓
    Sensor Fusion
            ↓
    Risk Engine
            ↓
    Alert Manager
            ↓
    Cellular / Cloud
            ↓
    Streamlit Dashboard

Run with:
    streamlit run app.py
"""

import time
from datetime import datetime

import streamlit as st

from config.settings import (
    APP_NAME,
    APP_VERSION,
    APP_MODE,
    DEMO_CONFIG,
)

from data.demo_scenarios import (
    get_available_scenarios,
    is_sos_scenario,
)

from data.simulated_data import (
    generate_sensor_data,
    set_simulation_scenario,
)

from ai.activity_recognition import (
    recognize_activity,
)

from ai.sensor_fusion import (
    fuse_sensor_data,
)

from ai.risk_engine import (
    assess_risk,
)

from communication.cellular import (
    get_cellular_status,
    transmit_data,
    transmit_emergency_alert,
)

from communication.cloud import (
    get_cloud_status,
    upload_sensor_data,
    upload_safety_event,
    upload_emergency_event,
)

from dashboard.alerts import (
    create_safety_alert,
    create_emergency_alert,
    create_sos_alert,
    get_alert_history,
    get_current_alert,
    clear_alert,
)

from dashboard.ui import (
    configure_page,
    render_header,
    render_sidebar,
    render_status_banner,
    render_sensor_cards,
    render_ai_analysis,
    render_communication_status,
    render_location,
    render_alert,
    render_alert_history,
    render_charts,
    render_system_overview,
    render_footer,
)

from utils.loggers import (
    log_info,
    log_ai_event,
    log_safety_event,
    log_emergency_event,
    log_communication_event,
    log_cloud_event,
    log_error,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

configure_page()


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():
    """Initialize persistent Streamlit session-state variables."""

    if "history" not in st.session_state:
        st.session_state.history = []

    if "running" not in st.session_state:
        st.session_state.running = True

    if "last_scenario" not in st.session_state:
        st.session_state.last_scenario = "NORMAL"

    if "last_update" not in st.session_state:
        st.session_state.last_update = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "alert_generated" not in st.session_state:
        st.session_state.alert_generated = False


initialize_session_state()


# ============================================================
# HEADER
# ============================================================

render_header()


# ============================================================
# SIDEBAR
# ============================================================

controls = render_sidebar(
    scenario=st.session_state.last_scenario,
    scenarios=get_available_scenarios(),
)

selected_scenario = controls["scenario"]


# ============================================================
# SCENARIO CHANGE
# ============================================================

if selected_scenario != st.session_state.last_scenario:

    set_simulation_scenario(
        selected_scenario
    )

    st.session_state.last_scenario = (
        selected_scenario
    )

    # Clear previous active alert when returning to NORMAL.
    if selected_scenario == "NORMAL":
        clear_alert()

    log_info(
        f"Demo scenario changed to "
        f"{selected_scenario}"
    )


# ============================================================
# GENERATE SENSOR DATA
# ============================================================

try:

    sensor_data = generate_sensor_data(
        selected_scenario
    )

except Exception as error:

    log_error(
        f"Sensor simulation failed: {error}"
    )

    st.error(
        "Unable to generate simulated sensor data."
    )

    st.stop()


# ============================================================
# ACTIVITY RECOGNITION
# ============================================================

try:

    activity_result = recognize_activity(
        sensor_data
    )

    log_ai_event(
        "Activity Recognition",
        (
            f"Activity={activity_result.get('activity', 'UNKNOWN')} | "
            f"Confidence={activity_result.get('confidence', 0):.2f}"
        )
    )

except Exception as error:

    log_error(
        f"Activity recognition failed: {error}"
    )

    activity_result = {
        "activity": "UNKNOWN",
        "confidence": 0.0,
        "description": "Activity recognition unavailable.",
        "emergency": False,
    }


# ============================================================
# SENSOR FUSION
# ============================================================

try:

    fusion_result = fuse_sensor_data(
        sensor_data,
        activity_result
    )

    log_ai_event(
        "Sensor Fusion",
        (
            f"Score={fusion_result.get('fusion_score', 0):.1f} | "
            f"Condition={fusion_result.get('condition', 'NORMAL')}"
        )
    )

except Exception as error:

    log_error(
        f"Sensor fusion failed: {error}"
    )

    fusion_result = {
        "fusion_score": 0.0,
        "confidence": 0.0,
        "condition": "NORMAL",
        "abnormal": False,
        "evidence": {},
    }


# ============================================================
# RISK ENGINE
# ============================================================

try:

    risk_result = assess_risk(
        sensor_data,
        activity_result
    )

    risk_score = float(
        risk_result.get(
            "risk_score",
            0.0
        )
    )

    risk_level = risk_result.get(
        "risk_level",
        "LOW"
    )

    risk_status = risk_result.get(
        "status",
        "SAFE"
    )

    risk_reason = risk_result.get(
        "reason",
        "No abnormal event detected."
    )

except Exception as error:

    log_error(
        f"Risk assessment failed: {error}"
    )

    risk_result = {
        "risk_score": 0.0,
        "risk_level": "LOW",
        "status": "SAFE",
        "reason": "Risk engine unavailable.",
        "emergency": False,
        "alert_required": False,
    }

    risk_score = 0.0
    risk_level = "LOW"
    risk_status = "SAFE"
    risk_reason = "Risk engine unavailable."


# ============================================================
# MANUAL SOS
# ============================================================

manual_sos = is_sos_scenario(
    sensor_data
)


# ============================================================
# ALERT PROCESSING
# ============================================================

try:

    if manual_sos:

        alert = create_sos_alert(
            risk_score=100.0,
            location=(
                f"{sensor_data.get('latitude', 0):.6f}, "
                f"{sensor_data.get('longitude', 0):.6f}"
            )
        )

        log_emergency_event(
            "SOS",
            100.0,
            "Manual SOS activated by user."
        )

    elif risk_result.get(
        "emergency",
        False
    ):

        alert = create_emergency_alert(
            activity=activity_result.get(
                "activity",
                "UNKNOWN"
            ),
            risk_score=risk_score,
            reason=risk_reason
        )

        log_emergency_event(
            activity_result.get(
                "activity",
                "UNKNOWN"
            ),
            risk_score,
            risk_reason
        )

    elif risk_result.get(
        "alert_required",
        False
    ):

        alert = create_safety_alert(
            risk_score=risk_score,
            risk_level=risk_level,
            activity=activity_result.get(
                "activity",
                "UNKNOWN"
            ),
            reason=risk_reason
        )

        log_safety_event(
            activity_result.get(
                "activity",
                "UNKNOWN"
            ),
            risk_score,
            risk_reason
        )

    else:

        alert = None

except Exception as error:

    log_error(
        f"Alert processing failed: {error}"
    )

    alert = None


# ============================================================
# COMMUNICATION
# ============================================================

try:

    cellular_status = get_cellular_status()

    cloud_status = get_cloud_status()

    # --------------------------------------------------------
    # Package data for communication
    # --------------------------------------------------------

    transmission_data = {
        "timestamp": sensor_data.get(
            "timestamp",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ),

        "scenario": selected_scenario,

        "sensors": sensor_data,

        "activity": activity_result,

        "sensor_fusion": fusion_result,

        "risk": risk_result,

        "location": {
            "latitude": sensor_data.get(
                "latitude"
            ),
            "longitude": sensor_data.get(
                "longitude"
            ),
        },
    }

    # --------------------------------------------------------
    # Cloud data upload
    # --------------------------------------------------------

    cloud_response = upload_sensor_data(
        transmission_data
    )

    if cloud_response.get(
        "success",
        False
    ):

        log_cloud_event(
            "Sensor data synchronized."
        )

    # --------------------------------------------------------
    # Cellular transmission
    # --------------------------------------------------------

    cellular_response = transmit_data(
        transmission_data
    )

    if cellular_response.get(
        "success",
        False
    ):

        log_communication_event(
            "EC200U",
            "Sensor data transmitted."
        )

    # --------------------------------------------------------
    # Emergency transmission
    # --------------------------------------------------------

    emergency_condition = (
        manual_sos
        or risk_result.get(
            "emergency",
            False
        )
    )

    if emergency_condition:

        emergency_payload = {
            "timestamp": sensor_data.get(
                "timestamp"
            ),

            "activity": activity_result.get(
                "activity",
                "UNKNOWN"
            ),

            "risk_score": (
                100.0
                if manual_sos
                else risk_score
            ),

            "risk_level": (
                "CRITICAL"
                if manual_sos
                else risk_level
            ),

            "reason": (
                "Manual SOS activated."
                if manual_sos
                else risk_reason
            ),

            "latitude": sensor_data.get(
                "latitude"
            ),

            "longitude": sensor_data.get(
                "longitude"
            ),
        }

        emergency_cellular = (
            transmit_emergency_alert(
                emergency_payload
            )
        )

        emergency_cloud = (
            upload_emergency_event(
                emergency_payload
            )
        )

        if emergency_cellular.get(
            "success",
            False
        ):

            log_communication_event(
                "EC200U",
                "Emergency alert transmitted."
            )

        if emergency_cloud.get(
            "success",
            False
        ):

            log_cloud_event(
                "Emergency event synchronized."
            )

except Exception as error:

    log_error(
        f"Communication pipeline failed: {error}"
    )

    cellular_status = get_cellular_status()
    cloud_status = get_cloud_status()


# ============================================================
# SAVE MONITORING HISTORY
# ============================================================

history_record = {
    "timestamp": sensor_data.get(
        "timestamp",
        datetime.now().strftime(
            "%H:%M:%S"
        )
    ),

    "heart_rate": sensor_data.get(
        "heart_rate",
        0
    ),

    "spo2": sensor_data.get(
        "spo2",
        0
    ),

    "temperature": sensor_data.get(
        "temperature",
        0
    ),

    "motion_intensity": sensor_data.get(
        "motion_intensity",
        0
    ),

    "risk_score": risk_score,

    "activity": activity_result.get(
        "activity",
        "UNKNOWN"
    ),
}

st.session_state.history.append(
    history_record
)


# Keep the dashboard history manageable.
if len(st.session_state.history) > 60:

    st.session_state.history = (
        st.session_state.history[-60:]
    )


st.session_state.last_result = {
    "sensor_data": sensor_data,
    "activity": activity_result,
    "fusion": fusion_result,
    "risk": risk_result,
}

st.session_state.last_update = (
    sensor_data.get("timestamp")
)


# ============================================================
# MAIN STATUS
# ============================================================

render_status_banner(
    status=(
        "EMERGENCY"
        if manual_sos
        else risk_status
    ),

    risk_level=(
        "CRITICAL"
        if manual_sos
        else risk_level
    ),

    risk_score=(
        100.0
        if manual_sos
        else risk_score
    ),
)


# ============================================================
# CURRENT ALERT
# ============================================================

if manual_sos or risk_result.get(
    "emergency",
    False
):

    render_alert(
        get_current_alert()
    )


# ============================================================
# SENSOR SECTION
# ============================================================

if controls["show_sensors"]:

    render_sensor_cards(
        sensor_data
    )

    st.divider()


# ============================================================
# AI SECTION
# ============================================================

if controls["show_ai"]:

    render_ai_analysis(
        activity_result,
        fusion_result,
        risk_result,
    )

    st.divider()


# ============================================================
# LOCATION
# ============================================================

render_location(
    sensor_data
)

st.divider()


# ============================================================
# COMMUNICATION
# ============================================================

if controls["show_communication"]:

    render_communication_status(
        cellular_status,
        cloud_status,
    )

    st.divider()


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

render_system_overview(
    sensor_status={
        "online": 5,
        "total": 5,
    },

    cellular_status=cellular_status,

    cloud_status=cloud_status,
)

st.divider()


# ============================================================
# CHARTS
# ============================================================

render_charts(
    st.session_state.history
)

st.divider()


# ============================================================
# ALERT HISTORY
# ============================================================

if controls["show_history"]:

    render_alert_history(
        get_alert_history()
    )

    st.divider()


# ============================================================
# FOOTER
# ============================================================

render_footer()


# ============================================================
# AUTO REFRESH
# ============================================================

if st.session_state.running:

    time.sleep(
        DEMO_CONFIG.get(
            "data_update_interval",
            2
        )
    )

    st.rerun()