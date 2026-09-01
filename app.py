"""
SAFEBAND AI - Main Application

Main Streamlit entry point for the SAFEBAND AI prototype.

Application pipeline:

    Sensor Layer
        ↓
    Activity Recognition
        ↓
    Sensor Fusion
        ↓
    Risk Assessment
        ↓
    Alert Processing
        ↓
    Cellular / Cloud Communication
        ↓
    Streamlit Dashboard

The application is intentionally kept as an orchestration layer.
Sensor logic, AI logic, communication logic, and dashboard logic
remain in their respective modules.

Run with:

    streamlit run app.py
"""


# ============================================================
# STANDARD LIBRARY
# ============================================================

from datetime import datetime
from typing import Any, Dict


# ============================================================
# STREAMLIT
# ============================================================

import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

from config.settings import (
    DEMO_CONFIG,
    SENSOR_CONFIG,
)


# ============================================================
# DATA / SIMULATION
# ============================================================

from data.demo_scenarios import (
    get_available_scenarios,
    is_sos_scenario,
)

from data.simulated_data import (
    generate_sensor_data,
    set_simulation_scenario,
)


# ============================================================
# AI / ANALYSIS
# ============================================================

from ai.activity_recognition import (
    recognize_activity,
)

from ai.sensor_fusion import (
    fuse_sensor_data,
)

from ai.risk_engine import (
    assess_risk,
)


# ============================================================
# COMMUNICATION
# ============================================================

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


# ============================================================
# ALERTS
# ============================================================

from dashboard.alerts import (
    create_safety_alert,
    create_emergency_alert,
    create_sos_alert,
    get_alert_history,
    get_current_alert,
    clear_alert,
)


# ============================================================
# DASHBOARD
# ============================================================

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


# ============================================================
# LOGGING
# ============================================================

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
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    """
    Initialize persistent Streamlit session-state values.
    """

    if "history" not in st.session_state:
        st.session_state.history = []

    if "running" not in st.session_state:
        st.session_state.running = True

    if "last_scenario" not in st.session_state:
        st.session_state.last_scenario = (
            DEMO_CONFIG.get(
                "default_scenario",
                "NORMAL",
            )
        )

    if "last_update" not in st.session_state:
        st.session_state.last_update = None

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "manual_sos" not in st.session_state:
        st.session_state.manual_sos = False

    if "last_alert_signature" not in st.session_state:
        st.session_state.last_alert_signature = None

    if "last_emergency_signature" not in st.session_state:
        st.session_state.last_emergency_signature = None


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


# ============================================================
# CONTROL VALUES
# ============================================================

selected_scenario = str(
    controls.get(
        "scenario",
        st.session_state.last_scenario,
    )
).upper()

simulation_enabled = bool(
    controls.get(
        "simulation",
        True,
    )
)

manual_sos_from_ui = bool(
    controls.get(
        "manual_sos",
        False,
    )
)


# ============================================================
# MANUAL SOS STATE
# ============================================================

st.session_state.manual_sos = manual_sos_from_ui


# ============================================================
# SCENARIO CHANGE
# ============================================================

if selected_scenario != st.session_state.last_scenario:

    try:

        set_simulation_scenario(
            selected_scenario
        )

        st.session_state.last_scenario = (
            selected_scenario
        )

        # A new scenario is a new demonstration state.
        st.session_state.last_alert_signature = None
        st.session_state.last_emergency_signature = None

        if selected_scenario == "NORMAL":
            clear_alert()

        log_info(
            "Demo scenario changed to "
            f"{selected_scenario}"
        )

    except Exception as error:

        log_error(
            f"Unable to change simulation scenario: {error}"
        )

        st.error(
            "Unable to change the selected scenario."
        )


# ============================================================
# SENSOR DATA
# ============================================================

try:

    # --------------------------------------------------------
    # Current prototype uses deterministic simulated profiles.
    # --------------------------------------------------------

    if simulation_enabled:

        sensor_data = generate_sensor_data(
            selected_scenario
        )

    else:

        # Hardware sensor aggregation will replace this branch
        # when the physical sensor pipeline is connected.
        st.warning(
            "Hardware mode is selected, but the physical "
            "sensor aggregation pipeline is not yet connected. "
            "Enable Simulation Mode for prototype testing."
        )

        sensor_data = generate_sensor_data(
            selected_scenario
        )

except Exception as error:

    log_error(
        f"Sensor acquisition failed: {error}"
    )

    st.error(
        "Unable to acquire sensor data."
    )

    st.stop()


# ============================================================
# NORMALIZE SENSOR DATA
# ============================================================

if "body_temperature" not in sensor_data:

    sensor_data["body_temperature"] = None


# ============================================================
# APPLY EXPLICIT MANUAL SOS
# ============================================================

# The sidebar TEST SOS button represents an explicit user action.
# It must not be confused with automatic activity recognition.

if manual_sos_from_ui:

    sensor_data["manual_sos"] = True


manual_sos = (
    is_sos_scenario(
        sensor_data
    )
    or manual_sos_from_ui
)


# ============================================================
# ACTIVITY RECOGNITION
# ============================================================

try:

    activity_result = recognize_activity(
        sensor_data
    )

    activity_name = str(
        activity_result.get(
            "activity",
            "UNKNOWN",
        )
    ).upper()

    activity_confidence = float(
        activity_result.get(
            "confidence",
            0.0,
        )
    )

    # Normalize percentage-style confidence if necessary.
    if activity_confidence > 1.0:
        activity_confidence /= 100.0

    log_ai_event(
        "Activity Recognition",
        (
            f"Activity={activity_name} | "
            f"Confidence={activity_confidence:.2f}"
        ),
    )

except Exception as error:

    log_error(
        f"Activity recognition failed: {error}"
    )

    activity_result = {
        "activity": "UNKNOWN",
        "confidence": 0.0,
        "description": (
            "Activity recognition unavailable."
        ),
        "emergency": False,
    }


# ============================================================
# SENSOR FUSION
# ============================================================

try:

    fusion_result = fuse_sensor_data(
        sensor_data,
        activity_result,
    )

    fusion_score = float(
        fusion_result.get(
            "fusion_score",
            0.0,
        )
    )

    fusion_condition = str(
        fusion_result.get(
            "condition",
            "NORMAL",
        )
    ).upper()

    log_ai_event(
        "Sensor Fusion",
        (
            f"Score={fusion_score:.1f} | "
            f"Condition={fusion_condition}"
        ),
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
# RISK ASSESSMENT
# ============================================================

try:

    risk_result = assess_risk(
        sensor_data,
        activity_result,
    )

    risk_score = float(
        risk_result.get(
            "risk_score",
            0.0,
        )
    )

    risk_level = str(
        risk_result.get(
            "risk_level",
            "LOW",
        )
    ).upper()

    risk_status = str(
        risk_result.get(
            "status",
            "SAFE",
        )
    ).upper()

    risk_reason = str(
        risk_result.get(
            "reason",
            "No abnormal event detected.",
        )
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
# SOS OVERRIDE
# ============================================================

if manual_sos:

    risk_score = 100.0
    risk_level = "CRITICAL"
    risk_status = "EMERGENCY"
    risk_reason = "Manual SOS activated by user."

    risk_result["risk_score"] = 100.0
    risk_result["risk_level"] = "CRITICAL"
    risk_result["status"] = "EMERGENCY"
    risk_result["reason"] = risk_reason
    risk_result["emergency"] = True
    risk_result["alert_required"] = True

    # SOS is a user emergency action, not an inferred activity.
    activity_result["emergency"] = True


# ============================================================
# ALERT PROCESSING
# ============================================================

alert = None

try:
    # --------------------------------------------------------
    # Build a stable signature for the current safety event.
    #
    # Streamlit reruns the entire script on every refresh.
    # The signature prevents one continuous FALL/SOS condition
    # from creating a new alert on every rerun.
    # --------------------------------------------------------
    if manual_sos:
        event_signature = (
            "SOS",
            selected_scenario,
        )
    elif risk_result.get("emergency", False):
        event_signature = (
            "EMERGENCY",
            activity_name,
            risk_level,
        )
    elif risk_result.get("alert_required", False):
        event_signature = (
            "WARNING",
            activity_name,
            risk_level,
        )
    else:
        event_signature = None

    is_new_event = (
        event_signature is not None
        and event_signature != st.session_state.last_alert_signature
    )

    # --------------------------------------------------------
    # Create a new alert only for a new event.
    # --------------------------------------------------------
    if manual_sos:
        latitude = sensor_data.get("latitude", 0.0)
        longitude = sensor_data.get("longitude", 0.0)

        try:
            location = (
                f"{float(latitude):.6f}, "
                f"{float(longitude):.6f}"
            )
        except (TypeError, ValueError):
            location = "Location unavailable"

        if is_new_event:
            alert = create_sos_alert(
                risk_score=100.0,
                location=location,
            )

            log_emergency_event(
                "SOS",
                100.0,
                "Manual SOS activated by user.",
            )

            st.session_state.last_alert_signature = event_signature
        else:
            alert = get_current_alert()

    elif risk_result.get("emergency", False):
        activity = activity_result.get(
            "activity",
            "UNKNOWN",
        )

        if is_new_event:
            alert = create_emergency_alert(
                activity=activity,
                risk_score=risk_score,
                reason=risk_reason,
            )

            log_emergency_event(
                activity,
                risk_score,
                risk_reason,
            )

            st.session_state.last_alert_signature = event_signature
        else:
            alert = get_current_alert()

    elif risk_result.get("alert_required", False):
        activity = activity_result.get(
            "activity",
            "UNKNOWN",
        )

        if is_new_event:
            alert = create_safety_alert(
                risk_score=risk_score,
                risk_level=risk_level,
                activity=activity,
                reason=risk_reason,
            )

            log_safety_event(
                activity,
                risk_score,
                risk_reason,
            )

            st.session_state.last_alert_signature = event_signature
        else:
            alert = get_current_alert()

    else:
        alert = get_current_alert()

        # Once the system returns to a non-alert state, the
        # previous event is considered closed. This allows a
        # later FALL/SOS to generate a fresh event.
        st.session_state.last_alert_signature = None
        st.session_state.last_emergency_signature = None

        if risk_level == "LOW" and not manual_sos:
            clear_alert()
            alert = None

except Exception as error:
    log_error(
        f"Alert processing failed: {error}"
    )
    alert = get_current_alert()


# ============================================================
# COMMUNICATION STATUS
# ============================================================

try:

    cellular_status = get_cellular_status()

except Exception as error:

    log_error(
        f"Unable to obtain cellular status: {error}"
    )

    cellular_status = {
        "connected": False,
        "signal_strength": 0,
        "network": "Unavailable",
    }


try:

    cloud_status = get_cloud_status()

except Exception as error:

    log_error(
        f"Unable to obtain cloud status: {error}"
    )

    cloud_status = {
        "connected": False,
        "records_uploaded": 0,
        "last_message": "Unavailable",
    }


# ============================================================
# COMMUNICATION PIPELINE
# ============================================================

try:
    timestamp = sensor_data.get(
        "timestamp",
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    transmission_data = {
        "timestamp": timestamp,
        "scenario": selected_scenario,
        "sensors": sensor_data,
        "activity": activity_result,
        "sensor_fusion": fusion_result,
        "risk": risk_result,
        "location": {
            "latitude": sensor_data.get("latitude"),
            "longitude": sensor_data.get("longitude"),
        },
    }

    # --------------------------------------------------------
    # CLOUD SENSOR DATA
    # --------------------------------------------------------
    try:
        cloud_response = upload_sensor_data(
            transmission_data
        )

        if cloud_response.get("success", False):
            log_cloud_event(
                "Sensor data synchronized."
            )

    except Exception as error:
        log_error(
            f"Cloud sensor upload failed: {error}"
        )

    # --------------------------------------------------------
    # CELLULAR SENSOR DATA
    # --------------------------------------------------------
    try:
        cellular_response = transmit_data(
            transmission_data
        )

        if cellular_response.get("success", False):
            log_communication_event(
                "EC200U",
                "Sensor data transmitted.",
            )

    except Exception as error:
        log_error(
            f"Cellular sensor transmission failed: {error}"
        )

    # --------------------------------------------------------
    # EMERGENCY COMMUNICATION
    #
    # IMPORTANT:
    # Only transmit once when a new emergency event begins.
    # Streamlit reruns must not generate duplicate emergency
    # cellular/cloud records.
    # --------------------------------------------------------
    emergency_condition = (
        manual_sos
        or risk_result.get("emergency", False)
    )

    if emergency_condition:
        emergency_signature = (
            "SOS",
            selected_scenario,
        ) if manual_sos else (
            "EMERGENCY",
            activity_name,
            risk_level,
        )

        is_new_emergency = (
            emergency_signature
            != st.session_state.last_emergency_signature
        )

        emergency_payload = {
            "timestamp": timestamp,
            "activity": activity_result.get(
                "activity",
                "UNKNOWN",
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
            "status": "EMERGENCY",
            "reason": (
                "Manual SOS activated."
                if manual_sos
                else risk_reason
            ),
            "latitude": sensor_data.get("latitude"),
            "longitude": sensor_data.get("longitude"),
            "heart_rate": sensor_data.get("heart_rate"),
            "spo2": sensor_data.get("spo2"),
            "body_temperature": sensor_data.get(
                "body_temperature"
            ),
            "sensor_data": sensor_data,
        }

        if is_new_emergency:
            # ------------------------------------------------
            # EC200U
            # ------------------------------------------------
            try:
                emergency_cellular = (
                    transmit_emergency_alert(
                        emergency_payload
                    )
                )

                if emergency_cellular.get(
                    "success",
                    False,
                ):
                    log_communication_event(
                        "EC200U",
                        "Emergency alert transmitted.",
                    )

            except Exception as error:
                log_error(
                    "Emergency cellular transmission failed: "
                    f"{error}"
                )

            # ------------------------------------------------
            # CLOUD
            # ------------------------------------------------
            try:
                emergency_cloud = (
                    upload_emergency_event(
                        emergency_payload
                    )
                )

                if emergency_cloud.get(
                    "success",
                    False,
                ):
                    log_cloud_event(
                        "Emergency event synchronized."
                    )

            except Exception as error:
                log_error(
                    "Emergency cloud upload failed: "
                    f"{error}"
                )

            st.session_state.last_emergency_signature = (
                emergency_signature
            )

    else:
        # Emergency condition has ended. A future emergency
        # should therefore be allowed to transmit again.
        st.session_state.last_emergency_signature = None

except Exception as error:
    log_error(
        f"Communication pipeline failed: {error}"
    )

    try:
        cellular_status = get_cellular_status()
    except Exception:
        cellular_status = {
            "connected": False,
            "signal_strength": 0,
            "network": "Unavailable",
        }

    try:
        cloud_status = get_cloud_status()
    except Exception:
        cloud_status = {
            "connected": False,
            "records_uploaded": 0,
            "last_message": "Unavailable",
        }


# ============================================================
# MONITORING HISTORY
# ============================================================

history_record = {
    "timestamp": sensor_data.get(
        "timestamp",
        datetime.now().strftime(
            "%H:%M:%S"
        ),
    ),

    # Physiological data.
    "heart_rate": sensor_data.get(
        "heart_rate",
        0.0,
    ),

    "spo2": sensor_data.get(
        "spo2",
        0.0,
    ),

    # BME680 environmental temperature.
    "temperature": sensor_data.get(
        "temperature",
        0.0,
    ),

    # MAX30208 body temperature.
    "body_temperature": sensor_data.get(
        "body_temperature",
        None,
    ),

    # Motion.
    "motion_intensity": sensor_data.get(
        "motion_intensity",
        0.0,
    ),

    # AI.
    "activity": activity_result.get(
        "activity",
        "UNKNOWN",
    ),

    "activity_confidence": activity_result.get(
        "confidence",
        0.0,
    ),

    # Fusion.
    "fusion_score": fusion_result.get(
        "fusion_score",
        0.0,
    ),

    # Risk.
    "risk_score": risk_score,

    "risk_level": risk_level,

    # SOS.
    "manual_sos": manual_sos,
}


# ============================================================
# HISTORY APPEND
# ============================================================

st.session_state.history.append(
    history_record
)


# ============================================================
# HISTORY LIMIT
# ============================================================

history_limit = int(
    DEMO_CONFIG.get(
        "history_limit",
        60,
    )
)

history_limit = max(
    10,
    history_limit,
)

if len(
    st.session_state.history
) > history_limit:

    st.session_state.history = (
        st.session_state.history[
            -history_limit:
        ]
    )


# ============================================================
# LAST RESULT
# ============================================================

st.session_state.last_result = {
    "sensor_data": sensor_data,

    "activity": activity_result,

    "fusion": fusion_result,

    "risk": risk_result,

    "alert": alert,

    "cellular": cellular_status,

    "cloud": cloud_status,
}

st.session_state.last_update = (
    sensor_data.get(
        "timestamp"
    )
)


# ============================================================
# MAIN STATUS
# ============================================================

display_status = (
    "EMERGENCY"
    if manual_sos
    else risk_status
)

display_risk_level = (
    "CRITICAL"
    if manual_sos
    else risk_level
)

display_risk_score = (
    100.0
    if manual_sos
    else risk_score
)


render_status_banner(
    status=display_status,
    risk_level=display_risk_level,
    risk_score=display_risk_score,
)


# ============================================================
# CURRENT ALERT
# ============================================================

current_alert = get_current_alert()

if current_alert is not None:

    render_alert(
        current_alert
    )


# ============================================================
# SENSOR SECTION
# ============================================================

if controls.get(
    "show_sensors",
    True,
):

    render_sensor_cards(
        sensor_data
    )

    st.divider()


# ============================================================
# AI SECTION
# ============================================================

if controls.get(
    "show_ai",
    True,
):

    render_ai_analysis(
        activity_result,
        fusion_result,
        risk_result,
    )

    st.divider()


# ============================================================
# LOCATION
# ============================================================

if controls.get(
    "show_location",
    True,
):

    render_location(
        sensor_data
    )

    st.divider()


# ============================================================
# COMMUNICATION
# ============================================================

if controls.get(
    "show_communication",
    True,
):

    render_communication_status(
        cellular_status,
        cloud_status,
    )

    st.divider()


# ============================================================
# SENSOR STATUS
# ============================================================

def get_sensor_status_summary() -> Dict[str, Any]:
    """
    Calculate the configured sensor count.

    A sensor is counted as available when enabled in
    SENSOR_CONFIG.
    """

    enabled_sensors = [
        config
        for config in SENSOR_CONFIG.values()
        if config.get(
            "enabled",
            True,
        )
    ]

    total = len(
        enabled_sensors
    )

    return {
        "online": total,
        "total": total,
    }


sensor_status = (
    get_sensor_status_summary()
)


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

render_system_overview(
    sensor_status=sensor_status,
    cellular_status=cellular_status,
    cloud_status=cloud_status,
)

st.divider()


# ============================================================
# CHARTS
# ============================================================

if controls.get(
    "show_charts",
    True,
):

    render_charts(
        st.session_state.history
    )

    st.divider()


# ============================================================
# ALERT HISTORY
# ============================================================

if controls.get(
    "show_history",
    True,
):

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

    refresh_seconds = float(
        DEMO_CONFIG.get(
            "data_update_interval",
            2,
        )
    )

    refresh_seconds = max(
        0.5,
        refresh_seconds,
    )

    try:

        from streamlit_autorefresh import (
            st_autorefresh,
        )

        st_autorefresh(
            interval=int(
                refresh_seconds * 1000
            ),
            key="safeband_auto_refresh",
        )

    except ImportError:

        st.info(
            "Install streamlit-autorefresh to enable "
            "automatic dashboard updates."
        )