"""
SAFEBAND AI - Dashboard UI Module

Streamlit presentation layer for SAFEBAND AI.

Responsibilities:
    - Page configuration
    - Application header and sidebar
    - Runtime simulation control
    - SOS test control
    - Sensor monitoring
    - AI/activity analysis
    - Sensor-fusion results
    - Risk assessment
    - Communication status
    - GPS location
    - Alerts and event history
    - Monitoring charts
    - System overview
    - Footer

Design principles:
    - Keep presentation separate from backend logic.
    - Do not manually select an inferred activity.
    - Keep environmental and body temperature separate.
    - Remain compatible with future TinyML integration.
    - Prefer Streamlit-native components.
    - Avoid theme-dependent hard-coded text colors.
"""


from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from config.settings import (
    APP_NAME,
    APP_VERSION,
    APP_MODE,
    DASHBOARD_CONFIG,
    SIMULATION_MODE,
    SIMULATION_MODE_TOGGLE,
    get_available_scenarios,
    get_simulation_mode,
    set_simulation_mode,
)

from dashboard.charts import (
    create_heart_rate_chart,
    create_spo2_chart,
    create_temperature_chart,
    create_body_temperature_chart,
    create_risk_chart,
    create_motion_chart,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def configure_page() -> None:
    """Configure the Streamlit application page."""

    st.set_page_config(
        page_title=DASHBOARD_CONFIG.get(
            "page_title",
            APP_NAME,
        ),
        page_icon=DASHBOARD_CONFIG.get(
            "page_icon",
            "🛡️",
        ),
        layout=DASHBOARD_CONFIG.get(
            "layout",
            "wide",
        ),
        initial_sidebar_state="expanded",
    )


# ============================================================
# HEADER
# ============================================================

def render_header() -> None:
    """Render the main SAFEBAND AI application header."""

    left, center, right = st.columns(
        [2.5, 4, 2.5]
    )

    with left:

        st.markdown(
            f"## 🛡️ {APP_NAME}"
        )

        st.caption(
            "Intelligent Safety Monitoring System"
        )

    with center:

        st.markdown(
            "<div style='text-align:center;'>"
            "<strong>REAL-TIME SAFETY MONITORING PROTOTYPE</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            "Multi-sensor monitoring • Sensor fusion • "
            "AI-ready architecture"
        )

    with right:

        st.markdown(
            "<div style='text-align:right;'>"
            "<strong>● PROTOTYPE</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Version {APP_VERSION}"
        )

    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
    scenario: str = "NORMAL",
    scenarios: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Render dashboard runtime and visibility controls.

    IMPORTANT
    ---------
    Activity is NOT manually selected here.

    The scenario control is a demonstration data source only.
    Actual activity displayed by the dashboard must come from
    the AI/activity-recognition pipeline.

    The SOS control is an explicit emergency input and is
    intentionally separate from activity inference.
    """

    st.sidebar.title(
        "🛡️ SAFEBAND AI"
    )

    st.sidebar.caption(
        "System Control Panel"
    )

    # ========================================================
    # SYSTEM MODE
    # ========================================================

    st.sidebar.subheader(
        "System Mode"
    )

    current_simulation = get_simulation_mode()

    simulation_enabled = st.sidebar.checkbox(
        "Simulation Mode",
        value=current_simulation,
        disabled=not SIMULATION_MODE_TOGGLE,
        help=(
            "Enable simulated sensor data for demonstration "
            "and testing. Disable this when real hardware "
            "integration is enabled."
        ),
        key="simulation_mode_toggle",
    )

    # Update runtime configuration if the user changed it.
    if simulation_enabled != current_simulation:
        set_simulation_mode(
            simulation_enabled
        )

        # Keep the application state synchronized.
        st.rerun()

    if simulation_enabled:

        st.sidebar.info(
            "Sensor values are currently simulated."
        )

    else:

        st.sidebar.warning(
            "Hardware mode selected. "
            "Real hardware must be enabled in settings."
        )

    st.sidebar.divider()

    # ========================================================
    # DEMONSTRATION CONTROLS
    # ========================================================

    st.sidebar.subheader(
        "Demonstration Controls"
    )

    st.sidebar.caption(
        "Use these controls only for prototype testing. "
        "Activity is still inferred from sensor data."
    )

    # --------------------------------------------------------
    # Scenario
    # --------------------------------------------------------

    available_scenarios = list(
        scenarios
        if scenarios
        else get_available_scenarios()
    )

    if not available_scenarios:

        available_scenarios = [
            "NORMAL",
            "WALKING",
            "RUNNING",
            "FALL",
            "HIGH_RISK",
            "SOS",
        ]

    normalized_scenario = str(
        scenario
    ).upper()

    if normalized_scenario not in available_scenarios:

        normalized_scenario = available_scenarios[0]

    selected_scenario = st.sidebar.selectbox(
        "Demo Sensor Profile",
        available_scenarios,
        index=available_scenarios.index(
            normalized_scenario
        ),
        help=(
            "Select a deterministic sensor-data profile "
            "for prototype testing. This does not directly "
            "set the detected activity."
        ),
        key="demo_scenario",
    )

    # --------------------------------------------------------
    # Manual SOS
    # --------------------------------------------------------

    sos_requested = st.sidebar.button(
        "🚨 TEST SOS",
        use_container_width=True,
        type="primary",
        help=(
            "Trigger a manual SOS test for the prototype. "
            "This represents an explicit user emergency action."
        ),
    )

    if sos_requested:

        st.session_state[
            "manual_sos"
        ] = True

    manual_sos = bool(
        st.session_state.get(
            "manual_sos",
            False,
        )
    )

    if manual_sos:

        st.sidebar.error(
            "🚨 SOS TEST ACTIVE"
        )

        if st.sidebar.button(
            "Clear SOS",
            use_container_width=True,
        ):

            st.session_state[
                "manual_sos"
            ] = False

            st.rerun()

    st.sidebar.divider()

    # ========================================================
    # MODULE VISIBILITY
    # ========================================================

    st.sidebar.subheader(
        "Dashboard Modules"
    )

    show_sensors = st.sidebar.checkbox(
        "Sensor Monitoring",
        value=DASHBOARD_CONFIG.get(
            "show_sensor_status",
            True,
        ),
    )

    show_ai = st.sidebar.checkbox(
        "AI Analysis",
        value=DASHBOARD_CONFIG.get(
            "show_activity",
            True,
        ),
    )

    show_communication = st.sidebar.checkbox(
        "Communication",
        value=DASHBOARD_CONFIG.get(
            "show_communication",
            True,
        ),
    )

    show_location = st.sidebar.checkbox(
        "GPS Location",
        value=DASHBOARD_CONFIG.get(
            "show_location",
            True,
        ),
    )

    show_history = st.sidebar.checkbox(
        "Alert History",
        value=DASHBOARD_CONFIG.get(
            "show_alert_history",
            True,
        ),
    )

    show_charts = st.sidebar.checkbox(
        "Monitoring Charts",
        value=DASHBOARD_CONFIG.get(
            "show_monitoring_charts",
            True,
        ),
    )

    st.sidebar.divider()

    # ========================================================
    # ARCHITECTURE
    # ========================================================

    st.sidebar.subheader(
        "Architecture"
    )

    st.sidebar.caption(
        "Sensors → ESP32-S3 → AI / Sensor Fusion → "
        "EC200U → Cloud / Caregiver"
    )

    # ========================================================
    # RETURN RUNTIME UI STATE
    # ========================================================

    return {
        # Backwards compatibility.
        "scenario": selected_scenario,

        # Runtime state.
        "simulation": simulation_enabled,

        # Explicit emergency input.
        "manual_sos": manual_sos,

        # Dashboard visibility.
        "show_sensors": show_sensors,
        "show_ai": show_ai,
        "show_communication": show_communication,
        "show_location": show_location,
        "show_history": show_history,
        "show_charts": show_charts,
    }


# ============================================================
# STATUS BANNER
# ============================================================

def render_status_banner(
    status: str,
    risk_level: str,
    risk_score: float,
) -> None:
    """Render the current overall safety status."""

    status = str(
        status
    ).upper()

    risk_level = str(
        risk_level
    ).upper()

    try:

        score = float(
            risk_score
        )

    except (TypeError, ValueError):

        score = 0.0

    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )

    if status == "EMERGENCY":

        st.error(
            f"🚨 **EMERGENCY — {risk_level}**  \n"
            f"Safety risk score: **{score:.0f}/100**"
        )

    elif status == "WARNING":

        st.warning(
            f"⚠️ **WARNING — {risk_level}**  \n"
            f"Safety risk score: **{score:.0f}/100**"
        )

    else:

        st.success(
            f"🟢 **SYSTEM SAFE — {risk_level}**  \n"
            f"Safety risk score: **{score:.0f}/100**"
        )


# ============================================================
# SENSOR VALUE HELPERS
# ============================================================

def _number(
    data: Dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    """Safely retrieve a numeric sensor value."""

    try:

        return float(
            data.get(
                key,
                default,
            )
        )

    except (TypeError, ValueError):

        return default


# ============================================================
# SENSOR CARDS
# ============================================================

def render_sensor_cards(
    sensor_data: Dict[str, Any],
) -> None:
    """
    Render current sensor readings.

    Temperature distinction:

        temperature
            BME680 environmental temperature

        body_temperature
            MAX30208 body temperature
    """

    st.subheader(
        "Live Sensor Monitoring"
    )

    # ========================================================
    # READ VALUES
    # ========================================================

    heart_rate = _number(
        sensor_data,
        "heart_rate",
        0.0,
    )

    spo2 = _number(
        sensor_data,
        "spo2",
        0.0,
    )

    body_temperature = _number(
        sensor_data,
        "body_temperature",
        0.0,
    )

    environmental_temperature = _number(
        sensor_data,
        "temperature",
        0.0,
    )

    humidity = _number(
        sensor_data,
        "humidity",
        0.0,
    )

    pressure = _number(
        sensor_data,
        "pressure",
        0.0,
    )

    motion = _number(
        sensor_data,
        "motion_intensity",
        0.0,
    )

    orientation = _number(
        sensor_data,
        "orientation",
        0.0,
    )

    audio = _number(
        sensor_data,
        "audio_level",
        0.0,
    )

    # ========================================================
    # PHYSIOLOGICAL SENSORS
    # ========================================================

    st.markdown(
        "#### ❤️ Physiological Sensors"
    )

    cols = st.columns(3)

    with cols[0]:

        st.metric(
            "Heart Rate",
            f"{heart_rate:.0f} BPM",
        )

        st.caption(
            "MAX30102"
        )

    with cols[1]:

        st.metric(
            "SpO₂",
            f"{spo2:.1f}%",
        )

        st.caption(
            "MAX30102"
        )

    with cols[2]:

        st.metric(
            "Body Temperature",
            f"{body_temperature:.1f} °C",
        )

        st.caption(
            "MAX30208 • Body"
        )

    # ========================================================
    # ENVIRONMENTAL SENSORS
    # ========================================================

    st.markdown(
        "#### 🌡️ Environmental Sensors"
    )

    cols = st.columns(3)

    with cols[0]:

        st.metric(
            "Environmental Temperature",
            f"{environmental_temperature:.1f} °C",
        )

        st.caption(
            "BME680 • Environment"
        )

    with cols[1]:

        st.metric(
            "Humidity",
            f"{humidity:.1f}%",
        )

        st.caption(
            "BME680"
        )

    with cols[2]:

        st.metric(
            "Pressure",
            f"{pressure:.0f} hPa",
        )

        st.caption(
            "BME680"
        )

    # ========================================================
    # MOTION / AUDIO
    # ========================================================

    st.markdown(
        "#### 🧭 Motion & Audio"
    )

    cols = st.columns(3)

    with cols[0]:

        st.metric(
            "Motion",
            f"{motion:.2f}",
        )

        st.caption(
            "BNO055"
        )

    with cols[1]:

        st.metric(
            "Orientation",
            f"{orientation:.1f}°",
        )

        st.caption(
            "BNO055"
        )

    with cols[2]:

        st.metric(
            "Audio Level",
            f"{audio:.2f}",
        )

        st.caption(
            "INMP441"
        )


# ============================================================
# AI ANALYSIS
# ============================================================

def render_ai_analysis(
    activity_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
    risk_result: Dict[str, Any],
) -> None:
    """
    Render AI activity recognition, sensor fusion and risk data.

    Activity displayed here is an inference result. It is never
    presented as a manual activity selector.
    """

    st.subheader(
        "🤖 AI Safety Analysis"
    )

    activity = str(
        activity_result.get(
            "activity",
            "UNKNOWN",
        )
    ).upper()

    confidence = _number(
        activity_result,
        "confidence",
        0.0,
    )

    # Support both 0-1 and percentage-style confidence.
    if confidence > 1.0:
        confidence /= 100.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    description = activity_result.get(
        "description",
        "No activity description available.",
    )

    fusion_score = _number(
        fusion_result,
        "fusion_score",
        0.0,
    )

    fusion_confidence = _number(
        fusion_result,
        "confidence",
        0.0,
    )

    if fusion_confidence > 1.0:
        fusion_confidence /= 100.0

    fusion_condition = str(
        fusion_result.get(
            "condition",
            "NORMAL",
        )
    ).upper()

    risk_score = _number(
        risk_result,
        "risk_score",
        0.0,
    )

    risk_level = str(
        risk_result.get(
            "risk_level",
            "LOW",
        )
    ).upper()

    reason = risk_result.get(
        "reason",
        "No abnormal event detected.",
    )

    cols = st.columns(3)

    # ========================================================
    # ACTIVITY
    # ========================================================

    with cols[0]:

        st.markdown(
            "#### Detected Activity"
        )

        st.metric(
            "Activity",
            activity,
        )

        st.progress(
            confidence
        )

        st.caption(
            f"Confidence: {confidence * 100:.0f}%"
        )

    # ========================================================
    # SENSOR FUSION
    # ========================================================

    with cols[1]:

        st.markdown(
            "#### Sensor Fusion"
        )

        st.metric(
            "Fusion Score",
            f"{fusion_score:.0f}/100",
        )

        st.metric(
            "Condition",
            fusion_condition,
        )

        st.caption(
            f"Fusion confidence: "
            f"{fusion_confidence * 100:.0f}%"
        )

    # ========================================================
    # RISK ENGINE
    # ========================================================

    with cols[2]:

        st.markdown(
            "#### Risk Engine"
        )

        st.metric(
            "Risk Score",
            f"{risk_score:.0f}/100",
        )

        st.metric(
            "Risk Level",
            risk_level,
        )

    st.info(
        f"**Activity:** {description}  \n"
        f"**Risk decision:** {reason}"
    )


# ============================================================
# COMMUNICATION STATUS
# ============================================================

def render_communication_status(
    cellular_status: Dict[str, Any],
    cloud_status: Dict[str, Any],
) -> None:
    """Render cellular and cloud communication status."""

    st.subheader(
        "📡 Communication & Cloud"
    )

    cellular_status = cellular_status or {}
    cloud_status = cloud_status or {}

    cellular_connected = bool(
        cellular_status.get(
            "connected",
            False,
        )
    )

    cloud_connected = bool(
        cloud_status.get(
            "connected",
            False,
        )
    )

    cols = st.columns(2)

    # ========================================================
    # CELLULAR
    # ========================================================

    with cols[0]:

        st.markdown(
            "#### 📶 EC200U Cellular"
        )

        st.metric(
            "Connection",
            (
                "CONNECTED"
                if cellular_connected
                else "DISCONNECTED"
            ),
        )

        signal = _number(
            cellular_status,
            "signal_strength",
            0.0,
        )

        st.metric(
            "Signal Strength",
            f"{signal:.0f}%",
        )

        st.caption(
            cellular_status.get(
                "network",
                "Unknown network",
            )
        )

    # ========================================================
    # CLOUD
    # ========================================================

    with cols[1]:

        st.markdown(
            "#### ☁️ Cloud"
        )

        st.metric(
            "Connection",
            (
                "CONNECTED"
                if cloud_connected
                else "DISCONNECTED"
            ),
        )

        records = cloud_status.get(
            "records_uploaded",
            0,
        )

        st.metric(
            "Records Uploaded",
            records,
        )

        st.caption(
            cloud_status.get(
                "last_message",
                "No synchronization yet.",
            )
        )


# ============================================================
# GPS LOCATION
# ============================================================

def render_location(
    sensor_data: Dict[str, Any],
) -> None:
    """Render the current GPS position."""

    st.subheader(
        "📍 Live Location"
    )

    latitude = _number(
        sensor_data,
        "latitude",
        0.0,
    )

    longitude = _number(
        sensor_data,
        "longitude",
        0.0,
    )

    gps_connected = bool(
        sensor_data.get(
            "gps_connected",
            True,
        )
    )

    cols = st.columns(3)

    with cols[0]:

        st.metric(
            "Latitude",
            f"{latitude:.6f}",
        )

    with cols[1]:

        st.metric(
            "Longitude",
            f"{longitude:.6f}",
        )

    with cols[2]:

        st.metric(
            "GPS Status",
            (
                "ACTIVE"
                if gps_connected
                else "INACTIVE"
            ),
        )

    if (
        gps_connected
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):

        location_df = pd.DataFrame(
            {
                "latitude": [latitude],
                "longitude": [longitude],
            }
        )

        st.map(
            location_df,
            latitude="latitude",
            longitude="longitude",
            zoom=15,
        )

    else:

        st.warning(
            "GPS coordinates are currently unavailable."
        )


# ============================================================
# ALERT DISPLAY
# ============================================================

def render_alert(
    alert: Optional[Dict[str, Any]],
) -> None:
    """Render the current safety alert."""

    if not alert:

        st.success(
            "🟢 No active alerts. System operating normally."
        )

        return

    severity = str(
        alert.get(
            "severity",
            "INFO",
        )
    ).upper()

    title = alert.get(
        "title",
        "Safety Alert",
    )

    message = alert.get(
        "message",
        "",
    )

    if severity == "CRITICAL":

        st.error(
            f"🚨 **{title}**\n\n"
            f"{message}"
        )

        st.caption(
            "Emergency notification workflow activated."
        )

    elif severity in (
        "HIGH",
        "WARNING",
    ):

        st.warning(
            f"⚠️ **{title}**\n\n"
            f"{message}"
        )

    else:

        st.info(
            f"ℹ️ **{title}**\n\n"
            f"{message}"
        )


# ============================================================
# ALERT HISTORY
# ============================================================

def render_alert_history(
    alert_history: Sequence[Dict[str, Any]],
) -> None:
    """Render previous safety alerts and events."""

    st.subheader(
        "📜 Alert & Event History"
    )

    if not alert_history:

        st.info(
            "No safety events recorded yet."
        )

        return

    rows: List[Dict[str, Any]] = []

    for alert in alert_history:

        if not isinstance(
            alert,
            dict,
        ):
            continue

        rows.append(
            {
                "Time": alert.get(
                    "timestamp",
                    "",
                ),

                "Type": alert.get(
                    "alert_type",
                    "",
                ),

                "Severity": alert.get(
                    "severity",
                    "",
                ),

                "Activity": alert.get(
                    "activity",
                    "",
                ),

                "Risk": alert.get(
                    "risk_score",
                    0,
                ),

                "Status": alert.get(
                    "status",
                    "",
                ),
            }
        )

    if not rows:

        st.info(
            "No valid safety events available."
        )

        return

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CHARTS
# ============================================================

def render_charts(
    history: Sequence[Dict[str, Any]],
) -> None:
    """
    Render historical monitoring charts.

    Environmental temperature and body temperature remain
    separate measurements throughout the dashboard.
    """

    if not history:

        st.info(
            "Charts will appear when monitoring data is available."
        )

        return

    valid_history = [
        item
        for item in history
        if isinstance(item, dict)
    ]

    if not valid_history:

        st.info(
            "Charts will appear when monitoring data is available."
        )

        return

    timestamps = [
        item.get(
            "timestamp",
            "",
        )
        for item in valid_history
    ]

    heart_rates = [
        item.get(
            "heart_rate",
            0,
        )
        for item in valid_history
    ]

    spo2_values = [
        item.get(
            "spo2",
            0,
        )
        for item in valid_history
    ]

    environmental_temperatures = [
        item.get(
            "temperature",
            0,
        )
        for item in valid_history
    ]

    body_temperatures = [
        item.get(
            "body_temperature",
            0,
        )
        for item in valid_history
    ]

    motion_values = [
        item.get(
            "motion_intensity",
            0,
        )
        for item in valid_history
    ]

    risk_scores = [
        item.get(
            "risk_score",
            0,
        )
        for item in valid_history
    ]

    st.subheader(
        "📊 Live Monitoring Trends"
    )

    # ========================================================
    # HEART RATE / RISK
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            create_heart_rate_chart(
                timestamps,
                heart_rates,
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    with col2:

        st.plotly_chart(
            create_risk_chart(
                timestamps,
                risk_scores,
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # ========================================================
    # SPO2 / ENVIRONMENT
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            create_spo2_chart(
                timestamps,
                spo2_values,
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    with col2:

        st.plotly_chart(
            create_temperature_chart(
                timestamps,
                environmental_temperatures,
            ),
            use_container_width=True,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
        )

    # ========================================================
    # BODY TEMPERATURE
    # ========================================================

    st.plotly_chart(
        create_body_temperature_chart(
            timestamps,
            body_temperatures,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    # ========================================================
    # MOTION
    # ========================================================

    st.plotly_chart(
        create_motion_chart(
            timestamps,
            motion_values,
        ),
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

def render_system_overview(
    sensor_status: Optional[Dict[str, Any]] = None,
    cellular_status: Optional[Dict[str, Any]] = None,
    cloud_status: Optional[Dict[str, Any]] = None,
) -> None:
    """Render high-level system component status."""

    st.subheader(
        "⚙️ System Overview"
    )

    sensor_status = sensor_status or {}
    cellular_status = cellular_status or {}
    cloud_status = cloud_status or {}

    sensors_online = sensor_status.get(
        "online",
        0,
    )

    total_sensors = sensor_status.get(
        "total",
        6,
    )

    cellular = (
        "ONLINE"
        if cellular_status.get(
            "connected",
            False,
        )
        else "OFFLINE"
    )

    cloud = (
        "ONLINE"
        if cloud_status.get(
            "connected",
            False,
        )
        else "OFFLINE"
    )

    cols = st.columns(5)

    with cols[0]:

        st.metric(
            "Sensors",
            f"{sensors_online}/{total_sensors}",
        )

    with cols[1]:

        st.metric(
            "ESP32-S3",
            "ACTIVE",
        )

    with cols[2]:

        st.metric(
            "AI Engine",
            "READY",
        )

    with cols[3]:

        st.metric(
            "EC200U",
            cellular,
        )

    with cols[4]:

        st.metric(
            "Cloud",
            cloud,
        )


# ============================================================
# FOOTER
# ============================================================

def render_footer() -> None:
    """Render application footer."""

    st.divider()

    st.caption(
        f"{APP_NAME} • Intelligent AI-Based Safety Monitoring "
        f"Prototype • {APP_MODE}"
    )

    st.caption(
        "Real sensor, AI and communication hardware integration "
        "will be implemented in subsequent development phases."
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "configure_page",
    "render_header",
    "render_sidebar",
    "render_status_banner",
    "render_sensor_cards",
    "render_ai_analysis",
    "render_communication_status",
    "render_location",
    "render_alert",
    "render_alert_history",
    "render_charts",
    "render_system_overview",
    "render_footer",
]