"""
SAFEBAND AI - Dashboard UI Module

Main Streamlit user-interface components for the SAFEBAND AI
prototype.

This module is responsible for displaying:
- System status
- Sensor readings
- AI activity recognition
- Sensor-fusion results
- Risk assessment
- GPS location
- Cellular/cloud status
- Emergency alerts
- Alert history
- Demonstration controls
"""

import streamlit as st
import pandas as pd

from config.settings import (
    APP_NAME,
    APP_VERSION,
    APP_MODE,
    SIMULATION_MODE,
)

from dashboard.charts import (
    create_heart_rate_chart,
    create_spo2_chart,
    create_temperature_chart,
    create_risk_chart,
    create_motion_chart,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

def configure_page():
    """Configure the Streamlit application page."""

    st.set_page_config(
        page_title="SAFEBAND AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================
# HEADER
# ============================================================

def render_header():
    """Render the SAFEBAND AI application header."""

    col1, col2, col3 = st.columns([2.5, 4, 2.5])

    with col1:
        st.markdown(
            """
            <div style="
                display:flex;
                align-items:center;
                gap:12px;
                padding-top:8px;
            ">
                <div style="
                    font-size:42px;
                ">
                    🛡️
                </div>
                <div>
                    <div style="
                        font-size:28px;
                        font-weight:800;
                        color:#0F172A;
                    ">
                        SAFEBAND AI
                    </div>
                    <div style="
                        font-size:12px;
                        color:#64748B;
                    ">
                        Intelligent Safety Monitoring
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                text-align:center;
                padding-top:15px;
            ">
                <div style="
                    font-size:13px;
                    color:#64748B;
                ">
                    REAL-TIME SAFETY MONITORING PROTOTYPE
                </div>
                <div style="
                    font-size:12px;
                    color:#94A3B8;
                    margin-top:4px;
                ">
                    Multi-Sensor AI • Sensor Fusion • Emergency Detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="
                text-align:right;
                padding-top:12px;
            ">
                <span style="
                    background:#DCFCE7;
                    color:#166534;
                    padding:6px 12px;
                    border-radius:20px;
                    font-size:12px;
                    font-weight:700;
                ">
                    ● {APP_MODE}
                </span>
                <div style="
                    font-size:11px;
                    color:#94A3B8;
                    margin-top:6px;
                ">
                    Version {APP_VERSION}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
    scenario: str = "NORMAL",
    scenarios=None,
):
    """
    Render the prototype control sidebar.

    Returns
    -------
    dict
        Selected dashboard controls.
    """

    if scenarios is None:
        scenarios = [
            "NORMAL",
            "WALKING",
            "RUNNING",
            "FALL",
            "HIGH_RISK",
            "SOS",
        ]

    st.sidebar.markdown(
        """
        <div style="
            font-size:20px;
            font-weight:800;
            margin-bottom:5px;
        ">
            🛡️ SAFEBAND AI
        </div>
        <div style="
            font-size:12px;
            color:#64748B;
            margin-bottom:15px;
        ">
            Prototype Control Panel
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Demo Controls")

    selected_scenario = st.sidebar.selectbox(
        "Simulation Scenario",
        scenarios,
        index=(
            scenarios.index(scenario)
            if scenario in scenarios
            else 0
        ),
    )

    st.sidebar.caption(
        "Sensor values are simulated for this prototype."
    )

    st.sidebar.divider()

    st.sidebar.subheader("System Modules")

    show_sensors = st.sidebar.checkbox(
        "Sensor Monitoring",
        value=True,
    )

    show_ai = st.sidebar.checkbox(
        "AI Analysis",
        value=True,
    )

    show_communication = st.sidebar.checkbox(
        "Communication",
        value=True,
    )

    show_history = st.sidebar.checkbox(
        "Alert History",
        value=True,
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **Prototype Architecture**

        Sensors  
        ↓  
        ESP32-S3 Processing  
        ↓  
        AI + Sensor Fusion  
        ↓  
        EC200U Communication  
        ↓  
        Cloud / Caregiver
        """
    )

    return {
        "scenario": selected_scenario,
        "show_sensors": show_sensors,
        "show_ai": show_ai,
        "show_communication": show_communication,
        "show_history": show_history,
    }


# ============================================================
# STATUS BANNER
# ============================================================

def render_status_banner(
    status: str,
    risk_level: str,
    risk_score: float,
):
    """Render the main safety-status banner."""

    status = str(status).upper()
    risk_level = str(risk_level).upper()

    if status == "EMERGENCY":
        background = "#FEE2E2"
        border = "#DC2626"
        text = "#991B1B"
        icon = "🚨"

    elif status == "WARNING":
        background = "#FEF3C7"
        border = "#F59E0B"
        text = "#92400E"
        icon = "⚠️"

    else:
        background = "#DCFCE7"
        border = "#16A34A"
        text = "#166534"
        icon = "🟢"

    # Main banner
    st.markdown(
        f"""
<div style="
background:{background};
border:2px solid {border};
border-radius:14px;
padding:18px 22px;
margin-bottom:20px;
">
<div style="
display:flex;
justify-content:space-between;
align-items:center;
">

<div>
<div style="
color:{text};
font-size:13px;
font-weight:700;
letter-spacing:1px;
">
CURRENT SAFETY STATUS
</div>

<div style="
color:{text};
font-size:30px;
font-weight:800;
margin-top:4px;
">
{icon} {status}
</div>
</div>

<div>
<div style="
color:{text};
font-size:12px;
font-weight:700;
text-align:right;
">
RISK LEVEL
</div>

<div style="
color:{text};
font-size:20px;
font-weight:700;
text-align:right;
margin-top:2px;
">
{risk_level}
</div>

<div style="
color:{text};
font-size:13px;
text-align:right;
margin-top:2px;
">
Score: {risk_score:.0f}/100
</div>
</div>

</div>
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# SENSOR CARDS
# ============================================================

def render_sensor_cards(sensor_data):
    """Render live sensor readings."""

    st.subheader("Live Sensor Monitoring")

    heart_rate = sensor_data.get("heart_rate", 0)
    spo2 = sensor_data.get("spo2", 0)
    temperature = sensor_data.get("temperature", 0)
    humidity = sensor_data.get("humidity", 0)
    pressure = sensor_data.get("pressure", 0)
    motion = sensor_data.get("motion_intensity", 0)
    orientation = sensor_data.get("orientation", 0)
    audio = sensor_data.get("audio_level", 0)

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "❤️ Heart Rate",
            f"{heart_rate:.0f} BPM",
        )
        st.caption("MAX30102")

    with cols[1]:
        st.metric(
            "🫁 SpO₂",
            f"{spo2:.1f}%",
        )
        st.caption("MAX30102")

    with cols[2]:
        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C",
        )
        st.caption("BME680")

    with cols[3]:
        st.metric(
            "💧 Humidity",
            f"{humidity:.1f}%",
        )
        st.caption("BME680")

    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "🌪️ Pressure",
            f"{pressure:.0f} hPa",
        )
        st.caption("BME680")

    with cols[1]:
        st.metric(
            "🧭 Motion",
            f"{motion:.2f}",
        )
        st.caption("BNO055")

    with cols[2]:
        st.metric(
            "📐 Orientation",
            f"{orientation:.1f}°",
        )
        st.caption("BNO055")

    with cols[3]:
        st.metric(
            "🎙️ Audio Level",
            f"{audio:.2f}",
        )
        st.caption("INMP441")


# ============================================================
# AI ANALYSIS
# ============================================================

def render_ai_analysis(
    activity_result,
    fusion_result,
    risk_result,
):
    """Render AI, sensor-fusion and risk-analysis information."""

    st.subheader("🤖 AI Safety Analysis")

    activity = activity_result.get(
        "activity",
        "UNKNOWN",
    )

    confidence = activity_result.get(
        "confidence",
        0,
    )

    description = activity_result.get(
        "description",
        "No description available.",
    )

    fusion_score = fusion_result.get(
        "fusion_score",
        0,
    )

    fusion_confidence = fusion_result.get(
        "confidence",
        0,
    )

    condition = fusion_result.get(
        "condition",
        "NORMAL",
    )

    risk_score = risk_result.get(
        "risk_score",
        0,
    )

    risk_level = risk_result.get(
        "risk_level",
        "LOW",
    )

    reason = risk_result.get(
        "reason",
        "No abnormal event detected.",
    )

    cols = st.columns(3)

    with cols[0]:
        st.markdown("### Activity")

        st.metric(
            "Detected Activity",
            activity,
        )

        st.progress(
            min(1.0, max(0.0, float(confidence)))
        )

        st.caption(
            f"Confidence: {float(confidence) * 100:.0f}%"
        )

    with cols[1]:
        st.markdown("### Sensor Fusion")

        st.metric(
            "Fusion Score",
            f"{fusion_score:.0f}/100",
        )

        st.metric(
            "Condition",
            condition,
        )

        st.caption(
            f"Fusion confidence: "
            f"{float(fusion_confidence) * 100:.0f}%"
        )

    with cols[2]:
        st.markdown("### Risk Engine")

        st.metric(
            "Risk Score",
            f"{risk_score:.0f}/100",
        )

        st.metric(
            "Risk Level",
            risk_level,
        )

    st.info(
        f"**Analysis:** {description}  \n"
        f"**Decision:** {reason}"
    )


# ============================================================
# COMMUNICATION STATUS
# ============================================================

def render_communication_status(
    cellular_status,
    cloud_status,
):
    """Render cellular and cloud communication status."""

    st.subheader("📡 Communication & Cloud")

    cellular_connected = cellular_status.get(
        "connected",
        False,
    )

    cloud_connected = cloud_status.get(
        "connected",
        False,
    )

    cols = st.columns(2)

    with cols[0]:
        st.markdown("### 📶 EC200U Cellular")

        st.metric(
            "Connection",
            "CONNECTED"
            if cellular_connected
            else "DISCONNECTED",
        )

        st.metric(
            "Signal Strength",
            f"{cellular_status.get('signal_strength', 0)}%",
        )

        st.caption(
            cellular_status.get(
                "network",
                "Unknown network",
            )
        )

    with cols[1]:
        st.markdown("### ☁️ Cloud")

        st.metric(
            "Connection",
            "CONNECTED"
            if cloud_connected
            else "DISCONNECTED",
        )

        st.metric(
            "Records Uploaded",
            cloud_status.get(
                "records_uploaded",
                0,
            ),
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

def render_location(sensor_data):
    """Render GPS information."""

    st.subheader("📍 Live Location")

    latitude = sensor_data.get(
        "latitude",
        0.0,
    )

    longitude = sensor_data.get(
        "longitude",
        0.0,
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
            "ACTIVE",
        )

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


# ============================================================
# ALERT DISPLAY
# ============================================================

def render_alert(alert):
    """Render the current alert."""

    if not alert:
        st.success(
            "🟢 No active alerts. System operating normally."
        )
        return

    severity = str(
        alert.get("severity", "INFO")
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
            f"🚨 **{title}**\n\n{message}"
        )

        st.warning(
            "📱 Caregiver notification simulated."
        )

    elif severity in ("HIGH", "WARNING"):
        st.warning(
            f"⚠️ **{title}**\n\n{message}"
        )

    else:
        st.info(
            f"ℹ️ **{title}**\n\n{message}"
        )


# ============================================================
# ALERT HISTORY
# ============================================================

def render_alert_history(alert_history):
    """Render previous alerts and safety events."""

    st.subheader("📜 Alert & Event History")

    if not alert_history:
        st.info(
            "No safety events recorded yet."
        )
        return

    rows = []

    for alert in alert_history:
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

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# CHARTS
# ============================================================

def render_charts(history):
    """Render sensor-history charts."""

    if not history:
        st.info(
            "Charts will appear when monitoring data is available."
        )
        return

    timestamps = [
        item.get("timestamp", "")
        for item in history
    ]

    heart_rates = [
        item.get("heart_rate", 0)
        for item in history
    ]

    spo2_values = [
        item.get("spo2", 0)
        for item in history
    ]

    temperatures = [
        item.get("temperature", 0)
        for item in history
    ]

    motion_values = [
        item.get("motion_intensity", 0)
        for item in history
    ]

    risk_scores = [
        item.get("risk_score", 0)
        for item in history
    ]

    st.subheader("📊 Live Monitoring Trends")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_heart_rate_chart(
                timestamps,
                heart_rates,
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_risk_chart(
                timestamps,
                risk_scores,
            ),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            create_spo2_chart(
                timestamps,
                spo2_values,
            ),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            create_temperature_chart(
                timestamps,
                temperatures,
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        create_motion_chart(
            timestamps,
            motion_values,
        ),
        use_container_width=True,
    )


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

def render_system_overview(
    sensor_status=None,
    cellular_status=None,
    cloud_status=None,
):
    """Render high-level system component status."""

    st.subheader("⚙️ System Overview")

    sensor_status = sensor_status or {}
    cellular_status = cellular_status or {}
    cloud_status = cloud_status or {}

    sensors_online = sensor_status.get(
        "online",
        5,
    )

    total_sensors = sensor_status.get(
        "total",
        5,
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
            "ACTIVE",
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

def render_footer():
    """Render application footer."""

    st.divider()

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#94A3B8;
            font-size:11px;
            padding:8px;
        ">
            SAFEBAND AI • Intelligent AI-Based Safety Monitoring
            Prototype • Simulation Mode
            <br>
            Real sensor, AI and communication hardware integration
            will be implemented in the subsequent development phase.
        </div>
        """,
        unsafe_allow_html=True,
    )