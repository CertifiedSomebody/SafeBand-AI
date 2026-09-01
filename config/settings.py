"""
SAFEBAND AI - Application Settings

Central configuration for the SAFEBAND AI project.

This module contains application-wide configuration used by:

    Sensors
    AI / TinyML
    Sensor Fusion
    Risk Engine
    Communication
    Cloud
    Dashboard
    Simulation
    Logging

The configuration is intentionally centralized so individual
modules do not need to hard-code project behaviour.

IMPORTANT SENSOR TEMPERATURE DISTINCTION
----------------------------------------

    temperature
        BME680 environmental temperature

    body_temperature
        MAX30208 body temperature

This distinction must be preserved throughout the SAFEBAND
processing pipeline because environmental and physiological
temperature represent different data sources.
"""


from pathlib import Path
from typing import Any, Dict


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
LOGO_DIR = ASSETS_DIR / "logo"

DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"


# ============================================================
# APPLICATION INFORMATION
# ============================================================

APP_NAME = "SAFEBAND AI"

APP_TITLE = (
    "SAFEBAND AI - Intelligent Safety Monitoring System"
)

APP_VERSION = "1.0.0"

APP_MODE = "PROTOTYPE"

APP_DESCRIPTION = (
    "AI-based wearable safety monitoring system "
    "using multi-sensor data fusion."
)


# ============================================================
# RUNTIME MODES
# ============================================================

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

# Default startup mode.
SIMULATION_MODE = True

# Allows the dashboard to expose a runtime simulation switch.
SIMULATION_MODE_TOGGLE = True


# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

# Real hardware is not enabled by default during development.
REAL_HARDWARE_ENABLED = False


# ------------------------------------------------------------
# AI / TinyML
# ------------------------------------------------------------

# Future trained model switch.
AI_MODEL_ENABLED = False


# ------------------------------------------------------------
# Communication
# ------------------------------------------------------------

CLOUD_SIMULATION_ENABLED = True

CELLULAR_SIMULATION_ENABLED = True


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

SENSOR_CONFIG: Dict[str, Dict[str, Any]] = {

    # --------------------------------------------------------
    # MAX30102
    # --------------------------------------------------------

    "MAX30102": {
        "name": "Heart Rate & SpO2",
        "type": "Physiological",
        "interface": "I2C",
        "enabled": True,
        "simulation": True,
    },

    # --------------------------------------------------------
    # MAX30208
    # --------------------------------------------------------

    "MAX30208": {
        "name": "Body Temperature",
        "type": "Physiological",
        "interface": "I2C",
        "enabled": True,
        "simulation": True,
    },

    # --------------------------------------------------------
    # BNO055
    # --------------------------------------------------------

    "BNO055": {
        "name": "Motion & Orientation",
        "type": "Motion",
        "interface": "I2C",
        "enabled": True,
        "simulation": True,
    },

    # --------------------------------------------------------
    # BME680
    # --------------------------------------------------------

    "BME680": {
        "name": "Environmental Sensor",
        "type": "Environmental",
        "interface": "I2C",
        "enabled": True,
        "simulation": True,
    },

    # --------------------------------------------------------
    # INMP441
    # --------------------------------------------------------

    "INMP441": {
        "name": "MEMS Microphone",
        "type": "Audio",
        "interface": "I2S",
        "enabled": True,
        "simulation": True,
    },

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    "GPS": {
        "name": "GPS Location",
        "type": "Location",
        "interface": "UART",
        "enabled": True,
        "simulation": True,
    },
}


# ============================================================
# SENSOR DEFAULT VALUES
# ============================================================

DEFAULT_SENSOR_VALUES: Dict[str, float] = {

    # --------------------------------------------------------
    # MAX30102 - Physiological
    # --------------------------------------------------------

    "heart_rate": 75.0,
    "spo2": 98.0,

    # --------------------------------------------------------
    # MAX30208 - Body Temperature
    # --------------------------------------------------------

    "body_temperature": 36.7,

    # --------------------------------------------------------
    # BNO055 - Motion
    # --------------------------------------------------------

    "acceleration_x": 0.0,
    "acceleration_y": 0.0,
    "acceleration_z": 1.0,

    "motion_intensity": 0.10,
    "orientation": 0.0,

    # --------------------------------------------------------
    # BME680 - Environment
    # --------------------------------------------------------

    "temperature": 25.0,
    "humidity": 50.0,
    "pressure": 1013.0,

    # --------------------------------------------------------
    # INMP441 - Audio
    # --------------------------------------------------------

    "audio_level": 0.10,

    # --------------------------------------------------------
    # GPS
    # --------------------------------------------------------

    "latitude": 19.0760,
    "longitude": 72.8777,
}


# ============================================================
# SAFETY THRESHOLDS
# ============================================================

SAFETY_THRESHOLDS: Dict[str, float] = {

    # --------------------------------------------------------
    # MAX30102 - Heart Rate
    # --------------------------------------------------------

    "heart_rate_high": 110.0,
    "heart_rate_critical": 130.0,
    "heart_rate_low": 50.0,

    # --------------------------------------------------------
    # MAX30102 - SpO2
    # --------------------------------------------------------

    "spo2_warning": 94.0,
    "spo2_critical": 90.0,

    # --------------------------------------------------------
    # BME680 - Environmental Temperature
    # --------------------------------------------------------

    "temperature_high": 45.0,
    "temperature_low": 5.0,

    # --------------------------------------------------------
    # MAX30208 - Body Temperature
    # --------------------------------------------------------

    "body_temperature_high": 38.0,
    "body_temperature_low": 35.0,

    # --------------------------------------------------------
    # BNO055 - Fall / Motion
    # --------------------------------------------------------

    "fall_acceleration": 2.5,
    "fall_orientation": 60.0,

    "high_motion": 1.2,
    "critical_motion": 2.0,
}


# ============================================================
# RISK LEVELS
# ============================================================

RISK_LEVELS: Dict[str, Dict[str, Any]] = {

    "LOW": {
        "minimum": 0,
        "maximum": 29,
        "status": "SAFE",
    },

    "MODERATE": {
        "minimum": 30,
        "maximum": 59,
        "status": "WARNING",
    },

    "HIGH": {
        "minimum": 60,
        "maximum": 79,
        "status": "WARNING",
    },

    "CRITICAL": {
        "minimum": 80,
        "maximum": 100,
        "status": "EMERGENCY",
    },
}


# ============================================================
# ACTIVITY RECOGNITION
# ============================================================

SUPPORTED_ACTIVITIES = [
    "SITTING",
    "STANDING",
    "WALKING",
    "RUNNING",
    "FALL",
    "UNKNOWN",
]


# ============================================================
# CELLULAR / EC200U
# ============================================================

CELLULAR_CONFIG: Dict[str, Any] = {

    "module": "Quectel EC200U",

    "network_type": "LTE Cat-1",

    "simulation": CELLULAR_SIMULATION_ENABLED,

    "default_signal_strength": 82,

    "enabled": True,
}


# ============================================================
# CLOUD
# ============================================================

CLOUD_CONFIG: Dict[str, Any] = {

    "server_name": "SAFEBAND AI Cloud",

    "simulation": CLOUD_SIMULATION_ENABLED,

    "sync_interval_seconds": 5,

    # Streamlit can later host the dashboard/backend-facing
    # prototype without changing this configuration structure.
    "provider": "STREAMLIT",
}


# ============================================================
# GPS
# ============================================================

GPS_CONFIG: Dict[str, Any] = {

    "simulation": SIMULATION_MODE,

    "default_latitude": 19.0760,

    "default_longitude": 72.8777,

    "accuracy_meters": 5.0,
}


# ============================================================
# DASHBOARD
# ============================================================

DASHBOARD_CONFIG: Dict[str, Any] = {

    "page_title": APP_TITLE,

    "page_icon": "🛡️",

    "layout": "wide",

    "refresh_interval_seconds": 2,

    # --------------------------------------------------------
    # Dashboard sections
    # --------------------------------------------------------

    "show_simulation_badge": True,

    "show_sensor_status": True,

    "show_risk_score": True,

    "show_activity": True,

    "show_location": True,

    "show_alert_history": True,

    "show_monitoring_charts": True,

    "show_communication": True,

    "show_cloud_status": True,

    # --------------------------------------------------------
    # Runtime controls
    # --------------------------------------------------------

    "show_simulation_toggle": SIMULATION_MODE_TOGGLE,

    "show_scenario_selector": True,

    "show_sos_control": True,
}


# ============================================================
# DEMONSTRATION / SIMULATION
# ============================================================

DEMO_CONFIG: Dict[str, Any] = {

    "default_scenario": "NORMAL",

    "available_scenarios": [
        "NORMAL",
        "WALKING",
        "RUNNING",
        "FALL",
        "HIGH_RISK",
        "SOS",
    ],

    "auto_generate_data": True,

    "data_update_interval": 2,
}


# ============================================================
# ALERT SETTINGS
# ============================================================

ALERT_CONFIG: Dict[str, Any] = {

    "enable_alerts": True,

    "caregiver_notifications": True,

    "emergency_alert_on_fall": True,

    "emergency_alert_on_sos": True,

    "emergency_alert_risk_threshold": 80,

    "max_history": 100,
}


# ============================================================
# LOGGING
# ============================================================

LOG_CONFIG: Dict[str, Any] = {

    "enabled": True,

    "log_level": "INFO",

    "max_history": 100,
}


# ============================================================
# RUNTIME MODE HELPERS
# ============================================================

def get_simulation_mode() -> bool:
    """Return the current application simulation mode."""

    return bool(
        SIMULATION_MODE
    )


def set_simulation_mode(
    enabled: bool,
) -> None:
    """
    Change simulation mode for the current application process.

    This does NOT modify this source file.

    Dashboard controls can use this function to switch between
    simulation and hardware-ready runtime behaviour.
    """

    global SIMULATION_MODE

    SIMULATION_MODE = bool(
        enabled
    )


def is_simulation_mode() -> bool:
    """Return whether the application is currently simulated."""

    return get_simulation_mode()


def is_hardware_enabled() -> bool:
    """
    Return whether real hardware operation is allowed.

    Hardware operation requires:

        REAL_HARDWARE_ENABLED = True
        SIMULATION_MODE = False
    """

    return bool(
        REAL_HARDWARE_ENABLED
        and not SIMULATION_MODE
    )


def is_ai_model_enabled() -> bool:
    """Return whether the trained AI/TinyML model is enabled."""

    return bool(
        AI_MODEL_ENABLED
    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

def get_project_info() -> Dict[str, Any]:
    """Return basic SAFEBAND AI project information."""

    return {
        "name": APP_NAME,
        "title": APP_TITLE,
        "version": APP_VERSION,
        "mode": APP_MODE,
        "simulation": SIMULATION_MODE,
        "hardware_enabled": is_hardware_enabled(),
        "ai_model_enabled": AI_MODEL_ENABLED,
    }


# ============================================================
# SENSOR HELPERS
# ============================================================

def get_sensor_config(
    sensor_name: str,
) -> Dict[str, Any]:
    """
    Return configuration for a specific sensor.

    Returns an empty dictionary when the sensor is unknown.
    """

    if not isinstance(
        sensor_name,
        str,
    ):
        return {}

    config = SENSOR_CONFIG.get(
        sensor_name.upper().strip()
    )

    if config is None:
        return {}

    return config.copy()


def is_sensor_enabled(
    sensor_name: str,
) -> bool:
    """Return whether a sensor is enabled."""

    config = get_sensor_config(
        sensor_name
    )

    return bool(
        config.get(
            "enabled",
            False,
        )
    )


def get_enabled_sensors() -> Dict[str, Dict[str, Any]]:
    """Return configurations for all enabled sensors."""

    return {
        name: config.copy()
        for name, config in SENSOR_CONFIG.items()
        if config.get(
            "enabled",
            False,
        )
    }


def get_enabled_sensor_names() -> list:
    """Return names of all enabled sensors."""

    return list(
        get_enabled_sensors().keys()
    )


def get_sensor_count() -> int:
    """Return the number of enabled sensors."""

    return len(
        get_enabled_sensors()
    )


# ============================================================
# SENSOR DEFAULT VALUES
# ============================================================

def get_default_sensor_values() -> Dict[str, float]:
    """
    Return an independent copy of the default sensor values.

    This prevents callers from modifying global defaults.
    """

    return DEFAULT_SENSOR_VALUES.copy()


# ============================================================
# SAFETY THRESHOLD HELPERS
# ============================================================

def get_safety_threshold(
    name: str,
    default: float = 0.0,
) -> float:
    """Return a configured safety threshold."""

    try:
        return float(
            SAFETY_THRESHOLDS.get(
                name,
                default,
            )
        )

    except (TypeError, ValueError):
        return float(
            default
        )


# ============================================================
# RISK HELPERS
# ============================================================

def get_risk_level(
    score: float,
) -> str:
    """
    Convert a numeric risk score into a configured risk level.

    Score range:

        0-29   LOW
        30-59  MODERATE
        60-79  HIGH
        80-100 CRITICAL
    """

    try:
        numeric_score = float(
            score
        )

    except (TypeError, ValueError):
        numeric_score = 0.0

    numeric_score = max(
        0.0,
        min(
            100.0,
            numeric_score,
        ),
    )

    for level, config in RISK_LEVELS.items():

        if (
            config["minimum"]
            <= numeric_score
            <= config["maximum"]
        ):
            return level

    return "LOW"


def get_risk_status(
    risk_level: str,
) -> str:
    """Return the status associated with a risk level."""

    config = RISK_LEVELS.get(
        str(
            risk_level
        ).upper().strip()
    )

    if config is None:
        return "SAFE"

    return str(
        config.get(
            "status",
            "SAFE",
        )
    )


# ============================================================
# DEMO HELPERS
# ============================================================

def get_available_scenarios() -> list:
    """
    Return scenarios available to the demonstration system.
    """

    return list(
        DEMO_CONFIG.get(
            "available_scenarios",
            [],
        )
    )


def get_default_scenario() -> str:
    """
    Return the configured default demonstration scenario.
    """

    return str(
        DEMO_CONFIG.get(
            "default_scenario",
            "NORMAL",
        )
    ).upper()


# ============================================================
# PATH HELPERS
# ============================================================

def ensure_project_directories() -> None:
    """
    Create required runtime directories if they do not exist.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    # Paths
    "BASE_DIR",
    "ASSETS_DIR",
    "LOGO_DIR",
    "DATA_DIR",
    "LOG_DIR",

    # Application
    "APP_NAME",
    "APP_TITLE",
    "APP_VERSION",
    "APP_MODE",
    "APP_DESCRIPTION",

    # Runtime
    "SIMULATION_MODE",
    "SIMULATION_MODE_TOGGLE",
    "REAL_HARDWARE_ENABLED",
    "AI_MODEL_ENABLED",

    # Sensors
    "SENSOR_CONFIG",
    "DEFAULT_SENSOR_VALUES",
    "SUPPORTED_ACTIVITIES",

    # Safety
    "SAFETY_THRESHOLDS",
    "RISK_LEVELS",

    # Communication
    "CELLULAR_CONFIG",
    "CLOUD_CONFIG",
    "GPS_CONFIG",

    # Dashboard / Demo
    "DASHBOARD_CONFIG",
    "DEMO_CONFIG",
    "ALERT_CONFIG",
    "LOG_CONFIG",

    # Runtime helpers
    "get_simulation_mode",
    "set_simulation_mode",
    "is_simulation_mode",
    "is_hardware_enabled",
    "is_ai_model_enabled",

    # Project
    "get_project_info",

    # Sensor helpers
    "get_sensor_config",
    "is_sensor_enabled",
    "get_enabled_sensors",
    "get_enabled_sensor_names",
    "get_sensor_count",
    "get_default_sensor_values",

    # Safety helpers
    "get_safety_threshold",
    "get_risk_level",
    "get_risk_status",

    # Demo helpers
    "get_available_scenarios",
    "get_default_scenario",

    # Paths
    "ensure_project_directories",
]