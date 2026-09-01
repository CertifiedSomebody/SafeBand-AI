"""
SAFEBAND AI - Application Settings

Central configuration for the SAFEBAND AI prototype.

All values in this file are prototype/demo settings. Hardware,
cloud and AI parameters can be updated here when the real system
is integrated.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
LOGO_DIR = ASSETS_DIR / "logo"

DATA_DIR = BASE_DIR / "data"


# ============================================================
# APPLICATION INFORMATION
# ============================================================

APP_NAME = "SAFEBAND AI"

APP_TITLE = "SAFEBAND AI - Intelligent Safety Monitoring System"

APP_VERSION = "1.0.0"

APP_MODE = "PROTOTYPE"

APP_DESCRIPTION = (
    "AI-based wearable safety monitoring prototype "
    "using multi-sensor data fusion."
)


# ============================================================
# PROTOTYPE MODE
# ============================================================

SIMULATION_MODE = True

REAL_HARDWARE_ENABLED = False

AI_MODEL_ENABLED = False

CLOUD_SIMULATION_ENABLED = True

CELLULAR_SIMULATION_ENABLED = True


# ============================================================
# SENSOR CONFIGURATION
# ============================================================

SENSOR_CONFIG = {
    "MAX30102": {
        "name": "Heart Rate & SpO2",
        "type": "Physiological",
        "enabled": True,
        "simulation": True,
    },

    "BNO055": {
        "name": "Motion & Orientation",
        "type": "Motion",
        "enabled": True,
        "simulation": True,
    },

    "BME680": {
        "name": "Environmental Sensor",
        "type": "Environmental",
        "enabled": True,
        "simulation": True,
    },

    "INMP441": {
        "name": "MEMS Microphone",
        "type": "Audio",
        "enabled": True,
        "simulation": True,
    },

    "GPS": {
        "name": "GPS Location",
        "type": "Location",
        "enabled": True,
        "simulation": True,
    },
}


# ============================================================
# SENSOR DEFAULT VALUES
# ============================================================

DEFAULT_SENSOR_VALUES = {
    "heart_rate": 75.0,
    "spo2": 98.0,

    "acceleration_x": 0.0,
    "acceleration_y": 0.0,
    "acceleration_z": 1.0,

    "motion_intensity": 0.10,
    "orientation": 0.0,

    "temperature": 25.0,
    "humidity": 50.0,
    "pressure": 1013.0,

    "audio_level": 0.10,

    "latitude": 19.0760,
    "longitude": 72.8777,
}


# ============================================================
# SAFETY THRESHOLDS
# ============================================================

SAFETY_THRESHOLDS = {
    "heart_rate_high": 110,
    "heart_rate_critical": 130,

    "heart_rate_low": 50,

    "spo2_warning": 94,
    "spo2_critical": 90,

    "temperature_high": 45,
    "temperature_low": 5,

    "fall_acceleration": 2.5,
    "fall_orientation": 60,

    "high_motion": 1.2,
    "critical_motion": 2.0,
}


# ============================================================
# RISK LEVELS
# ============================================================

RISK_LEVELS = {
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

CELLULAR_CONFIG = {
    "module": "Quectel EC200U",
    "network_type": "LTE Cat-1",
    "simulation": CELLULAR_SIMULATION_ENABLED,
    "default_signal_strength": 82,
}


# ============================================================
# CLOUD
# ============================================================

CLOUD_CONFIG = {
    "server_name": "SAFEBAND AI Cloud",
    "simulation": CLOUD_SIMULATION_ENABLED,
    "sync_interval_seconds": 5,
}


# ============================================================
# GPS
# ============================================================

GPS_CONFIG = {
    "simulation": True,

    # Prototype demonstration location.
    "default_latitude": 19.0760,
    "default_longitude": 72.8777,

    "accuracy_meters": 5.0,
}


# ============================================================
# DASHBOARD
# ============================================================

DASHBOARD_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": "🛡️",
    "layout": "wide",

    "refresh_interval_seconds": 2,

    "show_simulation_badge": True,
    "show_sensor_status": True,
    "show_risk_score": True,
    "show_activity": True,
    "show_location": True,
    "show_alert_history": True,
}


# ============================================================
# DEMO SETTINGS
# ============================================================

DEMO_CONFIG = {
    "default_scenario": "NORMAL",

    "available_scenarios": [
        "NORMAL",
        "WALKING",
        "RUNNING",
        "FALL",
        "SOS",
        "HIGH_RISK",
    ],

    "auto_generate_data": True,
    "data_update_interval": 2,
}


# ============================================================
# ALERT SETTINGS
# ============================================================

ALERT_CONFIG = {
    "enable_alerts": True,

    "caregiver_notifications": True,

    "emergency_alert_on_fall": True,

    "emergency_alert_on_sos": True,

    "emergency_alert_risk_threshold": 80,
}


# ============================================================
# LOGGING
# ============================================================

LOG_CONFIG = {
    "enabled": True,
    "log_level": "INFO",
    "max_history": 100,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_project_info():
    """Return basic project information."""

    return {
        "name": APP_NAME,
        "title": APP_TITLE,
        "version": APP_VERSION,
        "mode": APP_MODE,
        "simulation": SIMULATION_MODE,
    }


def is_simulation_mode() -> bool:
    """Return whether the application is running in prototype mode."""

    return SIMULATION_MODE


def get_sensor_config(sensor_name: str):
    """Return configuration for a specific sensor."""

    return SENSOR_CONFIG.get(sensor_name)


def get_default_sensor_values():
    """Return a copy of the default simulated sensor values."""

    return DEFAULT_SENSOR_VALUES.copy()