"""
SAFEBAND AI - Helper Utilities

Common utility functions shared across the SAFEBAND AI prototype.
"""

from datetime import datetime
from typing import Any, Dict, Iterable, Optional
import math


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0
) -> float:
    """
    Safely convert a value to float.

    Returns the default value if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(
    value: Any,
    default: int = 0
) -> int:
    """
    Safely convert a value to integer.

    Returns the default value if conversion fails.
    """

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(
    value: float,
    minimum: float,
    maximum: float
) -> float:
    """
    Restrict a numeric value to a specified range.
    """

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# SENSOR HELPERS
# ============================================================

def calculate_acceleration_magnitude(
    acceleration_x: float,
    acceleration_y: float,
    acceleration_z: float
) -> float:
    """
    Calculate resultant acceleration magnitude.

    Formula:
        A = sqrt(Ax² + Ay² + Az²)
    """

    magnitude = math.sqrt(
        acceleration_x ** 2
        + acceleration_y ** 2
        + acceleration_z ** 2
    )

    return round(
        magnitude,
        3
    )


def normalize_sensor_value(
    value: float,
    minimum: float,
    maximum: float
) -> float:
    """
    Normalize a sensor value to the range 0.0-1.0.
    """

    if maximum == minimum:
        return 0.0

    normalized = (
        (value - minimum)
        / (maximum - minimum)
    )

    return round(
        clamp(
            normalized,
            0.0,
            1.0
        ),
        3
    )


# ============================================================
# RISK HELPERS
# ============================================================

def risk_level_from_score(
    score: float
) -> str:
    """
    Convert a 0-100 risk score into a risk level.

    0-29   : LOW
    30-59  : MODERATE
    60-79  : HIGH
    80-100 : CRITICAL
    """

    score = clamp(
        safe_float(score),
        0.0,
        100.0
    )

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MODERATE"

    return "LOW"


def status_from_risk_level(
    risk_level: str
) -> str:
    """
    Convert a risk level into the corresponding system status.
    """

    level = str(
        risk_level
    ).upper()

    if level == "CRITICAL":
        return "EMERGENCY"

    if level in (
        "HIGH",
        "MODERATE"
    ):
        return "WARNING"

    return "SAFE"


def is_emergency(
    risk_score: float,
    activity: str = "UNKNOWN"
) -> bool:
    """
    Determine whether the current state should be treated
    as an emergency.
    """

    activity = str(
        activity
    ).upper()

    return (
        risk_score >= 80
        or activity == "FALL"
        or activity == "SOS"
    )


# ============================================================
# TIME HELPERS
# ============================================================

def current_timestamp() -> str:
    """
    Return the current timestamp in display-friendly format.
    """

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def current_time() -> str:
    """Return the current time."""

    return datetime.now().strftime(
        "%H:%M:%S"
    )


# ============================================================
# DATA HELPERS
# ============================================================

def merge_sensor_data(
    *sensor_sources: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge multiple sensor dictionaries into one dictionary.

    Later dictionaries override duplicate keys.
    """

    merged: Dict[str, Any] = {}

    for source in sensor_sources:
        if source:
            merged.update(source)

    return merged


def get_value(
    data: Dict[str, Any],
    key: str,
    default: Any = None
) -> Any:
    """
    Safely retrieve a value from a dictionary.
    """

    if not isinstance(data, dict):
        return default

    return data.get(
        key,
        default
    )


def copy_sensor_data(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a shallow copy of sensor data.

    Useful when processing a reading without modifying
    the original data.
    """

    return dict(data)


# ============================================================
# HISTORY HELPERS
# ============================================================

def append_history(
    history: list,
    record: Dict[str, Any],
    max_length: int = 100
) -> list:
    """
    Add a record to history while maintaining a maximum size.
    """

    history.append(
        record
    )

    if len(history) > max_length:
        del history[
            :-max_length
        ]

    return history


def get_latest(
    items: Iterable[Any],
    default: Any = None
) -> Any:
    """
    Return the latest item from an iterable.
    """

    items = list(items)

    if not items:
        return default

    return items[-1]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_percentage(
    value: float,
    decimals: int = 1
) -> str:
    """Format a numeric value as a percentage."""

    return f"{safe_float(value):.{decimals}f}%"


def format_risk_score(
    score: float
) -> str:
    """Format a risk score for dashboard display."""

    return f"{clamp(safe_float(score), 0, 100):.0f}/100"


def format_coordinates(
    latitude: float,
    longitude: float
) -> str:
    """Format GPS coordinates for display."""

    return (
        f"{safe_float(latitude):.6f}, "
        f"{safe_float(longitude):.6f}"
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_sensor_data(
    data: Dict[str, Any]
) -> bool:
    """
    Perform basic validation on a SAFEBAND sensor dataset.

    Returns True when the required core values are present.
    """

    required_fields = [
        "heart_rate",
        "spo2",
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "motion_intensity",
        "orientation",
        "temperature",
        "humidity",
        "pressure",
        "audio_level",
        "latitude",
        "longitude",
    ]

    if not isinstance(data, dict):
        return False

    return all(
        field in data
        for field in required_fields
    )


# ============================================================
# DEMO HELPERS
# ============================================================

def scenario_label(
    scenario: str
) -> str:
    """
    Convert an internal scenario name into a readable
    dashboard label.
    """

    labels = {
        "NORMAL": "Normal Monitoring",
        "WALKING": "Walking",
        "RUNNING": "Running",
        "FALL": "Fall Detection",
        "HIGH_RISK": "High Risk",
        "SOS": "Manual SOS",
    }

    return labels.get(
        str(scenario).upper(),
        str(scenario).replace(
            "_",
            " "
        ).title()
    )