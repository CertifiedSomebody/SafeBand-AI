"""
SAFEBAND AI - Helper Utilities

Shared utility functions used across the SAFEBAND AI system.

This module intentionally contains only generic helpers:
    - Safe type conversion
    - Numeric operations
    - Sensor-data utilities
    - Risk utilities
    - Time formatting
    - History management
    - Dashboard formatting
    - Basic data validation

Sensor-specific logic belongs in the corresponding sensor modules.
AI/TinyML logic belongs in the AI layer.
"""


from datetime import datetime
from typing import Any, Dict, Iterable, Optional
import math


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to float."""

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to integer."""

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Restrict a numeric value to a specified range.

    Raises
    ------
    ValueError
        If minimum is greater than maximum.
    """

    if minimum > maximum:
        raise ValueError(
            "Minimum value cannot be greater than maximum."
        )

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# SENSOR HELPERS
# ============================================================

def calculate_acceleration_magnitude(
    acceleration_x: float,
    acceleration_y: float,
    acceleration_z: float,
) -> float:
    """
    Calculate resultant acceleration magnitude.

    Formula:

        A = sqrt(Ax² + Ay² + Az²)
    """

    ax = safe_float(
        acceleration_x
    )

    ay = safe_float(
        acceleration_y
    )

    az = safe_float(
        acceleration_z
    )

    magnitude = math.sqrt(
        ax ** 2
        + ay ** 2
        + az ** 2
    )

    return round(
        magnitude,
        3,
    )


def calculate_acceleration_from_data(
    data: Dict[str, Any],
) -> float:
    """
    Calculate acceleration magnitude directly from sensor data.
    """

    if not isinstance(
        data,
        dict,
    ):
        return 0.0

    return calculate_acceleration_magnitude(
        data.get(
            "acceleration_x",
            0.0,
        ),
        data.get(
            "acceleration_y",
            0.0,
        ),
        data.get(
            "acceleration_z",
            0.0,
        ),
    )


def normalize_sensor_value(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Normalize a sensor value to the range 0.0-1.0.
    """

    if maximum == minimum:
        return 0.0

    normalized = (
        (
            safe_float(value)
            - minimum
        )
        / (
            maximum
            - minimum
        )
    )

    return round(
        clamp(
            normalized,
            0.0,
            1.0,
        ),
        3,
    )


# ============================================================
# RISK HELPERS
# ============================================================

def risk_level_from_score(
    score: float,
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
        100.0,
    )

    if score >= 80:
        return "CRITICAL"

    if score >= 60:
        return "HIGH"

    if score >= 30:
        return "MODERATE"

    return "LOW"


def status_from_risk_level(
    risk_level: str,
) -> str:
    """Convert a risk level into a system status."""

    level = str(
        risk_level
    ).upper().strip()

    if level == "CRITICAL":
        return "EMERGENCY"

    if level in (
        "HIGH",
        "MODERATE",
    ):
        return "WARNING"

    return "SAFE"


def is_emergency(
    risk_score: float,
    activity: str = "UNKNOWN",
    manual_sos: bool = False,
) -> bool:
    """
    Determine whether the current state should be treated
    as an emergency.

    Emergency conditions:
        - Risk score >= 80
        - FALL activity
        - SOS activity
        - Manual SOS flag
    """

    normalized_activity = str(
        activity
    ).upper().strip()

    return (
        safe_float(risk_score) >= 80
        or normalized_activity in (
            "FALL",
            "SOS",
        )
        or bool(manual_sos)
    )


# ============================================================
# TIME HELPERS
# ============================================================

def current_timestamp() -> str:
    """Return the current timestamp in display format."""

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def current_time() -> str:
    """Return the current local time."""

    return datetime.now().strftime(
        "%H:%M:%S"
    )


def timestamp_iso() -> str:
    """Return the current timestamp in ISO 8601 format."""

    return datetime.now().isoformat(
        timespec="seconds"
    )


# ============================================================
# DATA HELPERS
# ============================================================

def merge_sensor_data(
    *sensor_sources: Optional[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:
    """
    Merge multiple sensor dictionaries into one dictionary.

    Later dictionaries override duplicate keys.

    This is useful when combining:

        MAX30102
        MAX30208
        BNO055
        BME680
        INMP441
        GPS
    """

    merged: Dict[str, Any] = {}

    for source in sensor_sources:

        if isinstance(
            source,
            dict,
        ):
            merged.update(
                source
            )

    return merged


def get_value(
    data: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safely retrieve a value from a dictionary."""

    if not isinstance(
        data,
        dict,
    ):
        return default

    return data.get(
        key,
        default,
    )


def copy_sensor_data(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a shallow copy of sensor data."""

    if not isinstance(
        data,
        dict,
    ):
        return {}

    return dict(
        data
    )


def add_sensor_metadata(
    data: Dict[str, Any],
    source: str,
) -> Dict[str, Any]:
    """
    Add non-destructive metadata identifying the data source.

    The original dictionary is not modified.
    """

    result = copy_sensor_data(
        data
    )

    result.setdefault(
        "sensor_source",
        source,
    )

    return result


# ============================================================
# HISTORY HELPERS
# ============================================================

def append_history(
    history: list,
    record: Dict[str, Any],
    max_length: int = 100,
) -> list:
    """
    Add a record to history while maintaining a maximum size.
    """

    if max_length <= 0:
        return history

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
    default: Any = None,
) -> Any:
    """Return the latest item from an iterable."""

    items = list(
        items
    )

    if not items:
        return default

    return items[-1]


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_percentage(
    value: float,
    decimals: int = 1,
) -> str:
    """Format a numeric value as a percentage."""

    return (
        f"{safe_float(value):.{decimals}f}%"
    )


def format_risk_score(
    score: float,
) -> str:
    """Format a risk score for dashboard display."""

    return (
        f"{clamp(safe_float(score), 0, 100):.0f}/100"
    )


def format_coordinates(
    latitude: float,
    longitude: float,
) -> str:
    """Format GPS coordinates for display."""

    return (
        f"{safe_float(latitude):.6f}, "
        f"{safe_float(longitude):.6f}"
    )


def format_temperature(
    temperature: Any,
    decimals: int = 1,
) -> str:
    """
    Format a temperature value for display.

    Returns '--' when no valid value is available.
    """

    try:
        value = float(
            temperature
        )

    except (
        TypeError,
        ValueError,
    ):
        return "--"

    return (
        f"{value:.{decimals}f} °C"
    )


# ============================================================
# VALIDATION HELPERS
# ============================================================

CORE_SENSOR_FIELDS = (
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
)


def validate_sensor_data(
    data: Dict[str, Any],
) -> bool:
    """
    Perform basic validation on a SAFEBAND sensor dataset.

    The legacy environmental `temperature` field remains a
    required core field because it belongs to the BME680.

    MAX30208 body temperature is intentionally separate and
    optional at this generic validation layer.
    """

    if not isinstance(
        data,
        dict,
    ):
        return False

    return all(
        field in data
        for field in CORE_SENSOR_FIELDS
    )


def validate_body_temperature(
    temperature: Any,
) -> bool:
    """
    Validate a MAX30208 body-temperature value.

    This checks whether the value is within the broad valid
    sensor range used by the prototype.
    """

    try:
        value = float(
            temperature
        )

    except (
        TypeError,
        ValueError,
    ):
        return False

    return (
        25.0
        <= value
        <= 45.0
    )


# ============================================================
# DEMO HELPERS
# ============================================================

SCENARIO_LABELS = {
    "NORMAL": "Normal Monitoring",
    "WALKING": "Walking",
    "RUNNING": "Running",
    "FALL": "Fall Detection",
    "HIGH_RISK": "High Risk",
    "SOS": "Manual SOS",
}


def scenario_label(
    scenario: str,
) -> str:
    """
    Convert an internal scenario name into a readable
    dashboard label.
    """

    normalized = str(
        scenario
    ).upper().strip()

    return SCENARIO_LABELS.get(
        normalized,
        normalized.replace(
            "_",
            " ",
        ).title(),
    )


def is_known_scenario(
    scenario: str,
) -> bool:
    """Return whether a scenario is supported by the prototype."""

    return (
        str(
            scenario
        ).upper().strip()
        in SCENARIO_LABELS
    )


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "safe_float",
    "safe_int",
    "clamp",
    "calculate_acceleration_magnitude",
    "calculate_acceleration_from_data",
    "normalize_sensor_value",
    "risk_level_from_score",
    "status_from_risk_level",
    "is_emergency",
    "current_timestamp",
    "current_time",
    "timestamp_iso",
    "merge_sensor_data",
    "get_value",
    "copy_sensor_data",
    "add_sensor_metadata",
    "append_history",
    "get_latest",
    "format_percentage",
    "format_risk_score",
    "format_coordinates",
    "format_temperature",
    "validate_sensor_data",
    "validate_body_temperature",
    "scenario_label",
    "is_known_scenario",
]