"""
SAFEBAND AI - Dataset Schema

Canonical row-level schema for SafeBand training/validation data.
Public datasets will be adapted into this schema during dataset
research. The schema deliberately keeps raw sensor channels and
metadata separate from ML labels.
"""

from typing import Dict, List


RAW_SENSOR_COLUMNS: List[str] = [
    "timestamp",
    "subject_id",
    "session_id",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "orientation",
    "motion_intensity",
    "heart_rate",
    "spo2",
    "body_temperature",
    "temperature",
    "humidity",
    "pressure",
    "audio_level",
    "latitude",
    "longitude",
]

LABEL_COLUMNS: List[str] = [
    "activity_label",
]

# Minimum fields required for the first activity-recognition
# training task. Other sensors are optional because public datasets
# rarely contain the complete SafeBand sensor suite.
REQUIRED_ACTIVITY_COLUMNS: List[str] = [
    "timestamp",
    "subject_id",
    "session_id",
    "acceleration_x",
    "acceleration_y",
    "acceleration_z",
    "activity_label",
]

OPTIONAL_METADATA_COLUMNS: List[str] = [
    "scenario",
    "manual_sos",
    "sensor_source",
    "device_id",
]

ALL_COLUMNS: List[str] = (
    RAW_SENSOR_COLUMNS
    + LABEL_COLUMNS
    + OPTIONAL_METADATA_COLUMNS
)


# Units are documented here so dataset adapters do not silently mix
# incompatible measurements.
UNITS: Dict[str, str] = {
    "acceleration_x": "g or normalized equivalent",
    "acceleration_y": "g or normalized equivalent",
    "acceleration_z": "g or normalized equivalent",
    "gyroscope_x": "deg/s or sensor-native equivalent",
    "gyroscope_y": "deg/s or sensor-native equivalent",
    "gyroscope_z": "deg/s or sensor-native equivalent",
    "orientation": "degrees",
    "motion_intensity": "project-defined normalized value",
    "heart_rate": "bpm",
    "spo2": "%",
    "body_temperature": "degC",
    "temperature": "degC",
    "humidity": "%",
    "pressure": "hPa or sensor-native equivalent",
    "audio_level": "project-defined normalized value",
    "latitude": "degrees",
    "longitude": "degrees",
}


def validate_columns(columns: List[str]) -> List[str]:
    """Return required SafeBand columns missing from a dataset."""
    present = set(columns)
    return [
        column
        for column in REQUIRED_ACTIVITY_COLUMNS
        if column not in present
    ]


__all__ = [
    "RAW_SENSOR_COLUMNS",
    "LABEL_COLUMNS",
    "REQUIRED_ACTIVITY_COLUMNS",
    "OPTIONAL_METADATA_COLUMNS",
    "ALL_COLUMNS",
    "UNITS",
    "validate_columns",
]
