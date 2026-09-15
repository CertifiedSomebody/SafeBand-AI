"""SafeBand hardware recording schema and validation helpers.

Canonical long-form CSV columns:
subject_id,source_file,recording_type,label_id,label,sensor_type,timestamp,x,y,z

Supported sensor_type values are acc, gyro, mag, hr, env, audio_marker.
For the first hardware ML pass, acc is mandatory because the validated BITS-2
fall model is accelerometer-based. Other sensors are preserved for later fusion.
"""
from __future__ import annotations

REQUIRED_COLUMNS = ["subject_id", "source_file", "recording_type", "sensor_type", "x", "y", "z"]
OPTIONAL_COLUMNS = ["label_id", "label", "timestamp", "hr", "spo2", "temperature", "humidity", "pressure"]
ALLOWED_RECORDING_TYPES = {"fall", "nonfall", "unknown"}
ALLOWED_SENSOR_TYPES = {"acc", "gyro", "mag", "hr", "env", "audio_marker"}


def validate_columns(columns):
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    return missing


def validate_row(row):
    errors = []
    for c in REQUIRED_COLUMNS:
        if not str(row.get(c, "")).strip():
            errors.append(f"missing:{c}")
    if row.get("sensor_type") not in ALLOWED_SENSOR_TYPES:
        errors.append("invalid:sensor_type")
    if row.get("recording_type") not in ALLOWED_RECORDING_TYPES:
        errors.append("invalid:recording_type")
    if row.get("sensor_type") in {"acc", "gyro", "mag"}:
        for c in ("x", "y", "z"):
            try:
                float(row[c])
            except (TypeError, ValueError):
                errors.append(f"invalid:{c}")
    return errors
