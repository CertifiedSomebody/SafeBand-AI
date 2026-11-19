from __future__ import annotations

from pathlib import Path

import pandas as pd


ACTIVITY_MAP = {
    "lying": "STATIC",
    "sitting": "STATIC",
    "standing": "STATIC",
    "walking": "WALKING",
    "nordic walking": "WALKING",
    "ascending stairs": "WALKING",
    "descending stairs": "WALKING",
    "running": "RUNNING",
    "rope jumping": "HIGH_MOTION",
    "transient activities": "UNKNOWN",
    "cycling": "UNMAPPED",
    "ironing": "UNMAPPED",
    "vacuum cleaning": "UNMAPPED",
}


def observed_labels(csv_path: Path) -> list[str]:
    labels: set[str] = set()
    for chunk in pd.read_csv(csv_path, usecols=["activityID"], chunksize=250_000):
        labels.update(chunk["activityID"].dropna().astype(str).str.strip().unique())
    return sorted(labels, key=str.casefold)


def compatibility(subject_count: int, labels: list[str]) -> dict[str, str]:
    return {
        "dataset_id": "pamap2",
        "source_sensor": "Hand ACC/GYRO columns in merged dataset2.csv",
        "accelerometer_available": "yes",
        "gyroscope_available": "yes",
        "orientation_available": "no (raw ACC/GYRO/MAG columns)",
        "timestamp_available": "no timestamp column in local merged file",
        "sampling_rate": "100 Hz described by local experiment documentation; per-row timestamps unavailable",
        "accelerometer_units": "column indicates +/-16g range; exact stored unit conversion not verified in local raw schema",
        "gyroscope_units": "not established in local raw schema; do not convert",
        "coordinate_convention": "hand sensor axes available; coordinate directions not established",
        "sensor_placement": "hand/wrist column names; exact device orientation not established",
        "subject_count": str(subject_count),
        "session_count": "subject sequences observed; original session boundaries not retained",
        "activity_label_quality": "string activity labels observed: " + "; ".join(labels),
        "safe_band_compatibility": "REJECT",
        "compatibility_notes": "No timestamp column and unresolved physical units prevent timestamp-based window construction and safe normalization. Activity-label mapping is retained for audit only.",
        "decision": "REJECTED",
    }
