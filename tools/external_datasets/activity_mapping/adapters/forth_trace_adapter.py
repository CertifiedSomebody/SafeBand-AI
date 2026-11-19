from __future__ import annotations

import csv
from pathlib import Path


ACTIVITY_LABELS = {
    1: ("stand", "STATIC"),
    2: ("sit", "STATIC"),
    3: ("sit and talk", "STATIC"),
    4: ("walk", "WALKING"),
    5: ("walk and talk", "WALKING"),
    6: ("climb stairs up/down", "WALKING"),
    7: ("climb stairs up/down and talk", "WALKING"),
    8: ("stand to sit", "TRANSITION"),
    9: ("sit to stand", "TRANSITION"),
    10: ("stand to sit and talk", "TRANSITION"),
    11: ("sit and talk to stand", "TRANSITION"),
    12: ("stand to walk", "TRANSITION"),
    13: ("walk to stand", "TRANSITION"),
    14: ("stand to climb stairs", "TRANSITION"),
    15: ("climb stairs to walk", "TRANSITION"),
    16: ("climb stairs and talk to walk and talk", "TRANSITION"),
}


def observed_labels(raw_root: Path) -> list[str]:
    labels: set[str] = set()
    for path in sorted(raw_root.glob("part*/part*dev1.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row in csv.reader(source):
                if len(row) != 12:
                    raise ValueError(f"{path}: expected 12 columns, found {len(row)}")
                try:
                    label = int(row[11])
                except ValueError as exc:
                    raise ValueError(f"{path}: invalid activity label {row[11]!r}") from exc
                if label not in ACTIVITY_LABELS:
                    raise ValueError(f"{path}: unknown activity label ID {label}")
                labels.add(str(label))
    return sorted(labels, key=int)


def compatibility(subject_count: int) -> dict[str, str]:
    return {
        "dataset_id": "forth_trace",
        "source_sensor": "Shimmer 9-DoF; five sensor placements; candidate adapter uses dev1 (left wrist)",
        "accelerometer_available": "yes",
        "gyroscope_available": "yes",
        "orientation_available": "no (raw ACC/GYRO/MAG)",
        "timestamp_available": "timestamp column exists; documented epoch/unit not found locally",
        "sampling_rate": "51.2 Hz (local README)",
        "accelerometer_units": "not established in local documentation; do not convert",
        "gyroscope_units": "not established in local documentation; do not convert",
        "coordinate_convention": "axis directions not established",
        "sensor_placement": "dev1 = left wrist (local README)",
        "subject_count": str(subject_count),
        "session_count": f"{subject_count} participant dev1 files; separate session IDs not documented",
        "activity_label_quality": "16 activity labels and transition definitions are documented in the local README.",
        "safe_band_compatibility": "REJECT",
        "compatibility_notes": "Semantic label mapping is recorded, but physical ACC/GYRO units and timestamp unit are not verified locally. Therefore no values are normalized or used in the model.",
        "decision": "REJECTED",
    }
