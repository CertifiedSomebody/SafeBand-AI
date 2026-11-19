from __future__ import annotations

import re
import zipfile
from pathlib import Path


ADL_LABELS = {
    1: "walking_slowly",
    2: "walking_quickly",
    3: "jogging",
    4: "jumping",
    5: "climbing_up_slowly",
    6: "climbing_down_slowly",
    7: "climbing_up_normally",
    8: "climbing_down_normally",
    9: "slowly_sitting_on_chair",
    10: "rapidly_sitting_on_chair",
    11: "sitting_getting_up",
    12: "swinging_hands",
    13: "lying_on_bed",
    14: "lying_back_getting_up_slowly",
    15: "lying_back_getting_up_normally",
    16: "sideways_to_back_lying",
}
FALL_LABELS = {
    1: "forward_fall_knee",
    2: "right_fall",
    3: "left_fall",
    4: "forward_fall",
    5: "bed_to_ground_fall",
    6: "forward_fall_hand_support",
    7: "backward_fall_seated",
    8: "grabbing_while_falling",
}


def observed_labels(archive_path: Path) -> list[str]:
    labels: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            path = Path(member)
            if not path.suffix.lower() == ".csv":
                continue
            lower = member.lower()
            adl = re.search(r"adl(\d+)\.csv$", path.name, re.I)
            fall = re.search(r"fall(\d+)\.csv$", path.name, re.I)
            if "/adl/" in lower and adl:
                label_id = int(adl.group(1))
                labels.add(ADL_LABELS.get(label_id, f"adl_{label_id}"))
            elif "/fall/" in lower and fall:
                label_id = int(fall.group(1))
                labels.add(FALL_LABELS.get(label_id, f"fall_{label_id}"))
    return sorted(labels)


def compatibility() -> dict[str, str]:
    return {
        "dataset_id": "bits2",
        "source_sensor": "Raw CSV rows identify sensor tokens (acc/acg/gyro/mgm/hrt); physical units and synchronized clock semantics are not documented in the repository evidence reviewed.",
        "accelerometer_available": "yes",
        "gyroscope_available": "yes (sensor tokens exist; unit semantics unresolved)",
        "orientation_available": "unknown",
        "timestamp_available": "raw timestamp field exists; epoch/time unit and cross-sensor alignment unresolved",
        "sampling_rate": "not established",
        "accelerometer_units": "not established; do not convert",
        "gyroscope_units": "not established; do not convert",
        "coordinate_convention": "not established",
        "sensor_placement": "not established",
        "subject_count": "41 user IDs in the local archive layout",
        "session_count": "984 recording CSVs; subject/recording boundaries by path",
        "activity_label_quality": "ADL and staged-fall file labels are encoded by filename; source dictionaries exist in local ingestion code.",
        "safe_band_compatibility": "REJECT",
        "compatibility_notes": "Not normalized or included in model training because acceleration/angular units, sample rate, and synchronized stream semantics cannot be verified from local repository evidence. Raw dataset is untouched.",
        "decision": "REJECTED",
    }
