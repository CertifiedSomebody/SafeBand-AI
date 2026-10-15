
from pathlib import Path
import re

SENSOR_ALIASES = {
    "acc": {"acc", "accelerometer", "acceleration"},
    "gyro": {"gyro", "gyroscope"},
    "mag": {"mag", "magnetometer"},
    "baro": {"baro", "barometer"},
}

def normalize_sensor(stem):
    s = re.sub(r"[^a-z0-9]+", "", stem.lower())
    for canonical, aliases in SENSOR_ALIASES.items():
        if s in {re.sub(r"[^a-z0-9]+", "", a) for a in aliases}:
            return canonical
    return None

def discover_sensor_files(data_dir):
    """
    Discover AICOS-HAR files using the documented hierarchy:
    Participant / Activity_Trial / Device_Position / Sensor.txt

    Important: the sensor filename may be 'accelerometer.txt',
    'gyroscope.txt', etc.; it is NOT assumed to be 'acc.txt'.
    """
    data_dir = Path(data_dir).resolve()
    rows = []
    for p in data_dir.rglob("*.txt"):
        rel = p.relative_to(data_dir)
        if len(rel.parts) < 4:
            continue
        participant, activity_trial, device_position = rel.parts[:3]
        sensor = normalize_sensor(p.stem)
        if sensor is None:
            continue
        aid = "/".join((participant, activity_trial, device_position))
        m = re.match(r"^(.*)_(\d+)$", activity_trial)
        activity = m.group(1) if m else activity_trial
        trial = m.group(2) if m else ""
        rows.append({
            "acquisition_id": aid,
            "participant_id": participant,
            "activity_trial_id": activity_trial,
            "activity": activity,
            "trial_no": trial,
            "device_position_id": device_position,
            "sensor": sensor,
            "path": str(p),
            "relative_path": str(rel),
        })
    return rows
