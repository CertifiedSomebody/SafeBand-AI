from pathlib import Path
import re

SENSOR_ALIASES = {
    "acc": {"acc", "accelerometer", "acceleration"},
    "gyro": {"gyro", "gyroscope"},
    "mag": {"mag", "magnetometer"},
    "baro": {"baro", "barometer"},
}

# AICOS metadata uses LiftDown/LiftUp, while the raw directory uses
# ElevatorDown/ElevatorUp. This is a path-only alias; metadata labels remain
# unchanged for ML.
ACTIVITY_DIR_ALIASES = {
    "LiftDown": "ElevatorDown",
    "LiftUp": "ElevatorUp",
}

def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def normalize_sensor(stem):
    s = _norm(stem)
    for canonical, aliases in SENSOR_ALIASES.items():
        if s in {_norm(a) for a in aliases}:
            return canonical
    return None

def metadata_to_raw_acquisition_id(acquisition_id):
    parts = str(acquisition_id).strip().replace("\\", "/").split("/")
    if len(parts) >= 2:
        activity_trial = parts[1]
        m = re.match(r"^(.*)_(\d+)$", activity_trial)
        if m and m.group(1) in ACTIVITY_DIR_ALIASES:
            parts[1] = f"{ACTIVITY_DIR_ALIASES[m.group(1)]}_{m.group(2)}"
    return "/".join(parts)

def raw_to_metadata_acquisition_id(acquisition_id):
    parts = str(acquisition_id).strip().replace("\\", "/").split("/")
    if len(parts) >= 2:
        activity_trial = parts[1]
        m = re.match(r"^(.*)_(\d+)$", activity_trial)
        reverse = {v: k for k, v in ACTIVITY_DIR_ALIASES.items()}
        if m and m.group(1) in reverse:
            parts[1] = f"{reverse[m.group(1)]}_{m.group(2)}"
    return "/".join(parts)

def metadata_activity(acquisition_id):
    parts = str(acquisition_id).replace("\\", "/").split("/")
    if len(parts) < 2:
        return ""
    m = re.match(r"^(.*)_(\d+)$", parts[1])
    return m.group(1) if m else parts[1]

def discover_sensor_files(data_dir):
    """Discover Sensor.txt files under Participant/Activity_Trial/Device_Position."""
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
        raw_aid = "/".join((participant, activity_trial, device_position))
        metadata_aid = raw_to_metadata_acquisition_id(raw_aid)
        m = re.match(r"^(.*)_(\d+)$", activity_trial)
        activity_raw = m.group(1) if m else activity_trial
        trial = m.group(2) if m else ""
        rows.append({
            "acquisition_id": metadata_aid,
            "raw_acquisition_id": raw_aid,
            "participant_id": participant,
            "activity_trial_id": activity_trial,
            "activity": activity_raw,
            "trial_no": trial,
            "device_position_id": device_position,
            "sensor": sensor,
            "path": str(p),
            "relative_path": str(rel),
        })
    return rows

def index_sensor_files(data_dir):
    rows = discover_sensor_files(data_dir)
    return {(r["acquisition_id"], r["sensor"]): r for r in rows}, rows
