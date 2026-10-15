from pathlib import Path
import re

SENSOR_NAMES = ("acc", "gyro", "mag", "baro")

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())

def activity_alias(s):
    x = norm(s)
    return {
        "elevatordown": "liftdown",
        "elevatorup": "liftup",
        "liftdown": "liftdown",
        "liftup": "liftup",
    }.get(x, x)

def discover_sensor_files(data_dir):
    """
    Return rows describing every sensor TXT file without reading samples.
    AICOS-HAR raw hierarchy is inferred from the first path components:
    participant / activity_trial / device_position / sensor.txt.
    """
    data_dir = Path(data_dir)
    rows = []
    for p in data_dir.rglob("*.txt"):
        rel = p.relative_to(data_dir)
        if len(rel.parts) < 4:
            continue
        participant, activity_trial, device_position = rel.parts[:3]
        sensor = p.stem.lower()
        if sensor not in SENSOR_NAMES:
            continue
        m = re.match(r"^(.*)_(\d+)$", activity_trial)
        activity = m.group(1) if m else activity_trial
        trial = m.group(2) if m else ""
        aid = "/".join((participant, activity_trial, device_position))
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
