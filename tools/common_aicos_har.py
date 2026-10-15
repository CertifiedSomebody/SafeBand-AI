from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_SENSOR_NAMES = {
    "accelerometer", "acc", "accel",
    "gyroscope", "gyro",
    "magnetometer", "mag",
    "barometer", "baro",
}

def repo_path(*parts):
    return ROOT.joinpath(*parts)
