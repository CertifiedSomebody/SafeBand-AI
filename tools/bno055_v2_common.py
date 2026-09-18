"""Shared contracts for SafeBand BNO055 Synthetic V2.1 tools."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACTIVITIES = [
    "FALL", "LYING", "RUNNING", "SITTING", "SIT_TO_STAND",
    "STAIRS", "STANDING", "STAND_TO_SIT", "WALKING"
]

RAW9_CHANNELS = [
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_x_dps", "gyro_y_dps", "gyro_z_dps",
    "mag_x_uT", "mag_y_uT", "mag_z_uT",
]

META_COLUMNS = [
    "subject_id", "session_id", "timestamp_s", "activity_label",
    "event_type", "event_start_s", "event_end_s",
]

DERIVED_COLUMNS = [
    "euler_heading_deg", "euler_roll_deg", "euler_pitch_deg",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "linear_accel_x_mps2", "linear_accel_y_mps2", "linear_accel_z_mps2",
    "gravity_x_mps2", "gravity_y_mps2", "gravity_z_mps2",
    "temperature_c",
]

REQUIRED_COLUMNS = META_COLUMNS + RAW9_CHANNELS + DERIVED_COLUMNS
