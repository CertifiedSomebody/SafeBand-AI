from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


SENTINEL: Final[float] = -999.0
WINDOW_MS: Final[float] = 1000.0

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "accel_magnitude_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "gyro_magnitude_rad_s",
    "heading_sin",
    "heading_cos",
    "roll_sin",
    "roll_cos",
    "pitch_sin",
    "pitch_cos",
    "heading_rate_deg_s",
    "roll_rate_deg_s",
    "pitch_rate_deg_s",
    "accel_magnitude_delta_m_s2",
    "accel_jerk_m_s3",
    "rolling_accel_mean_1s_m_s2",
    "rolling_accel_std_1s_m_s2",
    "rolling_accel_max_1s_m_s2",
    "rolling_gyro_mean_1s_rad_s",
    "rolling_gyro_std_1s_rad_s",
    "rolling_jerk_abs_mean_1s_m_s3",
    "temperature_C",
    "humidity_pct",
    "pressure_hPa",
    "gas_kOhm",
    "audio_rms",
    "audio_dc",
    "audio_average_rms",
    "audio_max_rms",
    "audio_blocks",
    "ppg_max_ir",
    "ppg_max_red",
    "raw_hr_bpm",
    "raw_spo2_pct",
    "final_hr_bpm",
    "final_spo2_pct",
    "battery_voltage_V",
    "battery_percent",
    "charger_connected",
    "accel_available",
    "gyro_available",
    "orientation_available",
    "environment_available",
    "audio_available",
    "ppg_contact",
    "raw_hr_available",
    "raw_spo2_available",
    "final_hr_available",
    "final_spo2_available",
    "battery_available",
    "timing_interval_ms",
    "timing_valid",
    "timing_stall",
    "ppg_available",
)

_OPTIONAL_SENSOR_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "accel": (
        "accel_x_m_s2",
        "accel_y_m_s2",
        "accel_z_m_s2",
    ),
    "gyro": (
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
    ),
    "orientation": (
        "heading_deg",
        "roll_deg",
        "pitch_deg",
    ),
}

_SIMPLE_OPTIONAL_FEATURES: Final[dict[str, str]] = {
    "temperature_C": "temperature_C",
    "humidity_pct": "humidity_pct",
    "pressure_hPa": "pressure_hPa",
    "gas_kOhm": "gas_kOhm",
    "audio_rms": "mic_rms",
    "audio_dc": "mic_dc",
    "audio_average_rms": "mic_average_rms",
    "audio_max_rms": "mic_max_rms",
    "audio_blocks": "mic_blocks",
    "ppg_max_ir": "max_ir",
    "ppg_max_red": "max_red",
    "battery_voltage_V": "battery_voltage_V",
    "battery_percent": "battery_percent",
    "charger_connected": "charger_connected",
}

_PPG_MEASUREMENTS: Final[dict[str, str]] = {
    "raw_hr_bpm": "raw_hr",
    "raw_spo2_pct": "raw_spo2",
    "final_hr_bpm": "final_hr",
    "final_spo2_pct": "final_spo2",
}
_PPG_VALIDITY: Final[dict[str, str]] = {
    "raw_hr_bpm": "raw_hr_valid",
    "raw_spo2_pct": "raw_spo2_valid",
    "final_hr_bpm": "final_hr_valid",
    "final_spo2_pct": "final_spo2_valid",
}


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid_text = frame[column].notna() & values.isna()
    if invalid_text.any():
        raise ValueError(f"{column} contains non-numeric values")
    return values.to_numpy(dtype=float)


def _flags(frame: pd.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    flags = _numeric(frame, column)
    rounded = np.rint(flags)
    if (
        not np.isfinite(flags).all()
        or np.any(np.abs(flags - rounded) > 1e-6)
        or not np.isin(rounded, (0.0, 1.0)).all()
    ):
        raise ValueError(f"{column} must contain only finite 0/1 flags")
    return rounded == 1.0


def _present_group(
    frame: pd.DataFrame,
    group_name: str,
    columns: tuple[str, ...],
) -> bool:
    present = [column in frame.columns for column in columns]
    if any(present) and not all(present):
        raise ValueError(
            f"{group_name} sensor columns must be all present or all absent"
        )
    return all(present)


def _available(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & ~np.isclose(
        values,
        SENTINEL,
        rtol=0.0,
        atol=1e-6,
    )


def _masked_values(
    values: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    return np.where(available, values, 0.0).astype(float)


def _angular_rate(
    values_degrees: np.ndarray,
    available: np.ndarray,
    elapsed_ms: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(values_degrees), dtype=float)
    for index in range(1, len(values_degrees)):
        delta_ms = elapsed_ms[index] - elapsed_ms[index - 1]
        if delta_ms <= 0 or not (available[index] and available[index - 1]):
            continue
        delta_degrees = (
            (values_degrees[index] - values_degrees[index - 1] + 180.0)
            % 360.0
        ) - 180.0
        result[index] = delta_degrees / (delta_ms / 1000.0)
    return result


def _rolling_features(
    elapsed_ms: np.ndarray,
    acceleration_magnitude: np.ndarray,
    accel_available: np.ndarray,
    gyro_magnitude: np.ndarray,
    gyro_available: np.ndarray,
    jerk: np.ndarray,
) -> dict[str, np.ndarray]:
    size = len(elapsed_ms)
    acc_mean = np.zeros(size)
    acc_std = np.zeros(size)
    acc_max = np.zeros(size)
    gyro_mean = np.zeros(size)
    gyro_std = np.zeros(size)
    jerk_mean = np.zeros(size)
    for index, timestamp in enumerate(elapsed_ms):
        start = index
        while start > 0 and timestamp - elapsed_ms[start - 1] <= WINDOW_MS:
            start -= 1
        acc_window = acceleration_magnitude[start : index + 1][
            accel_available[start : index + 1]
        ]
        if acc_window.size:
            acc_mean[index] = float(np.mean(acc_window))
            acc_std[index] = float(np.std(acc_window))
            acc_max[index] = float(np.max(acc_window))
        gyro_window = gyro_magnitude[start : index + 1][
            gyro_available[start : index + 1]
        ]
        if gyro_window.size:
            gyro_mean[index] = float(np.mean(gyro_window))
            gyro_std[index] = float(np.std(gyro_window))
        jerk_window = np.abs(jerk[start : index + 1])
        jerk_mean[index] = float(np.mean(jerk_window)) if jerk_window.size else 0.0
    return {
        "rolling_accel_mean_1s_m_s2": acc_mean,
        "rolling_accel_std_1s_m_s2": acc_std,
        "rolling_accel_max_1s_m_s2": acc_max,
        "rolling_gyro_mean_1s_rad_s": gyro_mean,
        "rolling_gyro_std_1s_rad_s": gyro_std,
        "rolling_jerk_abs_mean_1s_m_s3": jerk_mean,
    }


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract causal per-row features and explicit sensor availability masks."""
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS, index=frame.index)
    for required in ("elapsed_ms", "sample_index"):
        if required not in frame.columns:
            raise ValueError(f"missing required timing column: {required}")

    elapsed_ms = _numeric(frame, "elapsed_ms")
    sample_index = _numeric(frame, "sample_index")
    if (
        not np.isfinite(elapsed_ms).all()
        or np.any(np.diff(elapsed_ms) <= 0.0)
    ):
        raise ValueError("elapsed_ms must be finite and strictly increasing")
    if (
        not np.isfinite(sample_index).all()
        or np.any(np.diff(sample_index) < 0.0)
    ):
        raise ValueError("sample_index must be finite and non-decreasing")

    result: dict[str, np.ndarray] = {}
    accel_columns = _OPTIONAL_SENSOR_GROUPS["accel"]
    accel_present = _present_group(frame, "accelerometer", accel_columns)
    accel_axes = [_numeric(frame, column) for column in accel_columns]
    accel_available = (
        np.logical_and.reduce([_available(axis) for axis in accel_axes])
        if accel_present
        else np.zeros(len(frame), dtype=bool)
    )
    for column, axis in zip(accel_columns, accel_axes):
        result[column] = _masked_values(axis, accel_available)
    acceleration_magnitude = np.sqrt(
        sum(np.square(result[column]) for column in accel_columns)
    )
    result["accel_magnitude_m_s2"] = acceleration_magnitude
    result["accel_available"] = accel_available.astype(float)

    gyro_columns = _OPTIONAL_SENSOR_GROUPS["gyro"]
    gyro_present = _present_group(frame, "gyroscope", gyro_columns)
    gyro_axes = [_numeric(frame, column) for column in gyro_columns]
    gyro_available = (
        np.logical_and.reduce([_available(axis) for axis in gyro_axes])
        if gyro_present
        else np.zeros(len(frame), dtype=bool)
    )
    for column, axis in zip(gyro_columns, gyro_axes):
        result[column] = _masked_values(axis, gyro_available)
    gyro_magnitude = np.sqrt(
        sum(np.square(result[column]) for column in gyro_columns)
    )
    result["gyro_magnitude_rad_s"] = gyro_magnitude
    result["gyro_available"] = gyro_available.astype(float)

    orientation_columns = _OPTIONAL_SENSOR_GROUPS["orientation"]
    orientation_present = _present_group(
        frame,
        "orientation",
        orientation_columns,
    )
    orientation = {
        column: _numeric(frame, column)
        for column in orientation_columns
    }
    orientation_available = (
        np.logical_and.reduce(
            [_available(values) for values in orientation.values()]
        )
        if orientation_present
        else np.zeros(len(frame), dtype=bool)
    )
    for column, feature_prefix in (
        ("heading_deg", "heading"),
        ("roll_deg", "roll"),
        ("pitch_deg", "pitch"),
    ):
        radians = np.deg2rad(
            _masked_values(orientation[column], orientation_available)
        )
        result[f"{feature_prefix}_sin"] = np.sin(radians)
        result[f"{feature_prefix}_cos"] = np.cos(radians)
    result["orientation_available"] = orientation_available.astype(float)
    result["heading_rate_deg_s"] = _angular_rate(
        orientation["heading_deg"],
        orientation_available,
        elapsed_ms,
    )
    result["roll_rate_deg_s"] = _angular_rate(
        orientation["roll_deg"],
        orientation_available,
        elapsed_ms,
    )
    result["pitch_rate_deg_s"] = _angular_rate(
        orientation["pitch_deg"],
        orientation_available,
        elapsed_ms,
    )

    accel_delta = np.zeros(len(frame), dtype=float)
    accel_jerk = np.zeros(len(frame), dtype=float)
    for index in range(1, len(frame)):
        delta_ms = elapsed_ms[index] - elapsed_ms[index - 1]
        if delta_ms > 0 and accel_available[index] and accel_available[index - 1]:
            accel_delta[index] = (
                acceleration_magnitude[index]
                - acceleration_magnitude[index - 1]
            )
            accel_jerk[index] = accel_delta[index] / (delta_ms / 1000.0)
    result["accel_magnitude_delta_m_s2"] = accel_delta
    result["accel_jerk_m_s3"] = accel_jerk
    result.update(
        _rolling_features(
            elapsed_ms,
            acceleration_magnitude,
            accel_available,
            gyro_magnitude,
            gyro_available,
            accel_jerk,
        )
    )

    environment_columns = (
        "temperature_C",
        "humidity_pct",
        "pressure_hPa",
        "gas_kOhm",
    )
    environment_values = {
        column: _numeric(frame, column)
        for column in environment_columns
    }
    environment_flags = _flags(frame, "bme_valid")
    environment_available = np.logical_and.reduce(
        [_available(values) for values in environment_values.values()]
    )
    if environment_flags is not None:
        if np.any(environment_flags & ~environment_available):
            raise ValueError("bme_valid=1 requires all finite environment measurements")
        environment_available &= environment_flags
    for column, values in environment_values.items():
        result[column] = _masked_values(values, environment_available)
    result["environment_available"] = environment_available.astype(float)

    for feature, column in _SIMPLE_OPTIONAL_FEATURES.items():
        if feature in result:
            continue
        values = _numeric(frame, column)
        result[feature] = _masked_values(values, _available(values))

    mic_blocks = _numeric(frame, "mic_blocks")
    audio_columns = (
        "mic_rms",
        "mic_dc",
        "mic_average_rms",
        "mic_max_rms",
        "mic_blocks",
    )
    audio_present = any(column in frame.columns for column in audio_columns)
    audio_values = [_numeric(frame, column) for column in audio_columns]
    audio_data_available = (
        np.logical_and.reduce([_available(values) for values in audio_values])
        if audio_present
        else np.zeros(len(frame), dtype=bool)
    )
    audio_available = (
        audio_data_available
        & (mic_blocks > 0.0)
        & np.logical_and.reduce([values >= 0.0 for values in audio_values])
    )
    for feature, column in (
        ("audio_rms", "mic_rms"),
        ("audio_dc", "mic_dc"),
        ("audio_average_rms", "mic_average_rms"),
        ("audio_max_rms", "mic_max_rms"),
        ("audio_blocks", "mic_blocks"),
    ):
        result[feature] = _masked_values(
            _numeric(frame, column),
            audio_available,
        )
    result["audio_available"] = audio_available.astype(float)

    finger_detected = _flags(frame, "finger_detected")
    raw_hr_available = np.zeros(len(frame), dtype=bool)
    raw_spo2_available = np.zeros(len(frame), dtype=bool)
    final_hr_available = np.zeros(len(frame), dtype=bool)
    final_spo2_available = np.zeros(len(frame), dtype=bool)
    for feature, measurement in _PPG_MEASUREMENTS.items():
        values = _numeric(frame, measurement)
        flags = _flags(frame, _PPG_VALIDITY[feature])
        available = _available(values)
        if flags is None:
            available &= False
        else:
            if np.any(flags & ~available):
                raise ValueError(
                    f"{_PPG_VALIDITY[feature]}=1 requires a non-sentinel value"
                )
            available &= flags
        if finger_detected is not None:
            available &= finger_detected
        result[feature] = _masked_values(values, available)
        if feature == "raw_hr_bpm":
            raw_hr_available = available
        elif feature == "raw_spo2_pct":
            raw_spo2_available = available
        elif feature == "final_hr_bpm":
            final_hr_available = available
        else:
            final_spo2_available = available
    result["ppg_contact"] = (
        finger_detected.astype(float)
        if finger_detected is not None
        else np.zeros(len(frame), dtype=float)
    )
    ppg_contact = (
        finger_detected
        if finger_detected is not None
        else np.zeros(len(frame), dtype=bool)
    )
    for feature, column in (
        ("ppg_max_ir", "max_ir"),
        ("ppg_max_red", "max_red"),
    ):
        result[feature] = _masked_values(
            _numeric(frame, column),
            ppg_contact & _available(_numeric(frame, column)),
        )
    result["ppg_available"] = (
        ppg_contact
        & (
            _available(_numeric(frame, "max_ir"))
            | _available(_numeric(frame, "max_red"))
        )
    ).astype(float)
    result["raw_hr_available"] = raw_hr_available.astype(float)
    result["raw_spo2_available"] = raw_spo2_available.astype(float)
    result["final_hr_available"] = final_hr_available.astype(float)
    result["final_spo2_available"] = final_spo2_available.astype(float)

    battery_values = {
        "battery_voltage_V": _numeric(frame, "battery_voltage_V"),
        "battery_percent": _numeric(frame, "battery_percent"),
    }
    battery_available = np.logical_and.reduce(
        [_available(values) for values in battery_values.values()]
    )
    charger = _flags(frame, "charger_connected")
    if charger is not None:
        result["charger_connected"] = charger.astype(float)
    else:
        result["charger_connected"] = np.zeros(len(frame), dtype=float)
    for feature, values in battery_values.items():
        result[feature] = _masked_values(values, battery_available)
    result["battery_available"] = battery_available.astype(float)

    timing_interval = np.zeros(len(frame), dtype=float)
    timing_interval[1:] = np.diff(elapsed_ms)
    timing_valid = np.zeros(len(frame), dtype=float)
    timing_valid[1:] = 1.0
    result["timing_interval_ms"] = timing_interval
    result["timing_valid"] = timing_valid
    result["timing_stall"] = (timing_interval > 1000.0).astype(float)

    features = pd.DataFrame(result, index=frame.index)
    features = features.loc[:, FEATURE_COLUMNS]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("feature extraction produced a non-finite value")
    return features


__all__ = ["FEATURE_COLUMNS", "extract_features"]
