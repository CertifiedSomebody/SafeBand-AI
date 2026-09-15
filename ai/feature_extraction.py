"""
SAFEBAND AI - Time-Series Feature Extraction

Converts a synchronized SafeBand sensor window into a compact,
model-friendly feature dictionary.

This is deliberately a small, deterministic feature layer. The
feature contract can be revised after public-dataset analysis.
"""

import math
from typing import Any, Dict, Iterable, List


FEATURE_COLUMNS = [
    "acceleration_mean",
    "acceleration_std",
    "acceleration_min",
    "acceleration_max",
    "acceleration_rms",
    "motion_mean",
    "motion_std",
    "motion_max",
    "gyro_magnitude_mean",
    "gyro_magnitude_std",
    "orientation_abs_mean",
    "orientation_abs_max",
    "heart_rate_mean",
    "heart_rate_std",
    "heart_rate_min",
    "heart_rate_max",
    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "body_temperature_mean",
    "body_temperature_std",
    "ambient_temperature_mean",
    "humidity_mean",
    "pressure_mean",
    "audio_level_mean",
    "audio_level_max",
]


def _number(sample: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = sample.get(key, default)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _stats(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "rms": 0.0}
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "rms": math.sqrt(sum(x * x for x in values) / len(values)),
    }


def extract_features(window: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract deterministic features from one SafeBand sensor window."""
    if not window:
        return {name: 0.0 for name in FEATURE_COLUMNS}

    acceleration = []
    motion = []
    gyro_magnitude = []
    orientation_abs = []
    heart_rate = []
    spo2 = []
    body_temperature = []
    ambient_temperature = []
    humidity = []
    pressure = []
    audio_level = []

    for sample in window:
        ax = _number(sample, "acceleration_x")
        ay = _number(sample, "acceleration_y")
        az = _number(sample, "acceleration_z")
        acceleration.append(math.sqrt(ax * ax + ay * ay + az * az))

        motion.append(_number(sample, "motion_intensity"))

        gx = _number(sample, "gyroscope_x")
        gy = _number(sample, "gyroscope_y")
        gz = _number(sample, "gyroscope_z")
        gyro_magnitude.append(math.sqrt(gx * gx + gy * gy + gz * gz))

        orientation_abs.append(abs(_number(sample, "orientation")))
        heart_rate.append(_number(sample, "heart_rate", 75.0))
        spo2.append(_number(sample, "spo2", 98.0))
        body_temperature.append(_number(sample, "body_temperature", 36.7))
        ambient_temperature.append(_number(sample, "temperature", 25.0))
        humidity.append(_number(sample, "humidity", 50.0))
        pressure.append(_number(sample, "pressure", 1013.0))
        audio_level.append(_number(sample, "audio_level"))

    a = _stats(acceleration)
    m = _stats(motion)
    g = _stats(gyro_magnitude)
    o = _stats(orientation_abs)
    h = _stats(heart_rate)
    s = _stats(spo2)
    bt = _stats(body_temperature)
    at = _stats(ambient_temperature)
    hu = _stats(humidity)
    pr = _stats(pressure)
    au = _stats(audio_level)

    return {
        "acceleration_mean": a["mean"],
        "acceleration_std": a["std"],
        "acceleration_min": a["min"],
        "acceleration_max": a["max"],
        "acceleration_rms": a["rms"],
        "motion_mean": m["mean"],
        "motion_std": m["std"],
        "motion_max": m["max"],
        "gyro_magnitude_mean": g["mean"],
        "gyro_magnitude_std": g["std"],
        "orientation_abs_mean": o["mean"],
        "orientation_abs_max": o["max"],
        "heart_rate_mean": h["mean"],
        "heart_rate_std": h["std"],
        "heart_rate_min": h["min"],
        "heart_rate_max": h["max"],
        "spo2_mean": s["mean"],
        "spo2_std": s["std"],
        "spo2_min": s["min"],
        "body_temperature_mean": bt["mean"],
        "body_temperature_std": bt["std"],
        "ambient_temperature_mean": at["mean"],
        "humidity_mean": hu["mean"],
        "pressure_mean": pr["mean"],
        "audio_level_mean": au["mean"],
        "audio_level_max": au["max"],
    }


__all__ = ["FEATURE_COLUMNS", "extract_features"]
