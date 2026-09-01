"""
SAFEBAND AI - Demonstration Scenarios

Provides deterministic sensor profiles for the prototype demo.

These scenarios simulate the type of data that the real SAFEBAND
sensors are expected to provide after hardware integration.

Available scenarios:
    NORMAL
    WALKING
    RUNNING
    FALL
    HIGH_RISK
    SOS
"""

from typing import Dict, Any


# ============================================================
# NORMAL
# ============================================================

NORMAL_SCENARIO: Dict[str, Any] = {
    "scenario": "NORMAL",
    "description": "User is stationary and all parameters are normal.",

    "heart_rate": 74.0,
    "spo2": 98.0,

    "acceleration_x": 0.02,
    "acceleration_y": 0.01,
    "acceleration_z": 1.00,

    "motion_intensity": 0.08,
    "orientation": 5.0,

    "temperature": 27.0,
    "humidity": 58.0,
    "pressure": 1012.0,

    "audio_level": 0.08,

    "latitude": 19.0760,
    "longitude": 72.8777,
}


# ============================================================
# WALKING
# ============================================================

WALKING_SCENARIO: Dict[str, Any] = {
    "scenario": "WALKING",
    "description": "User is walking normally.",

    "heart_rate": 86.0,
    "spo2": 98.0,

    "acceleration_x": 0.18,
    "acceleration_y": 0.12,
    "acceleration_z": 1.12,

    "motion_intensity": 0.45,
    "orientation": 8.0,

    "temperature": 27.2,
    "humidity": 57.0,
    "pressure": 1012.0,

    "audio_level": 0.12,

    "latitude": 19.0764,
    "longitude": 72.8781,
}


# ============================================================
# RUNNING
# ============================================================

RUNNING_SCENARIO: Dict[str, Any] = {
    "scenario": "RUNNING",
    "description": "User is performing high-intensity movement.",

    "heart_rate": 118.0,
    "spo2": 96.0,

    "acceleration_x": 0.85,
    "acceleration_y": 0.65,
    "acceleration_z": 1.35,

    "motion_intensity": 1.55,
    "orientation": 15.0,

    "temperature": 28.5,
    "humidity": 55.0,
    "pressure": 1011.0,

    "audio_level": 0.30,

    "latitude": 19.0770,
    "longitude": 72.8788,
}


# ============================================================
# FALL
# ============================================================

FALL_SCENARIO: Dict[str, Any] = {
    "scenario": "FALL",
    "description": "Sudden abnormal motion and orientation indicate a possible fall.",

    "heart_rate": 112.0,
    "spo2": 96.0,

    "acceleration_x": 1.80,
    "acceleration_y": 1.40,
    "acceleration_z": 2.00,

    "motion_intensity": 2.80,
    "orientation": 82.0,

    "temperature": 27.4,
    "humidity": 59.0,
    "pressure": 1012.0,

    "audio_level": 0.72,

    "latitude": 19.0775,
    "longitude": 72.8792,
}


# ============================================================
# HIGH RISK
# ============================================================

HIGH_RISK_SCENARIO: Dict[str, Any] = {
    "scenario": "HIGH_RISK",
    "description": "Multiple sensor parameters indicate an abnormal condition.",

    "heart_rate": 128.0,
    "spo2": 92.0,

    "acceleration_x": 0.90,
    "acceleration_y": 0.70,
    "acceleration_z": 1.40,

    "motion_intensity": 1.75,
    "orientation": 68.0,

    "temperature": 39.5,
    "humidity": 72.0,
    "pressure": 1008.0,

    "audio_level": 0.65,

    "latitude": 19.0778,
    "longitude": 72.8795,
}


# ============================================================
# SOS
# ============================================================

SOS_SCENARIO: Dict[str, Any] = {
    "scenario": "SOS",
    "description": "Manual emergency SOS has been activated by the user.",

    "heart_rate": 105.0,
    "spo2": 97.0,

    "acceleration_x": 0.10,
    "acceleration_y": 0.08,
    "acceleration_z": 1.02,

    "motion_intensity": 0.15,
    "orientation": 10.0,

    "temperature": 27.5,
    "humidity": 58.0,
    "pressure": 1012.0,

    "audio_level": 0.10,

    "latitude": 19.0782,
    "longitude": 72.8800,

    "manual_sos": True,
}


# ============================================================
# SCENARIO COLLECTION
# ============================================================

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "NORMAL": NORMAL_SCENARIO,
    "WALKING": WALKING_SCENARIO,
    "RUNNING": RUNNING_SCENARIO,
    "FALL": FALL_SCENARIO,
    "HIGH_RISK": HIGH_RISK_SCENARIO,
    "SOS": SOS_SCENARIO,
}


# ============================================================
# PUBLIC FUNCTIONS
# ============================================================

def get_scenario(name: str) -> Dict[str, Any]:
    """
    Return a copy of the requested demonstration scenario.

    Parameters
    ----------
    name : str
        Scenario name.

    Returns
    -------
    dict
        Sensor values for the selected scenario.

    Raises
    ------
    ValueError
        If the requested scenario does not exist.
    """

    scenario_name = str(name).upper().strip()

    if scenario_name not in SCENARIOS:
        available = ", ".join(SCENARIOS.keys())

        raise ValueError(
            f"Unknown scenario '{name}'. "
            f"Available scenarios: {available}"
        )

    return SCENARIOS[scenario_name].copy()


def get_available_scenarios() -> list:
    """Return the list of available demonstration scenarios."""

    return list(SCENARIOS.keys())


def is_sos_scenario(sensor_data: Dict[str, Any]) -> bool:
    """
    Check whether the current scenario represents a manual SOS.
    """

    return bool(
        sensor_data.get("manual_sos", False)
    )


def get_scenario_description(name: str) -> str:
    """Return the description of a demonstration scenario."""

    scenario = get_scenario(name)

    return scenario.get(
        "description",
        "No description available."
    )