"""
SAFEBAND AI - Simulated Sensor Data Engine

Generates realistic, continuously changing sensor data for the
SAFEBAND AI prototype.

The simulator provides the same type of data that will eventually
come from the real hardware sensors:

- MAX30102 : Heart rate and SpO2
- BNO055   : Acceleration, motion and orientation
- BME680   : Temperature, humidity and pressure
- INMP441  : Audio level
- GPS      : Latitude and longitude

The generated data is intended ONLY for prototype demonstration.
"""

import random
import math
from datetime import datetime
from typing import Dict, Any, Optional

from data.demo_scenarios import get_scenario


class SimulatedSensorData:
    """
    SAFEBAND AI simulated sensor-data generator.

    The simulator supports:
        - Normal real-time variation
        - Fixed demonstration scenarios
        - Automatic sensor drift
        - GPS movement
        - Timestamp generation
    """

    def __init__(self):
        self.current_scenario = "NORMAL"

        self.data = get_scenario(
            self.current_scenario
        )

        self.start_time = datetime.now()

        self.sample_count = 0

        self.previous_latitude = self.data["latitude"]
        self.previous_longitude = self.data["longitude"]

    # ========================================================
    # SCENARIO CONTROL
    # ========================================================

    def set_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Change the active simulation scenario.

        Parameters
        ----------
        scenario : str
            One of:
            NORMAL
            WALKING
            RUNNING
            FALL
            HIGH_RISK
            SOS

        Returns
        -------
        dict
            Base sensor data for the selected scenario.
        """

        self.current_scenario = str(
            scenario
        ).upper().strip()

        self.data = get_scenario(
            self.current_scenario
        )

        return self.data.copy()

    def get_scenario(self) -> str:
        """Return the currently active scenario."""

        return self.current_scenario

    # ========================================================
    # SENSOR NOISE
    # ========================================================

    @staticmethod
    def _noise(
        value: float,
        variation: float
    ) -> float:
        """
        Add small random variation to a sensor value.
        """

        return value + random.uniform(
            -variation,
            variation
        )

    # ========================================================
    # HEART RATE
    # MAX30102
    # ========================================================

    def _simulate_heart_rate(self) -> float:
        """Generate simulated heart-rate data."""

        base = float(
            self.data.get(
                "heart_rate",
                75.0
            )
        )

        variation = {
            "NORMAL": 2.5,
            "WALKING": 4.0,
            "RUNNING": 6.0,
            "FALL": 5.0,
            "HIGH_RISK": 5.0,
            "SOS": 4.0,
        }.get(
            self.current_scenario,
            3.0
        )

        return round(
            self._noise(
                base,
                variation
            ),
            1
        )

    # ========================================================
    # SPO2
    # MAX30102
    # ========================================================

    def _simulate_spo2(self) -> float:
        """Generate simulated SpO2 data."""

        base = float(
            self.data.get(
                "spo2",
                98.0
            )
        )

        return round(
            self._noise(
                base,
                0.5
            ),
            1
        )

    # ========================================================
    # ACCELERATION
    # BNO055
    # ========================================================

    def _simulate_acceleration(self) -> Dict[str, float]:
        """Generate simulated three-axis acceleration."""

        ax = self._noise(
            float(
                self.data.get(
                    "acceleration_x",
                    0.0
                )
            ),
            0.08
        )

        ay = self._noise(
            float(
                self.data.get(
                    "acceleration_y",
                    0.0
                )
            ),
            0.08
        )

        az = self._noise(
            float(
                self.data.get(
                    "acceleration_z",
                    1.0
                )
            ),
            0.08
        )

        return {
            "acceleration_x": round(ax, 3),
            "acceleration_y": round(ay, 3),
            "acceleration_z": round(az, 3),
        }

    # ========================================================
    # MOTION
    # BNO055
    # ========================================================

    def _simulate_motion(self) -> float:
        """Generate simulated motion intensity."""

        base = float(
            self.data.get(
                "motion_intensity",
                0.1
            )
        )

        if self.current_scenario == "FALL":
            # Keep fall detection deterministic.
            return round(
                self._noise(base, 0.10),
                2
            )

        return round(
            max(
                0.0,
                self._noise(
                    base,
                    max(0.03, base * 0.08)
                )
            ),
            2
        )

    # ========================================================
    # ORIENTATION
    # BNO055
    # ========================================================

    def _simulate_orientation(self) -> float:
        """Generate simulated body orientation."""

        base = float(
            self.data.get(
                "orientation",
                0.0
            )
        )

        return round(
            self._noise(
                base,
                2.0
            ),
            1
        )

    # ========================================================
    # ENVIRONMENT
    # BME680
    # ========================================================

    def _simulate_temperature(self) -> float:
        """Generate simulated environmental temperature."""

        base = float(
            self.data.get(
                "temperature",
                25.0
            )
        )

        return round(
            self._noise(
                base,
                0.3
            ),
            1
        )

    def _simulate_humidity(self) -> float:
        """Generate simulated humidity."""

        base = float(
            self.data.get(
                "humidity",
                50.0
            )
        )

        return round(
            max(
                0.0,
                min(
                    100.0,
                    self._noise(
                        base,
                        1.0
                    )
                )
            ),
            1
        )

    def _simulate_pressure(self) -> float:
        """Generate simulated atmospheric pressure."""

        base = float(
            self.data.get(
                "pressure",
                1013.0
            )
        )

        return round(
            self._noise(
                base,
                1.5
            ),
            1
        )

    # ========================================================
    # AUDIO
    # INMP441
    # ========================================================

    def _simulate_audio(self) -> float:
        """Generate simulated microphone/audio level."""

        base = float(
            self.data.get(
                "audio_level",
                0.1
            )
        )

        return round(
            max(
                0.0,
                min(
                    1.0,
                    self._noise(
                        base,
                        0.04
                    )
                )
            ),
            2
        )

    # ========================================================
    # GPS
    # ========================================================

    def _simulate_gps(self) -> Dict[str, float]:
        """
        Generate simulated GPS coordinates.

        Walking and running scenarios produce small position
        changes to demonstrate movement tracking.
        """

        latitude = float(
            self.data.get(
                "latitude",
                19.0760
            )
        )

        longitude = float(
            self.data.get(
                "longitude",
                72.8777
            )
        )

        if self.current_scenario in (
            "WALKING",
            "RUNNING"
        ):
            latitude += random.uniform(
                -0.0001,
                0.0002
            )

            longitude += random.uniform(
                -0.0001,
                0.0002
            )

        return {
            "latitude": round(
                latitude,
                6
            ),
            "longitude": round(
                longitude,
                6
            ),
        }

    # ========================================================
    # COMPLETE SENSOR SAMPLE
    # ========================================================

    def generate(self) -> Dict[str, Any]:
        """
        Generate one complete simulated sensor sample.

        Returns
        -------
        dict
            Complete SAFEBAND sensor dataset.
        """

        self.sample_count += 1

        acceleration = (
            self._simulate_acceleration()
        )

        gps = self._simulate_gps()

        sample = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "sample_id": self.sample_count,

            "scenario": self.current_scenario,

            # MAX30102
            "heart_rate": (
                self._simulate_heart_rate()
            ),

            "spo2": (
                self._simulate_spo2()
            ),

            # BNO055
            "acceleration_x": (
                acceleration["acceleration_x"]
            ),

            "acceleration_y": (
                acceleration["acceleration_y"]
            ),

            "acceleration_z": (
                acceleration["acceleration_z"]
            ),

            "motion_intensity": (
                self._simulate_motion()
            ),

            "orientation": (
                self._simulate_orientation()
            ),

            # BME680
            "temperature": (
                self._simulate_temperature()
            ),

            "humidity": (
                self._simulate_humidity()
            ),

            "pressure": (
                self._simulate_pressure()
            ),

            # INMP441
            "audio_level": (
                self._simulate_audio()
            ),

            # GPS
            "latitude": gps["latitude"],
            "longitude": gps["longitude"],

            # Manual emergency input
            "manual_sos": bool(
                self.data.get(
                    "manual_sos",
                    False
                )
            ),

            # Prototype marker
            "simulated": True,
        }

        return sample

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset simulator to the default NORMAL scenario."""

        self.current_scenario = "NORMAL"

        self.data = get_scenario(
            "NORMAL"
        )

        self.sample_count = 0

        self.previous_latitude = (
            self.data["latitude"]
        )

        self.previous_longitude = (
            self.data["longitude"]
        )


# ============================================================
# GLOBAL SIMULATOR
# ============================================================

_simulator = SimulatedSensorData()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def generate_sensor_data(
    scenario: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate one simulated sensor sample.

    If a scenario is supplied, it becomes the active scenario.
    Otherwise, the currently selected scenario is used.
    """

    if scenario is not None:
        _simulator.set_scenario(
            scenario
        )

    return _simulator.generate()


def set_simulation_scenario(
    scenario: str
) -> Dict[str, Any]:
    """Set the active simulation scenario."""

    return _simulator.set_scenario(
        scenario
    )


def get_simulation_scenario() -> str:
    """Return the active simulation scenario."""

    return _simulator.get_scenario()


def reset_simulator() -> None:
    """Reset the simulator to NORMAL."""

    _simulator.reset()