"""
SAFEBAND AI - MAX30102 Physiological Sensor Interface

Prototype interface for the MAX30102 pulse oximeter and
heart-rate sensor.

Provides:
- Heart rate (BPM)
- SpO2 (%)
- Physiological status
- Sensor connection status

Current version:
    Uses simulated physiological data for the software
    demonstration.

Future version:
    Replace the simulation methods with the actual MAX30102
    I2C sensor driver without changing the rest of the
    SAFEBAND AI processing pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any
import random


@dataclass
class MAX30102Reading:
    """Represents one MAX30102 physiological reading."""

    heart_rate: float
    spo2: float
    physiological_state: str
    connected: bool
    simulated: bool = True


class MAX30102Sensor:
    """
    MAX30102 heart-rate and SpO2 sensor interface.

    Prototype operation:
        Generates realistic simulated physiological readings.

    Hardware operation:
        Can later be connected to the real MAX30102 through I2C.
    """

    def __init__(
        self,
        simulation: bool = True
    ):
        self.simulation = simulation
        self.connected = False

        self.heart_rate = 74.0
        self.spo2 = 98.0
        self.physiological_state = "NORMAL"

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self) -> bool:
        """
        Initialize the MAX30102 sensor.

        Returns
        -------
        bool
            True if the sensor is available.
        """

        if self.simulation:
            self.connected = True
            return True

        # Real MAX30102 I2C initialization will be implemented here.
        self.connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect the sensor."""

        self.connected = False

    # =========================================================
    # HEART RATE
    # =========================================================

    def read_heart_rate(self) -> float:
        """
        Read simulated heart rate.

        Returns
        -------
        float
            Heart rate in BPM.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.heart_rate += random.uniform(
                -2.5,
                2.5
            )

        self.heart_rate = max(
            35.0,
            min(
                200.0,
                self.heart_rate
            )
        )

        return round(
            self.heart_rate,
            1
        )

    # =========================================================
    # SPO2
    # =========================================================

    def read_spo2(self) -> float:
        """
        Read simulated blood oxygen saturation.

        Returns
        -------
        float
            SpO2 percentage.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.spo2 += random.uniform(
                -0.4,
                0.4
            )

        self.spo2 = max(
            80.0,
            min(
                100.0,
                self.spo2
            )
        )

        return round(
            self.spo2,
            1
        )

    # =========================================================
    # PHYSIOLOGICAL STATUS
    # =========================================================

    def classify_physiological_state(
        self,
        heart_rate: float,
        spo2: float
    ) -> str:
        """
        Classify the current physiological state.

        Classification is intended only for prototype
        demonstration and is not a medical diagnostic system.
        """

        if spo2 < 90:
            return "CRITICAL"

        if heart_rate >= 130:
            return "ELEVATED"

        if heart_rate >= 110:
            return "HIGH"

        if heart_rate < 50:
            return "LOW"

        if spo2 < 94:
            return "WARNING"

        return "NORMAL"

    # =========================================================
    # COMPLETE READING
    # =========================================================

    def read(self) -> MAX30102Reading:
        """
        Acquire a complete MAX30102 reading.
        """

        heart_rate = self.read_heart_rate()
        spo2 = self.read_spo2()

        self.physiological_state = (
            self.classify_physiological_state(
                heart_rate,
                spo2
            )
        )

        return MAX30102Reading(
            heart_rate=heart_rate,
            spo2=spo2,
            physiological_state=self.physiological_state,
            connected=self.connected,
            simulated=self.simulation
        )

    # =========================================================
    # DICTIONARY OUTPUT
    # =========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the physiological reading as a dictionary.

        This format is compatible with the SAFEBAND AI
        sensor-fusion and risk-assessment modules.
        """

        reading = self.read()

        return {
            "heart_rate": reading.heart_rate,
            "spo2": reading.spo2,
            "physiological_state": (
                reading.physiological_state
            ),
            "max30102_connected": reading.connected,
            "simulated": reading.simulated,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current MAX30102 sensor status."""

        return {
            "sensor": "MAX30102",
            "name": "Heart Rate & SpO2 Sensor",
            "connected": self.connected,
            "simulation": self.simulation,
            "heart_rate": self.heart_rate,
            "spo2": self.spo2,
            "physiological_state": (
                self.physiological_state
            ),
        }


# =============================================================
# GLOBAL SENSOR INSTANCE
# =============================================================

_max30102 = MAX30102Sensor(
    simulation=True
)


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def initialize_max30102() -> bool:
    """Initialize the MAX30102 sensor."""

    return _max30102.connect()


def read_max30102() -> Dict[str, Any]:
    """Read heart rate and SpO2."""

    return _max30102.read_dict()


def get_max30102_status() -> Dict[str, Any]:
    """Return MAX30102 status."""

    return _max30102.get_status()