"""
SAFEBAND AI - MAX30102 Physiological Sensor Interface

Interface for the MAX30102 pulse oximeter and heart-rate sensor.

Measurements:
    - Heart rate (BPM)
    - SpO2 (%)
    - Basic physiological status
    - Sensor connection status

Current implementation:
    Simulated physiological readings.

Future implementation:
    Replace the simulation section with the actual MAX30102
    I2C driver without changing the application-level API.

IMPORTANT
---------
MAX30102 provides heart-rate and SpO2 measurements.

Body temperature is handled separately by MAX30208.

The physiological state reported by this module is a basic
threshold-based sensor status for prototype use. It is NOT
a medical diagnosis.
"""


from dataclasses import dataclass
from typing import Any, Dict

import random


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "MAX30102"
SENSOR_TYPE = "Physiological"

DEFAULT_HEART_RATE = 74.0
DEFAULT_SPO2 = 98.0

MIN_HEART_RATE = 30.0
MAX_HEART_RATE = 220.0

MIN_SPO2 = 70.0
MAX_SPO2 = 100.0

SPO2_CRITICAL_THRESHOLD = 90.0
SPO2_WARNING_THRESHOLD = 94.0

HEART_RATE_LOW_THRESHOLD = 50.0
HEART_RATE_HIGH_THRESHOLD = 110.0
HEART_RATE_ELEVATED_THRESHOLD = 130.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class MAX30102Reading:
    """Represents one complete MAX30102 reading."""

    heart_rate: float
    spo2: float
    physiological_state: str
    connected: bool
    simulated: bool = True


# ============================================================
# SENSOR CLASS
# ============================================================

class MAX30102Sensor:
    """
    MAX30102 heart-rate and SpO2 sensor interface.

    Responsibilities:
        - Heart-rate acquisition
        - SpO2 acquisition
        - Basic physiological status
        - Sensor connection state

    The class supports simulated operation now and provides a
    stable interface for future I2C hardware integration.
    """

    def __init__(
        self,
        simulation: bool = True,
    ) -> None:

        self.simulation = bool(
            simulation
        )

        self.connected = False

        self.heart_rate = (
            DEFAULT_HEART_RATE
        )

        self.spo2 = (
            DEFAULT_SPO2
        )

        self.physiological_state = (
            "NORMAL"
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Initialize the MAX30102 sensor.

        Returns
        -------
        bool
            True when the sensor is available.
        """

        if self.simulation:

            self.connected = True

            return True

        # ----------------------------------------------------
        # Future hardware implementation:
        #
        #   I2C initialization
        #   MAX30102 address detection
        #   LED/current configuration
        #   Sampling configuration
        #   FIFO configuration
        # ----------------------------------------------------

        self.connected = False

        return False

    def disconnect(self) -> None:
        """Disconnect the sensor."""

        self.connected = False

    def _ensure_connected(self) -> None:
        """Ensure the sensor is initialized before reading."""

        if not self.connected:
            self.connect()

    # ========================================================
    # HEART RATE
    # ========================================================

    def read_heart_rate(self) -> float:
        """
        Read heart rate.

        Returns
        -------
        float
            Heart rate in beats per minute.
        """

        self._ensure_connected()

        if self.simulation:

            self.heart_rate += random.uniform(
                -2.5,
                2.5,
            )

        self.heart_rate = max(
            MIN_HEART_RATE,
            min(
                MAX_HEART_RATE,
                self.heart_rate,
            ),
        )

        return round(
            self.heart_rate,
            1,
        )

    # ========================================================
    # SPO2
    # ========================================================

    def read_spo2(self) -> float:
        """
        Read blood oxygen saturation.

        Returns
        -------
        float
            SpO2 percentage.
        """

        self._ensure_connected()

        if self.simulation:

            self.spo2 += random.uniform(
                -0.4,
                0.4,
            )

        self.spo2 = max(
            MIN_SPO2,
            min(
                MAX_SPO2,
                self.spo2,
            ),
        )

        return round(
            self.spo2,
            1,
        )

    # ========================================================
    # BASIC PHYSIOLOGICAL STATUS
    # ========================================================

    @staticmethod
    def classify_physiological_state(
        heart_rate: float,
        spo2: float,
    ) -> str:
        """
        Classify the current physiological sensor status.

        Classification:

            SpO2 < 90       -> CRITICAL
            HR >= 130       -> ELEVATED
            HR >= 110       -> HIGH
            HR < 50         -> LOW
            SpO2 < 94       -> WARNING
            Otherwise       -> NORMAL

        This is a prototype threshold classification and is
        not a medical diagnostic system.
        """

        try:
            heart_rate = float(
                heart_rate
            )

        except (TypeError, ValueError):
            heart_rate = DEFAULT_HEART_RATE

        try:
            spo2 = float(
                spo2
            )

        except (TypeError, ValueError):
            spo2 = DEFAULT_SPO2

        if spo2 < SPO2_CRITICAL_THRESHOLD:
            return "CRITICAL"

        if heart_rate >= HEART_RATE_ELEVATED_THRESHOLD:
            return "ELEVATED"

        if heart_rate >= HEART_RATE_HIGH_THRESHOLD:
            return "HIGH"

        if heart_rate < HEART_RATE_LOW_THRESHOLD:
            return "LOW"

        if spo2 < SPO2_WARNING_THRESHOLD:
            return "WARNING"

        return "NORMAL"

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> MAX30102Reading:
        """
        Acquire one complete MAX30102 reading.
        """

        self._ensure_connected()

        heart_rate = (
            self.read_heart_rate()
        )

        spo2 = (
            self.read_spo2()
        )

        self.physiological_state = (
            self.classify_physiological_state(
                heart_rate,
                spo2,
            )
        )

        return MAX30102Reading(
            heart_rate=heart_rate,
            spo2=spo2,
            physiological_state=(
                self.physiological_state
            ),
            connected=(
                self.connected
            ),
            simulated=(
                self.simulation
            ),
        )

    # ========================================================
    # DICTIONARY OUTPUT
    # ========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the complete physiological reading as a dictionary.

        This format is consumed by the SAFEBAND sensor-fusion
        and risk-assessment pipeline.
        """

        reading = self.read()

        return {
            "heart_rate": (
                reading.heart_rate
            ),

            "spo2": (
                reading.spo2
            ),

            "physiological_state": (
                reading.physiological_state
            ),

            "max30102_connected": (
                reading.connected
            ),

            "simulated": (
                reading.simulated
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return the current MAX30102 status."""

        return {
            "sensor": SENSOR_NAME,

            "name": (
                "Heart Rate & SpO2 Sensor"
            ),

            "type": SENSOR_TYPE,

            "connected": (
                self.connected
            ),

            "simulation": (
                self.simulation
            ),

            "heart_rate": round(
                self.heart_rate,
                1,
            ),

            "spo2": round(
                self.spo2,
                1,
            ),

            "physiological_state": (
                self.physiological_state
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset simulated readings to their defaults."""

        self.heart_rate = (
            DEFAULT_HEART_RATE
        )

        self.spo2 = (
            DEFAULT_SPO2
        )

        self.physiological_state = (
            "NORMAL"
        )


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

_max30102 = MAX30102Sensor(
    simulation=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_max30102() -> bool:
    """Initialize the global MAX30102 sensor."""

    return _max30102.connect()


def read_max30102() -> Dict[str, Any]:
    """Read heart rate and SpO2."""

    return _max30102.read_dict()


def get_max30102_status() -> Dict[str, Any]:
    """Return the current MAX30102 status."""

    return _max30102.get_status()


def reset_max30102() -> None:
    """Reset simulated MAX30102 readings."""

    _max30102.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "MAX30102Reading",
    "MAX30102Sensor",
    "initialize_max30102",
    "read_max30102",
    "get_max30102_status",
    "reset_max30102",
]