"""
SAFEBAND AI - BME680 Environmental Sensor Interface

Interface for the Bosch BME680 environmental sensor.

Measurements:
    - Environmental temperature
    - Relative humidity
    - Atmospheric pressure

Current implementation:
    Simulated sensor readings.

Future implementation:
    Replace the simulation section with the real BME680
    I2C driver without changing the application-level API.

IMPORTANT
---------
BME680 measures environmental conditions.

Body temperature is NOT provided by this module.
Body temperature is handled separately by MAX30208.
"""


from dataclasses import dataclass
from typing import Any, Dict

import random


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "BME680"
SENSOR_TYPE = "Environmental"

DEFAULT_TEMPERATURE = 27.0
DEFAULT_HUMIDITY = 58.0
DEFAULT_PRESSURE = 1012.0

MIN_TEMPERATURE = -40.0
MAX_TEMPERATURE = 85.0

MIN_HUMIDITY = 0.0
MAX_HUMIDITY = 100.0

MIN_PRESSURE = 300.0
MAX_PRESSURE = 1100.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class BME680Reading:
    """Represents one complete BME680 reading."""

    temperature: float
    humidity: float
    pressure: float
    simulated: bool = True


# ============================================================
# SENSOR CLASS
# ============================================================

class BME680Sensor:
    """
    BME680 environmental sensor interface.

    Responsibilities:
        - Environmental temperature
        - Relative humidity
        - Atmospheric pressure

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

        self.temperature = (
            DEFAULT_TEMPERATURE
        )

        self.humidity = (
            DEFAULT_HUMIDITY
        )

        self.pressure = (
            DEFAULT_PRESSURE
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Initialize the BME680 sensor.

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
        #   BME680 address detection
        #   Sensor configuration
        #   Calibration
        # ----------------------------------------------------

        self.connected = False

        return False

    def disconnect(self) -> None:
        """Disconnect the sensor."""

        self.connected = False

    # ========================================================
    # CONNECTION SAFETY
    # ========================================================

    def _ensure_connected(self) -> None:
        """Ensure the sensor is initialized before reading."""

        if not self.connected:

            self.connect()

    # ========================================================
    # ENVIRONMENTAL TEMPERATURE
    # ========================================================

    def read_temperature(self) -> float:
        """
        Read environmental temperature.

        Returns
        -------
        float
            Temperature in degrees Celsius.
        """

        self._ensure_connected()

        if self.simulation:

            self.temperature += random.uniform(
                -0.25,
                0.25,
            )

        self.temperature = max(
            MIN_TEMPERATURE,
            min(
                MAX_TEMPERATURE,
                self.temperature,
            ),
        )

        return round(
            self.temperature,
            1,
        )

    # ========================================================
    # HUMIDITY
    # ========================================================

    def read_humidity(self) -> float:
        """
        Read relative humidity.

        Returns
        -------
        float
            Relative humidity in percent.
        """

        self._ensure_connected()

        if self.simulation:

            self.humidity += random.uniform(
                -0.8,
                0.8,
            )

        self.humidity = max(
            MIN_HUMIDITY,
            min(
                MAX_HUMIDITY,
                self.humidity,
            ),
        )

        return round(
            self.humidity,
            1,
        )

    # ========================================================
    # ATMOSPHERIC PRESSURE
    # ========================================================

    def read_pressure(self) -> float:
        """
        Read atmospheric pressure.

        Returns
        -------
        float
            Atmospheric pressure in hPa.
        """

        self._ensure_connected()

        if self.simulation:

            self.pressure += random.uniform(
                -1.0,
                1.0,
            )

        self.pressure = max(
            MIN_PRESSURE,
            min(
                MAX_PRESSURE,
                self.pressure,
            ),
        )

        return round(
            self.pressure,
            1,
        )

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> BME680Reading:
        """
        Acquire one complete environmental reading.
        """

        self._ensure_connected()

        return BME680Reading(
            temperature=self.read_temperature(),
            humidity=self.read_humidity(),
            pressure=self.read_pressure(),
            simulated=self.simulation,
        )

    # ========================================================
    # DICTIONARY OUTPUT
    # ========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the complete BME680 reading as a dictionary.

        This format is consumed by the SAFEBAND sensor pipeline.
        """

        reading = self.read()

        return {
            "temperature": (
                reading.temperature
            ),

            "humidity": (
                reading.humidity
            ),

            "pressure": (
                reading.pressure
            ),

            "simulated": (
                reading.simulated
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current BME680 status."""

        return {
            "sensor": SENSOR_NAME,

            "name": (
                "Environmental Sensor"
            ),

            "type": SENSOR_TYPE,

            "connected": (
                self.connected
            ),

            "simulation": (
                self.simulation
            ),

            "temperature": (
                round(
                    self.temperature,
                    1,
                )
            ),

            "humidity": (
                round(
                    self.humidity,
                    1,
                )
            ),

            "pressure": (
                round(
                    self.pressure,
                    1,
                )
            ),
        }


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

_bme680 = BME680Sensor(
    simulation=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_bme680() -> bool:
    """Initialize the global BME680 sensor."""

    return _bme680.connect()


def read_bme680() -> Dict[str, Any]:
    """Read all BME680 environmental parameters."""

    return _bme680.read_dict()


def get_bme680_status() -> Dict[str, Any]:
    """Return the current BME680 status."""

    return _bme680.get_status()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "BME680Reading",
    "BME680Sensor",
    "initialize_bme680",
    "read_bme680",
    "get_bme680_status",
]