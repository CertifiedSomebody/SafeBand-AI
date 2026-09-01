"""
SAFEBAND AI - BME680 Environmental Sensor Interface

Prototype interface for the BME680 environmental sensor.

Measures:
- Temperature
- Humidity
- Atmospheric pressure

Current version:
    Uses simulated values for the software demonstration.

Future version:
    Replace the simulation methods with the actual BME680
    I2C sensor driver without changing the rest of the application.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import random


@dataclass
class BME680Reading:
    """Represents one BME680 sensor reading."""

    temperature: float
    humidity: float
    pressure: float
    simulated: bool = True


class BME680Sensor:
    """
    BME680 environmental sensor interface.

    Prototype operation:
        Generates realistic environmental readings.

    Hardware operation:
        Can later be connected to a real BME680 through I2C.
    """

    def __init__(
        self,
        simulation: bool = True
    ):
        self.simulation = simulation
        self.connected = False

        self.temperature = 27.0
        self.humidity = 58.0
        self.pressure = 1012.0

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self) -> bool:
        """
        Initialize the BME680 sensor.

        Returns
        -------
        bool
            True if the sensor is available.
        """

        if self.simulation:
            self.connected = True
            return True

        # Real hardware initialization will be implemented here.
        self.connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect the sensor."""

        self.connected = False

    # =========================================================
    # TEMPERATURE
    # =========================================================

    def read_temperature(self) -> float:
        """Return temperature in degrees Celsius."""

        if not self.connected:
            self.connect()

        if self.simulation:
            self.temperature += random.uniform(
                -0.25,
                0.25
            )

        return round(
            self.temperature,
            1
        )

    # =========================================================
    # HUMIDITY
    # =========================================================

    def read_humidity(self) -> float:
        """Return relative humidity percentage."""

        if not self.connected:
            self.connect()

        if self.simulation:
            self.humidity += random.uniform(
                -0.8,
                0.8
            )

        self.humidity = max(
            0.0,
            min(
                100.0,
                self.humidity
            )
        )

        return round(
            self.humidity,
            1
        )

    # =========================================================
    # PRESSURE
    # =========================================================

    def read_pressure(self) -> float:
        """Return atmospheric pressure in hPa."""

        if not self.connected:
            self.connect()

        if self.simulation:
            self.pressure += random.uniform(
                -1.0,
                1.0
            )

        return round(
            self.pressure,
            1
        )

    # =========================================================
    # COMPLETE READING
    # =========================================================

    def read(self) -> BME680Reading:
        """
        Acquire a complete environmental reading.
        """

        return BME680Reading(
            temperature=self.read_temperature(),
            humidity=self.read_humidity(),
            pressure=self.read_pressure(),
            simulated=self.simulation
        )

    # =========================================================
    # DICTIONARY OUTPUT
    # =========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the sensor reading as a dictionary.

        This format is used by the SAFEBAND AI processing pipeline.
        """

        reading = self.read()

        return {
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "pressure": reading.pressure,
            "simulated": reading.simulated,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current BME680 sensor status."""

        return {
            "sensor": "BME680",
            "name": "Environmental Sensor",
            "connected": self.connected,
            "simulation": self.simulation,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "pressure": self.pressure,
        }


# =============================================================
# GLOBAL SENSOR INSTANCE
# =============================================================

_bme680 = BME680Sensor(
    simulation=True
)


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def initialize_bme680() -> bool:
    """Initialize the BME680 sensor."""

    return _bme680.connect()


def read_bme680() -> Dict[str, Any]:
    """Read all BME680 environmental parameters."""

    return _bme680.read_dict()


def get_bme680_status() -> Dict[str, Any]:
    """Return BME680 status."""

    return _bme680.get_status()