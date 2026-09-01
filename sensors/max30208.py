"""
SAFEBAND AI - MAX30208 Body Temperature Sensor Interface

Interface for the MAX30208 digital body-temperature sensor.

Responsibilities:
    - Body-temperature measurement
    - Sensor availability
    - I2C configuration/status

The BME680 remains responsible for environmental measurements.

Current implementation:
    Simulated body-temperature readings.

Future implementation:
    Replace the isolated hardware-read method with the finalized
    MAX30208 I2C register implementation for the selected
    hardware platform.

IMPORTANT
---------
The MAX30208 value is exposed as `body_temperature`.

It must not be confused with the BME680 environmental
`temperature` value.
"""


from dataclasses import dataclass
from typing import Any, Dict, Optional

import random
import time


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "MAX30208"
SENSOR_TYPE = "BODY_TEMPERATURE"
INTERFACE = "I2C"

DEFAULT_I2C_ADDRESS = 0x50
DEFAULT_BUS = 1

DEFAULT_BODY_TEMPERATURE = 36.7

MIN_VALID_TEMPERATURE = 25.0
MAX_VALID_TEMPERATURE = 45.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class MAX30208Reading:
    """Represents one MAX30208 body-temperature reading."""

    body_temperature: Optional[float]
    connected: bool
    available: bool
    simulated: bool
    timestamp: float


# ============================================================
# SENSOR CLASS
# ============================================================

class MAX30208:
    """
    MAX30208 body-temperature sensor interface.

    Responsibilities:
        - Body-temperature acquisition
        - Sensor initialization
        - Sensor availability
        - I2C status

    Simulation mode provides realistic readings for the
    SAFEBAND software prototype.

    Hardware-specific communication is isolated inside
    `_read_hardware()` so the rest of the application does not
    depend on register-level implementation details.
    """

    SENSOR_NAME = SENSOR_NAME

    def __init__(
        self,
        bus: int = DEFAULT_BUS,
        address: int = DEFAULT_I2C_ADDRESS,
        simulated: bool = True,
    ) -> None:

        self.bus = int(bus)

        self.address = int(address)

        self.simulated = bool(
            simulated
        )

        self.connected = False
        self.initialized = False

        self.last_temperature: Optional[
            float
        ] = None

        self.last_read_time: Optional[
            float
        ] = None

        self._i2c = None

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def initialize(self) -> bool:
        """
        Initialize the MAX30208 sensor.

        Returns
        -------
        bool
            True when initialization succeeds.
        """

        if self.initialized:
            return self.connected

        if self.simulated:

            self.connected = True
            self.initialized = True

            return True

        try:

            from smbus2 import SMBus

            self._i2c = SMBus(
                self.bus
            )

            # ------------------------------------------------
            # Hardware-specific MAX30208 configuration will be
            # added once the final host platform and sensor
            # board configuration are fixed.
            # ------------------------------------------------

            self.connected = True
            self.initialized = True

            return True

        except Exception:

            self.connected = False
            self.initialized = False
            self._i2c = None

            return False

    # ========================================================
    # CONNECTION STATUS
    # ========================================================

    def is_connected(self) -> bool:
        """Return whether the sensor is currently connected."""

        return self.connected

    def is_available(self) -> bool:
        """
        Return whether the sensor is initialized and available.
        """

        return (
            self.initialized
            and self.connected
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return the current MAX30208 sensor status."""

        return {
            "sensor": self.SENSOR_NAME,

            "name": (
                "Body Temperature Sensor"
            ),

            "type": SENSOR_TYPE,

            "interface": INTERFACE,

            "bus": self.bus,

            "address": hex(
                self.address
            ),

            "simulated": (
                self.simulated
            ),

            "connected": (
                self.connected
            ),

            "initialized": (
                self.initialized
            ),

            "available": (
                self.is_available()
            ),

            "last_temperature": (
                self.last_temperature
            ),

            "last_read_time": (
                self.last_read_time
            ),
        }

    # ========================================================
    # SIMULATION
    # ========================================================

    def _simulate_temperature(self) -> float:
        """
        Generate a realistic simulated body-temperature value.

        A small random walk is used so consecutive readings
        change gradually rather than behaving as independent
        random samples.
        """

        if self.last_temperature is None:

            base_temperature = (
                DEFAULT_BODY_TEMPERATURE
            )

        else:

            base_temperature = (
                self.last_temperature
            )

        variation = random.uniform(
            -0.08,
            0.08,
        )

        temperature = (
            base_temperature
            + variation
        )

        temperature = max(
            MIN_VALID_TEMPERATURE,
            min(
                MAX_VALID_TEMPERATURE,
                temperature,
            ),
        )

        return round(
            temperature,
            2,
        )

    # ========================================================
    # TEMPERATURE READING
    # ========================================================

    def read_temperature(
        self,
    ) -> Optional[float]:
        """
        Read body temperature in degrees Celsius.

        Returns
        -------
        float or None
            Body temperature when available.

        None
            If the sensor is unavailable or the hardware read
            fails.
        """

        if not self.initialized:

            if not self.initialize():

                return None

        if not self.connected:
            return None

        try:

            if self.simulated:

                temperature = (
                    self._simulate_temperature()
                )

            else:

                temperature = (
                    self._read_hardware()
                )

            if not (
                MIN_VALID_TEMPERATURE
                <= temperature
                <= MAX_VALID_TEMPERATURE
            ):
                return None

            self.last_temperature = (
                round(
                    float(temperature),
                    2,
                )
            )

            self.last_read_time = (
                time.time()
            )

            return self.last_temperature

        except (
            TypeError,
            ValueError,
            RuntimeError,
            OSError,
        ):

            return None

    # ========================================================
    # HARDWARE READING
    # ========================================================

    def _read_hardware(self) -> float:
        """
        Read body temperature from the physical MAX30208.

        Hardware-specific register access intentionally remains
        isolated here.

        The finalized implementation will depend on the exact
        MAX30208 board, host platform and I2C configuration.
        """

        if self._i2c is None:

            raise RuntimeError(
                "MAX30208 I2C interface "
                "is not initialized."
            )

        raise NotImplementedError(
            "MAX30208 hardware register "
            "implementation is not configured."
        )

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> MAX30208Reading:
        """
        Acquire one complete MAX30208 reading.
        """

        temperature = (
            self.read_temperature()
        )

        return MAX30208Reading(
            body_temperature=(
                temperature
            ),

            connected=(
                self.connected
            ),

            available=(
                temperature is not None
            ),

            simulated=(
                self.simulated
            ),

            timestamp=time.time(),
        )

    # ========================================================
    # DICTIONARY OUTPUT
    # ========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return a standardized MAX30208 sensor record.

        `body_temperature` is intentionally kept separate from
        the BME680 `temperature` field.
        """

        reading = self.read()

        return {
            "sensor": self.SENSOR_NAME,

            "body_temperature": (
                reading.body_temperature
            ),

            "unit": "°C",

            "timestamp": (
                reading.timestamp
            ),

            "connected": (
                reading.connected
            ),

            "available": (
                reading.available
            ),

            "simulated": (
                reading.simulated
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset the software state of the sensor."""

        self.last_temperature = None
        self.last_read_time = None

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """Close the I2C interface when hardware mode is active."""

        if self._i2c is not None:

            try:
                self._i2c.close()

            except Exception:
                pass

            finally:
                self._i2c = None

        self.connected = False
        self.initialized = False


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

max30208 = MAX30208(
    simulated=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_sensor() -> bool:
    """Initialize the global MAX30208 sensor."""

    return max30208.initialize()


def read_body_temperature() -> Optional[float]:
    """Read body temperature from the global MAX30208."""

    return max30208.read_temperature()


def read_sensor() -> Dict[str, Any]:
    """Return a standardized MAX30208 sensor reading."""

    return max30208.read_dict()


def get_sensor_status() -> Dict[str, Any]:
    """Return the global MAX30208 sensor status."""

    return max30208.get_status()


def reset_sensor() -> None:
    """Reset the global MAX30208 sensor."""

    max30208.reset()


def close_sensor() -> None:
    """Close the global MAX30208 sensor."""

    max30208.close()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "MAX30208Reading",
    "MAX30208",
    "initialize_sensor",
    "read_body_temperature",
    "read_sensor",
    "get_sensor_status",
    "reset_sensor",
    "close_sensor",
]