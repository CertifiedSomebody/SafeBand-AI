"""
SAFEBAND AI - BNO055 Motion Sensor Interface

Interface for the BNO055 9-DOF absolute orientation sensor.

Measurements:
    - Acceleration X/Y/Z
    - Resultant acceleration magnitude
    - Motion intensity
    - Body orientation

Current implementation:
    Simulated sensor readings.

Future implementation:
    Replace the simulation section with the real BNO055
    I2C driver without changing the application-level API.

IMPORTANT
---------
This module is responsible only for motion/orientation sensing.
Activity classification belongs to the AI/activity-recognition
layer.
"""


from dataclasses import dataclass
from typing import Any, Dict

import math
import random


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "BNO055"
SENSOR_TYPE = "Motion & Orientation"

DEFAULT_ACCELERATION_X = 0.02
DEFAULT_ACCELERATION_Y = 0.01
DEFAULT_ACCELERATION_Z = 1.00

DEFAULT_MOTION_INTENSITY = 0.08
DEFAULT_ORIENTATION = 5.0

MIN_MOTION_INTENSITY = 0.0
MAX_MOTION_INTENSITY = 10.0

MIN_ORIENTATION = -180.0
MAX_ORIENTATION = 180.0


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class BNO055Reading:
    """Represents one complete BNO055 reading."""

    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    motion_intensity: float
    orientation: float
    simulated: bool = True


# ============================================================
# SENSOR CLASS
# ============================================================

class BNO055Sensor:
    """
    BNO055 motion and orientation sensor interface.

    Responsibilities:
        - Three-axis acceleration
        - Resultant acceleration
        - Motion intensity
        - Body orientation

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

        self.acceleration_x = (
            DEFAULT_ACCELERATION_X
        )

        self.acceleration_y = (
            DEFAULT_ACCELERATION_Y
        )

        self.acceleration_z = (
            DEFAULT_ACCELERATION_Z
        )

        self.motion_intensity = (
            DEFAULT_MOTION_INTENSITY
        )

        self.orientation = (
            DEFAULT_ORIENTATION
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Initialize the BNO055 sensor.

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
        #   BNO055 address detection
        #   Operating-mode configuration
        #   Sensor calibration
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
    # ACCELERATION
    # ========================================================

    def read_acceleration(self) -> Dict[str, float]:
        """
        Read three-axis acceleration.

        Returns
        -------
        dict
            Acceleration values for X, Y and Z axes.
        """

        self._ensure_connected()

        if self.simulation:

            self.acceleration_x += random.uniform(
                -0.05,
                0.05,
            )

            self.acceleration_y += random.uniform(
                -0.05,
                0.05,
            )

            self.acceleration_z += random.uniform(
                -0.05,
                0.05,
            )

        return {
            "acceleration_x": round(
                self.acceleration_x,
                3,
            ),
            "acceleration_y": round(
                self.acceleration_y,
                3,
            ),
            "acceleration_z": round(
                self.acceleration_z,
                3,
            ),
        }

    # ========================================================
    # ACCELERATION MAGNITUDE
    # ========================================================

    def get_acceleration_magnitude(self) -> float:
        """
        Calculate resultant acceleration magnitude.

        Formula:

            A = sqrt(Ax² + Ay² + Az²)
        """

        acceleration = (
            self.read_acceleration()
        )

        magnitude = math.sqrt(
            acceleration["acceleration_x"] ** 2
            + acceleration["acceleration_y"] ** 2
            + acceleration["acceleration_z"] ** 2
        )

        return round(
            magnitude,
            3,
        )

    # ========================================================
    # MOTION INTENSITY
    # ========================================================

    def read_motion_intensity(self) -> float:
        """
        Read motion intensity.

        Higher values represent stronger movement.

        Note:
            This is currently a simulated/derived prototype
            value. A future implementation can derive it from
            a window of BNO055 acceleration samples.
        """

        self._ensure_connected()

        if self.simulation:

            self.motion_intensity += random.uniform(
                -0.04,
                0.04,
            )

        self.motion_intensity = max(
            MIN_MOTION_INTENSITY,
            min(
                MAX_MOTION_INTENSITY,
                self.motion_intensity,
            ),
        )

        return round(
            self.motion_intensity,
            2,
        )

    # ========================================================
    # ORIENTATION
    # ========================================================

    def read_orientation(self) -> float:
        """
        Read body orientation in degrees.

        Range:
            -180° to +180°
        """

        self._ensure_connected()

        if self.simulation:

            self.orientation += random.uniform(
                -1.5,
                1.5,
            )

        self.orientation = max(
            MIN_ORIENTATION,
            min(
                MAX_ORIENTATION,
                self.orientation,
            ),
        )

        return round(
            self.orientation,
            1,
        )

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> BNO055Reading:
        """
        Acquire one complete BNO055 sensor reading.
        """

        self._ensure_connected()

        acceleration = (
            self.read_acceleration()
        )

        return BNO055Reading(
            acceleration_x=acceleration[
                "acceleration_x"
            ],
            acceleration_y=acceleration[
                "acceleration_y"
            ],
            acceleration_z=acceleration[
                "acceleration_z"
            ],
            motion_intensity=(
                self.read_motion_intensity()
            ),
            orientation=(
                self.read_orientation()
            ),
            simulated=self.simulation,
        )

    # ========================================================
    # DICTIONARY OUTPUT
    # ========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the complete sensor reading as a dictionary.

        This format is consumed by the SAFEBAND sensor,
        activity-recognition and AI processing pipeline.
        """

        reading = self.read()

        return {
            "acceleration_x": (
                reading.acceleration_x
            ),
            "acceleration_y": (
                reading.acceleration_y
            ),
            "acceleration_z": (
                reading.acceleration_z
            ),
            "motion_intensity": (
                reading.motion_intensity
            ),
            "orientation": (
                reading.orientation
            ),
            "simulated": (
                reading.simulated
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return the current BNO055 sensor status."""

        return {
            "sensor": SENSOR_NAME,

            "name": (
                "Motion & Orientation Sensor"
            ),

            "type": SENSOR_TYPE,

            "connected": (
                self.connected
            ),

            "simulation": (
                self.simulation
            ),

            "motion_intensity": round(
                self.motion_intensity,
                2,
            ),

            "orientation": round(
                self.orientation,
                1,
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset simulated readings to their defaults."""

        self.acceleration_x = (
            DEFAULT_ACCELERATION_X
        )

        self.acceleration_y = (
            DEFAULT_ACCELERATION_Y
        )

        self.acceleration_z = (
            DEFAULT_ACCELERATION_Z
        )

        self.motion_intensity = (
            DEFAULT_MOTION_INTENSITY
        )

        self.orientation = (
            DEFAULT_ORIENTATION
        )


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

_bno055 = BNO055Sensor(
    simulation=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_bno055() -> bool:
    """Initialize the global BNO055 sensor."""

    return _bno055.connect()


def read_bno055() -> Dict[str, Any]:
    """Read BNO055 motion and orientation parameters."""

    return _bno055.read_dict()


def get_bno055_status() -> Dict[str, Any]:
    """Return the current BNO055 status."""

    return _bno055.get_status()


def reset_bno055() -> None:
    """Reset the simulated BNO055 readings."""

    _bno055.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "BNO055Reading",
    "BNO055Sensor",
    "initialize_bno055",
    "read_bno055",
    "get_bno055_status",
    "reset_bno055",
]