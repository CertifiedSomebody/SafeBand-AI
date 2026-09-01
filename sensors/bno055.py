"""
SAFEBAND AI - BNO055 Motion Sensor Interface

Prototype interface for the BNO055 9-DOF absolute orientation
sensor.

Provides:
- Acceleration X/Y/Z
- Motion intensity
- Orientation
- Sensor connection status

Current version:
    Uses simulated values for the software demonstration.

Future version:
    Replace the simulation methods with the actual BNO055 I2C
    driver without changing the rest of the SAFEBAND AI pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any
import math
import random


@dataclass
class BNO055Reading:
    """Represents one BNO055 sensor reading."""

    acceleration_x: float
    acceleration_y: float
    acceleration_z: float
    motion_intensity: float
    orientation: float
    simulated: bool = True


class BNO055Sensor:
    """
    BNO055 motion and orientation sensor interface.

    Prototype operation:
        Generates realistic simulated motion/orientation data.

    Hardware operation:
        Can later be connected to the real BNO055 through I2C.
    """

    def __init__(
        self,
        simulation: bool = True
    ):
        self.simulation = simulation
        self.connected = False

        self.acceleration_x = 0.02
        self.acceleration_y = 0.01
        self.acceleration_z = 1.00

        self.motion_intensity = 0.08
        self.orientation = 5.0

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self) -> bool:
        """
        Initialize the BNO055 sensor.

        Returns
        -------
        bool
            True if the sensor is available.
        """

        if self.simulation:
            self.connected = True
            return True

        # Real BNO055 I2C initialization will be implemented here.
        self.connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect the sensor."""

        self.connected = False

    # =========================================================
    # ACCELERATION
    # =========================================================

    def read_acceleration(self) -> Dict[str, float]:
        """
        Read three-axis acceleration.

        Returns
        -------
        dict
            Acceleration values for X, Y and Z axes.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.acceleration_x += random.uniform(
                -0.05,
                0.05
            )

            self.acceleration_y += random.uniform(
                -0.05,
                0.05
            )

            self.acceleration_z += random.uniform(
                -0.05,
                0.05
            )

        return {
            "acceleration_x": round(
                self.acceleration_x,
                3
            ),
            "acceleration_y": round(
                self.acceleration_y,
                3
            ),
            "acceleration_z": round(
                self.acceleration_z,
                3
            ),
        }

    # =========================================================
    # ACCELERATION MAGNITUDE
    # =========================================================

    def get_acceleration_magnitude(self) -> float:
        """Calculate resultant acceleration magnitude."""

        acceleration = self.read_acceleration()

        magnitude = math.sqrt(
            acceleration["acceleration_x"] ** 2
            + acceleration["acceleration_y"] ** 2
            + acceleration["acceleration_z"] ** 2
        )

        return round(
            magnitude,
            3
        )

    # =========================================================
    # MOTION INTENSITY
    # =========================================================

    def read_motion_intensity(self) -> float:
        """
        Return simulated motion intensity.

        Higher values represent stronger movement.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.motion_intensity += random.uniform(
                -0.04,
                0.04
            )

        self.motion_intensity = max(
            0.0,
            self.motion_intensity
        )

        return round(
            self.motion_intensity,
            2
        )

    # =========================================================
    # ORIENTATION
    # =========================================================

    def read_orientation(self) -> float:
        """
        Return simulated body orientation in degrees.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.orientation += random.uniform(
                -1.5,
                1.5
            )

        self.orientation = max(
            -180.0,
            min(
                180.0,
                self.orientation
            )
        )

        return round(
            self.orientation,
            1
        )

    # =========================================================
    # COMPLETE READING
    # =========================================================

    def read(self) -> BNO055Reading:
        """
        Acquire a complete BNO055 reading.
        """

        acceleration = self.read_acceleration()

        return BNO055Reading(
            acceleration_x=acceleration["acceleration_x"],
            acceleration_y=acceleration["acceleration_y"],
            acceleration_z=acceleration["acceleration_z"],
            motion_intensity=self.read_motion_intensity(),
            orientation=self.read_orientation(),
            simulated=self.simulation
        )

    # =========================================================
    # DICTIONARY OUTPUT
    # =========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the complete sensor reading as a dictionary.

        This format is compatible with the SAFEBAND AI
        activity-recognition and sensor-fusion modules.
        """

        reading = self.read()

        return {
            "acceleration_x": reading.acceleration_x,
            "acceleration_y": reading.acceleration_y,
            "acceleration_z": reading.acceleration_z,
            "motion_intensity": reading.motion_intensity,
            "orientation": reading.orientation,
            "simulated": reading.simulated,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current BNO055 sensor status."""

        return {
            "sensor": "BNO055",
            "name": "Motion & Orientation Sensor",
            "connected": self.connected,
            "simulation": self.simulation,
            "motion_intensity": self.motion_intensity,
            "orientation": self.orientation,
        }


# =============================================================
# GLOBAL SENSOR INSTANCE
# =============================================================

_bno055 = BNO055Sensor(
    simulation=True
)


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def initialize_bno055() -> bool:
    """Initialize the BNO055 sensor."""

    return _bno055.connect()


def read_bno055() -> Dict[str, Any]:
    """Read BNO055 motion and orientation parameters."""

    return _bno055.read_dict()


def get_bno055_status() -> Dict[str, Any]:
    """Return BNO055 status."""

    return _bno055.get_status()