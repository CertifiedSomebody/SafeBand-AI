"""
SAFEBAND AI - GPS/GNSS Location Sensor Interface

Interface for GPS/GNSS location tracking.

Measurements:
    - Latitude
    - Longitude
    - Location accuracy
    - Visible satellites
    - Connection status

Current implementation:
    Simulated GPS/GNSS readings.

Future implementation:
    Replace the simulation section with the actual GPS/GNSS
    receiver interface without changing the application-level API.

IMPORTANT
---------
GPS is responsible only for location information.

It does not perform activity recognition, risk assessment,
or emergency decisions.
"""


from dataclasses import dataclass
from typing import Any, Dict

import random


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "GPS"
SENSOR_TYPE = "Location"

DEFAULT_LATITUDE = 19.0760
DEFAULT_LONGITUDE = 72.8777

DEFAULT_ACCURACY = 5.0
DEFAULT_SATELLITES = 8

MIN_ACCURACY = 2.0
MAX_ACCURACY = 20.0

MIN_SATELLITES = 0
MAX_SATELLITES = 20


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class GPSReading:
    """Represents one complete GPS/GNSS reading."""

    latitude: float
    longitude: float
    accuracy: float
    satellites: int
    connected: bool
    simulated: bool = True


# ============================================================
# SENSOR CLASS
# ============================================================

class GPSSensor:
    """
    GPS/GNSS sensor interface.

    Responsibilities:
        - Geographic coordinates
        - Location accuracy
        - Satellite visibility
        - Receiver connection state

    The class supports simulated operation now and provides a
    stable interface for future hardware integration.
    """

    def __init__(
        self,
        simulation: bool = True,
    ) -> None:

        self.simulation = bool(
            simulation
        )

        self.connected = False

        self.latitude = (
            DEFAULT_LATITUDE
        )

        self.longitude = (
            DEFAULT_LONGITUDE
        )

        self.accuracy = (
            DEFAULT_ACCURACY
        )

        self.satellites = (
            DEFAULT_SATELLITES
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Initialize the GPS/GNSS receiver.

        Returns
        -------
        bool
            True when the receiver is available.
        """

        if self.simulation:

            self.connected = True

            return True

        # ----------------------------------------------------
        # Future hardware implementation:
        #
        #   UART initialization
        #   GNSS receiver configuration
        #   NMEA/UBX parsing
        #   Satellite acquisition
        # ----------------------------------------------------

        self.connected = False

        return False

    def disconnect(self) -> None:
        """Disconnect the GPS/GNSS receiver."""

        self.connected = False

    def _ensure_connected(self) -> None:
        """Ensure the receiver is initialized before reading."""

        if not self.connected:
            self.connect()

    # ========================================================
    # LOCATION
    # ========================================================

    def read_location(self) -> Dict[str, float]:
        """
        Read the current GPS/GNSS coordinates.

        Returns
        -------
        dict
            Latitude and longitude in decimal degrees.
        """

        self._ensure_connected()

        if self.simulation:

            # Small coordinate changes simulate movement.
            self.latitude += random.uniform(
                -0.00005,
                0.00008,
            )

            self.longitude += random.uniform(
                -0.00005,
                0.00008,
            )

        return {
            "latitude": round(
                self.latitude,
                6,
            ),

            "longitude": round(
                self.longitude,
                6,
            ),
        }

    # ========================================================
    # ACCURACY
    # ========================================================

    def read_accuracy(self) -> float:
        """
        Read estimated horizontal location accuracy.

        Returns
        -------
        float
            Estimated accuracy in metres.
        """

        self._ensure_connected()

        if self.simulation:

            self.accuracy += random.uniform(
                -0.5,
                0.5,
            )

        self.accuracy = max(
            MIN_ACCURACY,
            min(
                MAX_ACCURACY,
                self.accuracy,
            ),
        )

        return round(
            self.accuracy,
            1,
        )

    # ========================================================
    # SATELLITES
    # ========================================================

    def read_satellites(self) -> int:
        """
        Read the number of visible/usable satellites.

        Returns
        -------
        int
            Number of satellites.
        """

        self._ensure_connected()

        if self.simulation:

            self.satellites += random.choice(
                [-1, 0, 0, 0, 1]
            )

        self.satellites = max(
            MIN_SATELLITES,
            min(
                MAX_SATELLITES,
                self.satellites,
            ),
        )

        return int(
            self.satellites
        )

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> GPSReading:
        """
        Acquire one complete GPS/GNSS reading.
        """

        self._ensure_connected()

        location = (
            self.read_location()
        )

        return GPSReading(
            latitude=location[
                "latitude"
            ],

            longitude=location[
                "longitude"
            ],

            accuracy=(
                self.read_accuracy()
            ),

            satellites=(
                self.read_satellites()
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
        Return the complete GPS reading as a dictionary.

        This format is consumed by the SAFEBAND monitoring,
        sensor-fusion and emergency-alert pipeline.
        """

        reading = self.read()

        return {
            "latitude": (
                reading.latitude
            ),

            "longitude": (
                reading.longitude
            ),

            "gps_accuracy": (
                reading.accuracy
            ),

            "satellites": (
                reading.satellites
            ),

            "gps_connected": (
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
        """Return the current GPS/GNSS status."""

        return {
            "sensor": SENSOR_NAME,

            "name": (
                "GPS Location"
            ),

            "type": SENSOR_TYPE,

            "connected": (
                self.connected
            ),

            "simulation": (
                self.simulation
            ),

            "latitude": round(
                self.latitude,
                6,
            ),

            "longitude": round(
                self.longitude,
                6,
            ),

            "accuracy": round(
                self.accuracy,
                1,
            ),

            "satellites": int(
                self.satellites
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset simulated GPS values to their defaults."""

        self.latitude = (
            DEFAULT_LATITUDE
        )

        self.longitude = (
            DEFAULT_LONGITUDE
        )

        self.accuracy = (
            DEFAULT_ACCURACY
        )

        self.satellites = (
            DEFAULT_SATELLITES
        )


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

_gps = GPSSensor(
    simulation=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_gps() -> bool:
    """Initialize the global GPS/GNSS receiver."""

    return _gps.connect()


def read_gps() -> Dict[str, Any]:
    """Read the current GPS/GNSS location."""

    return _gps.read_dict()


def get_gps_status() -> Dict[str, Any]:
    """Return the current GPS/GNSS status."""

    return _gps.get_status()


def reset_gps() -> None:
    """Reset simulated GPS readings."""

    _gps.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "GPSReading",
    "GPSSensor",
    "initialize_gps",
    "read_gps",
    "get_gps_status",
    "reset_gps",
]