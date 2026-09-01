"""
SAFEBAND AI - GPS Location Sensor Interface

Prototype interface for GPS location tracking.

Provides:
- Latitude
- Longitude
- GPS signal/status
- Location accuracy

Current version:
    Uses simulated coordinates for the software demonstration.

Future version:
    Replace the simulation methods with a real GPS/GNSS receiver
    interface without changing the rest of the SAFEBAND AI pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any
import random


@dataclass
class GPSReading:
    """Represents one GPS location reading."""

    latitude: float
    longitude: float
    accuracy: float
    satellites: int
    connected: bool
    simulated: bool = True


class GPSSensor:
    """
    GPS/GNSS sensor interface.

    Prototype operation:
        Generates simulated GPS coordinates and signal information.

    Hardware operation:
        Can later be connected to a real GPS/GNSS receiver.
    """

    def __init__(
        self,
        simulation: bool = True
    ):
        self.simulation = simulation
        self.connected = False

        self.latitude = 19.0760
        self.longitude = 72.8777

        self.accuracy = 5.0
        self.satellites = 8

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self) -> bool:
        """
        Initialize the GPS receiver.

        Returns
        -------
        bool
            True if GPS is available.
        """

        if self.simulation:
            self.connected = True
            return True

        # Real GPS/GNSS initialization will be implemented here.
        self.connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect the GPS receiver."""

        self.connected = False

    # =========================================================
    # LOCATION
    # =========================================================

    def read_location(self) -> Dict[str, float]:
        """
        Read the current GPS coordinates.

        Returns
        -------
        dict
            Latitude and longitude.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            # Small movement to simulate live location tracking.
            self.latitude += random.uniform(
                -0.00005,
                0.00008
            )

            self.longitude += random.uniform(
                -0.00005,
                0.00008
            )

        return {
            "latitude": round(
                self.latitude,
                6
            ),
            "longitude": round(
                self.longitude,
                6
            ),
        }

    # =========================================================
    # ACCURACY
    # =========================================================

    def read_accuracy(self) -> float:
        """Return estimated GPS accuracy in metres."""

        if not self.connected:
            self.connect()

        if self.simulation:
            self.accuracy += random.uniform(
                -0.5,
                0.5
            )

        self.accuracy = max(
            2.0,
            min(
                20.0,
                self.accuracy
            )
        )

        return round(
            self.accuracy,
            1
        )

    # =========================================================
    # SATELLITES
    # =========================================================

    def read_satellites(self) -> int:
        """Return the simulated number of visible satellites."""

        if not self.connected:
            self.connect()

        if self.simulation:
            self.satellites += random.choice(
                [-1, 0, 0, 0, 1]
            )

        self.satellites = max(
            4,
            min(
                12,
                self.satellites
            )
        )

        return self.satellites

    # =========================================================
    # COMPLETE READING
    # =========================================================

    def read(self) -> GPSReading:
        """
        Acquire a complete GPS reading.
        """

        location = self.read_location()

        return GPSReading(
            latitude=location["latitude"],
            longitude=location["longitude"],
            accuracy=self.read_accuracy(),
            satellites=self.read_satellites(),
            connected=self.connected,
            simulated=self.simulation
        )

    # =========================================================
    # DICTIONARY OUTPUT
    # =========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the complete GPS reading as a dictionary.

        This format is compatible with the SAFEBAND AI
        monitoring and emergency-alert pipeline.
        """

        reading = self.read()

        return {
            "latitude": reading.latitude,
            "longitude": reading.longitude,
            "gps_accuracy": reading.accuracy,
            "satellites": reading.satellites,
            "gps_connected": reading.connected,
            "simulated": reading.simulated,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current GPS status."""

        return {
            "sensor": "GPS",
            "name": "GPS Location",
            "connected": self.connected,
            "simulation": self.simulation,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy": self.accuracy,
            "satellites": self.satellites,
        }


# =============================================================
# GLOBAL SENSOR INSTANCE
# =============================================================

_gps = GPSSensor(
    simulation=True
)


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def initialize_gps() -> bool:
    """Initialize the GPS receiver."""

    return _gps.connect()


def read_gps() -> Dict[str, Any]:
    """Read the current GPS location."""

    return _gps.read_dict()


def get_gps_status() -> Dict[str, Any]:
    """Return GPS status."""

    return _gps.get_status()