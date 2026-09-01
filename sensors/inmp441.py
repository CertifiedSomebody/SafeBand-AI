"""
SAFEBAND AI - INMP441 Audio Sensor Interface

Interface for the INMP441 MEMS microphone.

Measurements:
    - Normalized audio level
    - Acoustic state
    - Microphone connection status

Current implementation:
    Simulated audio-level readings.

Future implementation:
    Replace the simulation section with actual I2S microphone
    acquisition and signal-processing logic.

IMPORTANT
---------
This module is responsible for acquiring audio-related sensor
information.

Advanced acoustic-event recognition should eventually be handled
by the AI/TinyML layer rather than by the hardware interface.
"""


from dataclasses import dataclass
from typing import Any, Dict

import random


# ============================================================
# CONSTANTS
# ============================================================

SENSOR_NAME = "INMP441"
SENSOR_TYPE = "Audio"

INTERFACE = "I2S"

DEFAULT_AUDIO_LEVEL = 0.08

MIN_AUDIO_LEVEL = 0.0
MAX_AUDIO_LEVEL = 1.0

QUIET_THRESHOLD = 0.20
NORMAL_THRESHOLD = 0.50
LOUD_THRESHOLD = 0.75


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class INMP441Reading:
    """Represents one complete INMP441 microphone reading."""

    audio_level: float
    acoustic_state: str
    connected: bool
    simulated: bool = True


# ============================================================
# SENSOR CLASS
# ============================================================

class INMP441Sensor:
    """
    INMP441 MEMS microphone interface.

    Responsibilities:
        - Audio-level acquisition
        - Basic acoustic-level categorization
        - Microphone connection state

    Audio-level range:

        0.00 -> Very quiet
        1.00 -> Very loud

    The class supports simulated operation now and provides a
    stable interface for future I2S hardware integration.
    """

    def __init__(
        self,
        simulation: bool = True,
    ) -> None:

        self.simulation = bool(
            simulation
        )

        self.connected = False

        self.audio_level = (
            DEFAULT_AUDIO_LEVEL
        )

        self.acoustic_state = (
            "QUIET"
        )

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Initialize the INMP441 microphone.

        Returns
        -------
        bool
            True when the microphone is available.
        """

        if self.simulation:

            self.connected = True

            return True

        # ----------------------------------------------------
        # Future hardware implementation:
        #
        #   ESP32-S3 I2S initialization
        #   INMP441 channel configuration
        #   Sample-rate configuration
        #   Audio-buffer initialization
        # ----------------------------------------------------

        self.connected = False

        return False

    def disconnect(self) -> None:
        """Disconnect the microphone."""

        self.connected = False

    def _ensure_connected(self) -> None:
        """Ensure the microphone is initialized before reading."""

        if not self.connected:
            self.connect()

    # ========================================================
    # AUDIO LEVEL
    # ========================================================

    def read_audio_level(self) -> float:
        """
        Read the normalized audio level.

        Returns
        -------
        float
            Audio level between 0.00 and 1.00.
        """

        self._ensure_connected()

        if self.simulation:

            self.audio_level += random.uniform(
                -0.04,
                0.04,
            )

        self.audio_level = max(
            MIN_AUDIO_LEVEL,
            min(
                MAX_AUDIO_LEVEL,
                self.audio_level,
            ),
        )

        return round(
            self.audio_level,
            2,
        )

    # ========================================================
    # BASIC ACOUSTIC LEVEL
    # ========================================================

    @staticmethod
    def classify_audio(
        audio_level: float,
    ) -> str:
        """
        Categorize the current acoustic level.

        Classification:

            0.00 - 0.20 : QUIET
            0.21 - 0.50 : NORMAL
            0.51 - 0.75 : LOUD
            0.76 - 1.00 : VERY LOUD

        This is a basic level categorization, not AI-based
        acoustic-event recognition.
        """

        try:
            level = float(
                audio_level
            )

        except (TypeError, ValueError):
            level = 0.0

        level = max(
            MIN_AUDIO_LEVEL,
            min(
                MAX_AUDIO_LEVEL,
                level,
            ),
        )

        if level <= QUIET_THRESHOLD:
            return "QUIET"

        if level <= NORMAL_THRESHOLD:
            return "NORMAL"

        if level <= LOUD_THRESHOLD:
            return "LOUD"

        return "VERY LOUD"

    # ========================================================
    # COMPLETE READING
    # ========================================================

    def read(self) -> INMP441Reading:
        """
        Acquire one complete microphone reading.
        """

        self._ensure_connected()

        audio_level = (
            self.read_audio_level()
        )

        self.acoustic_state = (
            self.classify_audio(
                audio_level
            )
        )

        return INMP441Reading(
            audio_level=audio_level,
            acoustic_state=(
                self.acoustic_state
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
        Return the complete microphone reading as a dictionary.

        This format is consumed by the SAFEBAND sensor and
        processing pipeline.
        """

        reading = self.read()

        return {
            "audio_level": (
                reading.audio_level
            ),

            "acoustic_state": (
                reading.acoustic_state
            ),

            "microphone_connected": (
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
        """Return the current INMP441 status."""

        return {
            "sensor": SENSOR_NAME,

            "name": (
                "MEMS Microphone"
            ),

            "type": SENSOR_TYPE,

            "interface": INTERFACE,

            "connected": (
                self.connected
            ),

            "simulation": (
                self.simulation
            ),

            "audio_level": round(
                self.audio_level,
                2,
            ),

            "acoustic_state": (
                self.acoustic_state
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset simulated microphone readings."""

        self.audio_level = (
            DEFAULT_AUDIO_LEVEL
        )

        self.acoustic_state = (
            "QUIET"
        )


# ============================================================
# GLOBAL SENSOR INSTANCE
# ============================================================

_inmp441 = INMP441Sensor(
    simulation=True
)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def initialize_inmp441() -> bool:
    """Initialize the global INMP441 microphone."""

    return _inmp441.connect()


def read_inmp441() -> Dict[str, Any]:
    """Read the current microphone data."""

    return _inmp441.read_dict()


def get_inmp441_status() -> Dict[str, Any]:
    """Return the current INMP441 status."""

    return _inmp441.get_status()


def reset_inmp441() -> None:
    """Reset simulated microphone readings."""

    _inmp441.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "INMP441Reading",
    "INMP441Sensor",
    "initialize_inmp441",
    "read_inmp441",
    "get_inmp441_status",
    "reset_inmp441",
]