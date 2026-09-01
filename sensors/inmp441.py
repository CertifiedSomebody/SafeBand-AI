"""
SAFEBAND AI - INMP441 Audio Sensor Interface

Prototype interface for the INMP441 MEMS microphone.

Provides:
- Audio level
- Acoustic activity classification
- Microphone connection status

Current version:
    Uses simulated audio-level data for the software demonstration.

Future version:
    Replace the simulation methods with actual I2S microphone
    acquisition and signal processing without changing the rest
    of the SAFEBAND AI pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any
import random


@dataclass
class INMP441Reading:
    """Represents one INMP441 microphone reading."""

    audio_level: float
    acoustic_state: str
    connected: bool
    simulated: bool = True


class INMP441Sensor:
    """
    INMP441 MEMS microphone interface.

    Prototype operation:
        Generates simulated normalized audio levels.

    Audio level range:
        0.00 -> Very quiet
        1.00 -> Very loud

    Hardware operation:
        Can later be connected to the real INMP441 through I2S.
    """

    def __init__(
        self,
        simulation: bool = True
    ):
        self.simulation = simulation
        self.connected = False

        self.audio_level = 0.08
        self.acoustic_state = "QUIET"

    # =========================================================
    # CONNECTION
    # =========================================================

    def connect(self) -> bool:
        """
        Initialize the INMP441 microphone.

        Returns
        -------
        bool
            True if the microphone is available.
        """

        if self.simulation:
            self.connected = True
            return True

        # Real INMP441 I2S initialization will be implemented here.
        self.connected = False
        return False

    def disconnect(self) -> None:
        """Disconnect the microphone."""

        self.connected = False

    # =========================================================
    # AUDIO LEVEL
    # =========================================================

    def read_audio_level(self) -> float:
        """
        Read normalized simulated audio level.

        Returns
        -------
        float
            Audio level between 0.00 and 1.00.
        """

        if not self.connected:
            self.connect()

        if self.simulation:
            self.audio_level += random.uniform(
                -0.04,
                0.04
            )

        self.audio_level = max(
            0.0,
            min(
                1.0,
                self.audio_level
            )
        )

        return round(
            self.audio_level,
            2
        )

    # =========================================================
    # ACOUSTIC CLASSIFICATION
    # =========================================================

    def classify_audio(
        self,
        audio_level: float
    ) -> str:
        """
        Classify the current acoustic environment.

        Classification:
            0.00 - 0.20 : QUIET
            0.21 - 0.50 : NORMAL
            0.51 - 0.75 : LOUD
            0.76 - 1.00 : VERY LOUD
        """

        if audio_level <= 0.20:
            return "QUIET"

        if audio_level <= 0.50:
            return "NORMAL"

        if audio_level <= 0.75:
            return "LOUD"

        return "VERY LOUD"

    # =========================================================
    # COMPLETE READING
    # =========================================================

    def read(self) -> INMP441Reading:
        """
        Acquire a complete microphone reading.
        """

        audio_level = self.read_audio_level()

        self.acoustic_state = self.classify_audio(
            audio_level
        )

        return INMP441Reading(
            audio_level=audio_level,
            acoustic_state=self.acoustic_state,
            connected=self.connected,
            simulated=self.simulation
        )

    # =========================================================
    # DICTIONARY OUTPUT
    # =========================================================

    def read_dict(self) -> Dict[str, Any]:
        """
        Return the microphone reading as a dictionary.

        This format is compatible with the SAFEBAND AI
        sensor-fusion pipeline.
        """

        reading = self.read()

        return {
            "audio_level": reading.audio_level,
            "acoustic_state": reading.acoustic_state,
            "microphone_connected": reading.connected,
            "simulated": reading.simulated,
        }

    # =========================================================
    # STATUS
    # =========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current INMP441 microphone status."""

        return {
            "sensor": "INMP441",
            "name": "MEMS Microphone",
            "interface": "I2S",
            "connected": self.connected,
            "simulation": self.simulation,
            "audio_level": self.audio_level,
            "acoustic_state": self.acoustic_state,
        }


# =============================================================
# GLOBAL SENSOR INSTANCE
# =============================================================

_inmp441 = INMP441Sensor(
    simulation=True
)


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def initialize_inmp441() -> bool:
    """Initialize the INMP441 microphone."""

    return _inmp441.connect()


def read_inmp441() -> Dict[str, Any]:
    """Read the current microphone data."""

    return _inmp441.read_dict()


def get_inmp441_status() -> Dict[str, Any]:
    """Return INMP441 status."""

    return _inmp441.get_status()