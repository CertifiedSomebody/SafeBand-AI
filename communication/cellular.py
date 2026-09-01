"""
SAFEBAND AI - Cellular Communication Module

Prototype interface for the Quectel EC200U cellular module.

For the current software demonstration, cellular communication is
simulated. The interface is designed so that real EC200U UART/AT
commands can be integrated later without changing the dashboard
logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class CellularStatus:
    """Current cellular-module status."""

    connected: bool
    network: str
    signal_strength: int
    module: str
    last_message: str


class CellularManager:
    """
    SAFEBAND AI cellular communication manager.

    Current prototype capabilities:
        - Simulate EC200U network connection
        - Simulate signal strength
        - Simulate data transmission
        - Simulate emergency alert transmission

    Future hardware integration:
        ESP32-S3 -> UART -> EC200U -> Cellular Network
    """

    def __init__(self):
        self.module_name = "Quectel EC200U"
        self.connected = True
        self.network = "Prototype Network"
        self.signal_strength = 82
        self.last_message = "Waiting for transmission..."
        self.message_history = []

    # ---------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------

    def connect(self) -> bool:
        """
        Simulate establishing a cellular connection.

        Returns
        -------
        bool
            True when the connection is established.
        """

        self.connected = True
        self.network = "Prototype LTE Network"
        self.signal_strength = 82
        self.last_message = "Cellular network connected."

        return True

    def disconnect(self) -> None:
        """Simulate cellular disconnection."""

        self.connected = False
        self.network = "Disconnected"
        self.signal_strength = 0
        self.last_message = "Cellular network disconnected."

    # ---------------------------------------------------------
    # SIGNAL
    # ---------------------------------------------------------

    def get_signal_strength(self) -> int:
        """
        Return simulated signal strength.

        Returns
        -------
        int
            Signal strength from 0 to 100.
        """

        if not self.connected:
            return 0

        return self.signal_strength

    # ---------------------------------------------------------
    # DATA TRANSMISSION
    # ---------------------------------------------------------

    def send_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate transmission of data through the EC200U.

        Parameters
        ----------
        data : dict
            Data that would normally be transmitted to the cloud.

        Returns
        -------
        dict
            Transmission status.
        """

        if not self.connected:
            return {
                "success": False,
                "message": "Cellular connection unavailable.",
                "timestamp": datetime.now().isoformat()
            }

        timestamp = datetime.now().isoformat()

        transmission = {
            "type": "DATA",
            "timestamp": timestamp,
            "payload": data
        }

        self.message_history.append(transmission)

        self.last_message = "Sensor data transmitted successfully."

        return {
            "success": True,
            "message": self.last_message,
            "timestamp": timestamp,
            "network": self.network,
            "signal_strength": self.signal_strength
        }

    # ---------------------------------------------------------
    # EMERGENCY ALERT
    # ---------------------------------------------------------

    def send_emergency_alert(
        self,
        alert_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate transmission of an emergency alert.

        Parameters
        ----------
        alert_data : dict
            Emergency information such as activity, risk score
            and GPS location.

        Returns
        -------
        dict
            Emergency transmission status.
        """

        if not self.connected:
            return {
                "success": False,
                "alert_type": "EMERGENCY",
                "message": "Emergency alert failed: cellular connection unavailable.",
                "timestamp": datetime.now().isoformat()
            }

        timestamp = datetime.now().isoformat()

        alert = {
            "type": "EMERGENCY",
            "timestamp": timestamp,
            "payload": alert_data
        }

        self.message_history.append(alert)

        self.last_message = "EMERGENCY ALERT TRANSMITTED."

        return {
            "success": True,
            "alert_type": "EMERGENCY",
            "message": self.last_message,
            "timestamp": timestamp,
            "network": self.network,
            "signal_strength": self.signal_strength
        }

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def get_status(self) -> CellularStatus:
        """Return the current cellular status."""

        return CellularStatus(
            connected=self.connected,
            network=self.network,
            signal_strength=self.signal_strength,
            module=self.module_name,
            last_message=self.last_message
        )

    def get_status_dict(self) -> Dict[str, Any]:
        """Return cellular status as a dictionary."""

        status = self.get_status()

        return {
            "connected": status.connected,
            "network": status.network,
            "signal_strength": status.signal_strength,
            "module": status.module,
            "last_message": status.last_message
        }

    # ---------------------------------------------------------
    # MESSAGE HISTORY
    # ---------------------------------------------------------

    def get_message_history(self) -> list:
        """Return all simulated cellular transmissions."""

        return self.message_history.copy()


# -------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------------------------------------

_cellular_manager = CellularManager()


def get_cellular_status() -> Dict[str, Any]:
    """Return the current simulated EC200U status."""

    return _cellular_manager.get_status_dict()


def transmit_data(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Transmit simulated sensor data through EC200U."""

    return _cellular_manager.send_data(data)


def transmit_emergency_alert(
    alert_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Transmit a simulated emergency alert through EC200U."""

    return _cellular_manager.send_emergency_alert(alert_data)