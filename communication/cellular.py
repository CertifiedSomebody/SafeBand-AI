"""
SAFEBAND AI - Cellular Communication Module

Communication abstraction for the Quectel EC200U cellular module.

Current mode:
    Simulated cellular communication for software development.

Future mode:
    ESP32-S3 UART communication with the physical EC200U using
    modem/AT commands.

Important prototype behaviour:
    Emergency alerts are idempotent.

    The same emergency event will not be transmitted repeatedly
    when the Streamlit application reruns. A new event must have
    a different event_id, or the caller may explicitly use
    force=True for testing.

The public API remains independent of the transport implementation
so the dashboard and application layers do not need to know whether
communication is simulated or hardware-backed.
"""


from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODULE_NAME = "Quectel EC200U"

DEFAULT_NETWORK = "Prototype LTE Network"

DEFAULT_SIGNAL_STRENGTH = 82

MIN_SIGNAL_STRENGTH = 0

MAX_SIGNAL_STRENGTH = 100

# Prevent accidental duplicate emergency transmissions.
#
# This is not a time-based cooldown. Instead, the manager keeps
# track of the current emergency event identity.
#
# A new event_id creates a new transmission.
# The same event_id is ignored on subsequent calls.
EMERGENCY_DEDUPLICATION_ENABLED = True


# ============================================================
# STATUS
# ============================================================

@dataclass
class CellularStatus:
    """Current cellular-module status."""

    connected: bool
    network: str
    signal_strength: int
    module: str
    last_message: str

    def as_dict(self) -> Dict[str, Any]:
        """Return status as a dictionary."""

        return {
            "connected": self.connected,
            "network": self.network,
            "signal_strength": self.signal_strength,
            "module": self.module,
            "last_message": self.last_message,
        }


# ============================================================
# CELLULAR MANAGER
# ============================================================

class CellularManager:
    """
    SAFEBAND cellular communication manager.

    Current prototype capabilities:

        - EC200U connection simulation
        - Network status simulation
        - Signal-strength simulation
        - Sensor-data transmission simulation
        - Emergency-alert transmission simulation
        - Duplicate emergency protection
        - Transmission history

    Architecture:

        SAFEBAND Application
                ↓
        CellularManager
                ↓
        Transport Layer
          ┌─────┴─────┐
          ↓           ↓
       SIMULATED    EC200U
                       ↓
                    UART/AT
    """

    def __init__(
        self,
        simulated: bool = True,
        module_name: str = DEFAULT_MODULE_NAME,
    ) -> None:
        """
        Initialize the cellular manager.

        Parameters
        ----------
        simulated:
            Whether to use simulated communication.

        module_name:
            Display name of the cellular module.
        """

        self.simulated = simulated
        self.module_name = module_name

        # ----------------------------------------------------
        # Connection state
        # ----------------------------------------------------

        self.connected = True
        self.network = DEFAULT_NETWORK
        self.signal_strength = DEFAULT_SIGNAL_STRENGTH

        self.last_message = (
            "Waiting for transmission..."
        )

        # ----------------------------------------------------
        # Transmission history
        # ----------------------------------------------------

        self.message_history: List[
            Dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # Emergency event tracking
        # ----------------------------------------------------

        # Identity of the most recently transmitted emergency
        # event.
        self.last_emergency_event_id: Optional[str] = None

        # Hash of the most recently transmitted emergency payload.
        self.last_emergency_fingerprint: Optional[str] = None

        # Number of emergency alerts actually transmitted.
        self.emergency_alert_count = 0

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _timestamp() -> str:
        """Return a timezone-aware UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    # ========================================================
    # CONNECTION
    # ========================================================

    def connect(self) -> bool:
        """
        Establish the cellular connection.

        In simulation mode this immediately succeeds.
        """

        if self.simulated:

            self.connected = True

            self.network = (
                DEFAULT_NETWORK
            )

            self.signal_strength = (
                DEFAULT_SIGNAL_STRENGTH
            )

            self.last_message = (
                "Cellular network connected."
            )

            return True

        return self._connect_hardware()

    def disconnect(self) -> None:
        """Disconnect from the cellular network."""

        if not self.simulated:
            self._disconnect_hardware()

        self.connected = False
        self.network = "Disconnected"
        self.signal_strength = 0

        self.last_message = (
            "Cellular network disconnected."
        )

    def is_connected(self) -> bool:
        """Return whether cellular communication is available."""

        return self.connected

    # ========================================================
    # HARDWARE CONNECTION PLACEHOLDERS
    # ========================================================

    def _connect_hardware(self) -> bool:
        """
        Establish a physical EC200U connection.

        Reserved for the future ESP32-S3 UART implementation.
        """

        raise NotImplementedError(
            "Physical EC200U communication "
            "is not implemented yet."
        )

    def _disconnect_hardware(self) -> None:
        """Close the future physical EC200U connection."""

        return None

    # ========================================================
    # SIGNAL
    # ========================================================

    def get_signal_strength(self) -> int:
        """
        Return signal strength from 0 to 100.

        A disconnected modem always reports zero.
        """

        if not self.connected:
            return 0

        return max(
            MIN_SIGNAL_STRENGTH,
            min(
                MAX_SIGNAL_STRENGTH,
                int(self.signal_strength),
            ),
        )

    def set_simulated_signal(
        self,
        signal_strength: int,
    ) -> None:
        """
        Set simulated signal strength.

        Useful for dashboard communication-failure testing.
        """

        if not self.simulated:

            raise RuntimeError(
                "Simulated signal can only be changed "
                "in simulation mode."
            )

        self.signal_strength = max(
            MIN_SIGNAL_STRENGTH,
            min(
                MAX_SIGNAL_STRENGTH,
                int(signal_strength),
            ),
        )

    # ========================================================
    # TRANSMISSION VALIDATION
    # ========================================================

    @staticmethod
    def _validate_payload(
        payload: Dict[str, Any],
    ) -> bool:
        """Validate a transmission payload."""

        return isinstance(
            payload,
            dict,
        )

    # ========================================================
    # EVENT ID / FINGERPRINT
    # ========================================================

    @staticmethod
    def _get_event_id(
        alert_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract an explicit emergency event ID.

        The application should preferably provide event_id.

        Example:

            {
                "event_id": "FALL-001",
                "activity": "FALL",
                ...
            }
        """

        event_id = alert_data.get(
            "event_id"
        )

        if event_id is None:
            return None

        event_id = str(
            event_id
        ).strip()

        return event_id or None

    @staticmethod
    def _create_fingerprint(
        alert_data: Dict[str, Any],
    ) -> str:
        """
        Create a stable fingerprint for an emergency payload.

        Dynamic fields such as timestamps are intentionally not
        used so that Streamlit reruns produce the same identity
        for the same emergency state.

        Explicit event_id takes priority when available.
        """

        # ----------------------------------------------------
        # Prefer explicit event ID.
        # ----------------------------------------------------

        event_id = (
            CellularManager._get_event_id(
                alert_data
            )
        )

        if event_id:

            return (
                f"EVENT:{event_id}"
            )

        # ----------------------------------------------------
        # Otherwise construct a stable subset of emergency data.
        # ----------------------------------------------------

        identity_fields = {
            "activity": alert_data.get(
                "activity"
            ),

            "risk_level": alert_data.get(
                "risk_level"
            ),

            "emergency": alert_data.get(
                "emergency"
            ),

            "manual_sos": alert_data.get(
                "manual_sos"
            ),

            "latitude": alert_data.get(
                "latitude"
            ),

            "longitude": alert_data.get(
                "longitude"
            ),
        }

        # Remove fields that are completely absent.
        identity_fields = {
            key: value
            for key, value in identity_fields.items()
            if value is not None
        }

        serialized = json.dumps(
            identity_fields,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    # ========================================================
    # DATA TRANSMISSION
    # ========================================================

    def send_data(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transmit sensor/application data.

        Normal sensor-data transmissions are NOT deduplicated.
        Each call represents a new telemetry sample.
        """

        timestamp = self._timestamp()

        if not self._validate_payload(data):

            return {
                "success": False,
                "type": "DATA",
                "message": (
                    "Transmission failed: "
                    "payload must be a dictionary."
                ),
                "timestamp": timestamp,
            }

        if not self.connected:

            return {
                "success": False,
                "type": "DATA",
                "message": (
                    "Cellular connection unavailable."
                ),
                "timestamp": timestamp,
            }

        if not self.simulated:

            return self._send_data_hardware(
                data,
                timestamp,
            )

        transmission = {
            "type": "DATA",
            "timestamp": timestamp,
            "payload": dict(data),
        }

        self.message_history.append(
            transmission
        )

        self.last_message = (
            "Sensor data transmitted successfully."
        )

        return {
            "success": True,
            "type": "DATA",
            "message": self.last_message,
            "timestamp": timestamp,
            "network": self.network,
            "signal_strength": (
                self.get_signal_strength()
            ),
        }

    def _send_data_hardware(
        self,
        data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Reserved for physical EC200U data transmission."""

        raise NotImplementedError(
            "Physical EC200U data transmission "
            "is not implemented yet."
        )

    # ========================================================
    # EMERGENCY ALERT
    # ========================================================

    def send_emergency_alert(
        self,
        alert_data: Dict[str, Any],
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Transmit an emergency alert exactly once per event.

        Parameters
        ----------
        alert_data:
            Emergency information to transmit.

        force:
            Force a new transmission even if the event appears
            to be a duplicate.

            This is primarily useful for testing.

        Returns
        -------
        dict
            Emergency transmission result.

        Duplicate behaviour
        -------------------

        If the same emergency event is passed repeatedly,
        subsequent calls return:

            success = True
            duplicate = True
            transmitted = False

        This means the dashboard can safely call this method
        during Streamlit reruns without filling the history with
        repeated alerts.
        """

        timestamp = self._timestamp()

        if not self._validate_payload(
            alert_data
        ):

            return {
                "success": False,
                "alert_type": "EMERGENCY",
                "transmitted": False,
                "duplicate": False,
                "message": (
                    "Emergency alert failed: "
                    "payload must be a dictionary."
                ),
                "timestamp": timestamp,
            }

        if not self.connected:

            return {
                "success": False,
                "alert_type": "EMERGENCY",
                "transmitted": False,
                "duplicate": False,
                "message": (
                    "Emergency alert failed: "
                    "cellular connection unavailable."
                ),
                "timestamp": timestamp,
            }

        # ----------------------------------------------------
        # Identify the emergency event.
        # ----------------------------------------------------

        event_id = (
            self._get_event_id(
                alert_data
            )
        )

        fingerprint = (
            self._create_fingerprint(
                alert_data
            )
        )

        # ----------------------------------------------------
        # Duplicate protection.
        # ----------------------------------------------------

        if (
            EMERGENCY_DEDUPLICATION_ENABLED
            and not force
            and (
                fingerprint
                == self.last_emergency_fingerprint
            )
        ):

            self.last_message = (
                "Duplicate emergency alert suppressed."
            )

            return {
                "success": True,
                "alert_type": "EMERGENCY",
                "transmitted": False,
                "duplicate": True,
                "event_id": event_id,
                "message": self.last_message,
                "timestamp": timestamp,
                "network": self.network,
                "signal_strength": (
                    self.get_signal_strength()
                ),
                "emergency_alert_count": (
                    self.emergency_alert_count
                ),
            }

        # ----------------------------------------------------
        # Hardware transport.
        # ----------------------------------------------------

        if not self.simulated:

            result = (
                self._send_emergency_hardware(
                    alert_data,
                    timestamp,
                )
            )

            # Only remember the event if hardware transmission
            # actually succeeded.
            if result.get(
                "success",
                False,
            ):

                self.last_emergency_event_id = (
                    event_id
                )

                self.last_emergency_fingerprint = (
                    fingerprint
                )

                self.emergency_alert_count += 1

            return result

        # ----------------------------------------------------
        # Simulated transmission.
        # ----------------------------------------------------

        alert = {
            "type": "EMERGENCY",
            "timestamp": timestamp,
            "event_id": event_id,
            "payload": dict(alert_data),
        }

        self.message_history.append(
            alert
        )

        self.last_emergency_event_id = (
            event_id
        )

        self.last_emergency_fingerprint = (
            fingerprint
        )

        self.emergency_alert_count += 1

        self.last_message = (
            "EMERGENCY ALERT TRANSMITTED."
        )

        return {
            "success": True,
            "alert_type": "EMERGENCY",
            "transmitted": True,
            "duplicate": False,
            "event_id": event_id,
            "message": self.last_message,
            "timestamp": timestamp,
            "network": self.network,
            "signal_strength": (
                self.get_signal_strength()
            ),
            "emergency_alert_count": (
                self.emergency_alert_count
            ),
        }

    def _send_emergency_hardware(
        self,
        alert_data: Dict[str, Any],
        timestamp: str,
    ) -> Dict[str, Any]:
        """Reserved for physical emergency transmission."""

        raise NotImplementedError(
            "Physical EC200U emergency transmission "
            "is not implemented yet."
        )

    # ========================================================
    # EMERGENCY STATE
    # ========================================================

    def clear_emergency_state(self) -> None:
        """
        Clear the current emergency-event identity.

        Call this when the application has returned to a safe
        state and wants the next emergency to be treated as a
        completely new event.
        """

        self.last_emergency_event_id = None
        self.last_emergency_fingerprint = None

        self.last_message = (
            "Emergency transmission state cleared."
        )

    def get_emergency_state(self) -> Dict[str, Any]:
        """Return the current emergency transmission state."""

        return {
            "deduplication_enabled": (
                EMERGENCY_DEDUPLICATION_ENABLED
            ),

            "last_event_id": (
                self.last_emergency_event_id
            ),

            "last_fingerprint": (
                self.last_emergency_fingerprint
            ),

            "alerts_transmitted": (
                self.emergency_alert_count
            ),
        }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> CellularStatus:
        """Return the current cellular status."""

        return CellularStatus(
            connected=self.connected,
            network=self.network,
            signal_strength=(
                self.get_signal_strength()
            ),
            module=self.module_name,
            last_message=self.last_message,
        )

    def get_status_dict(
        self,
    ) -> Dict[str, Any]:
        """Return cellular status as a dictionary."""

        return self.get_status().as_dict()

    # ========================================================
    # MESSAGE HISTORY
    # ========================================================

    def get_message_history(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Return a copy of the transmission history.

        A copy prevents callers from accidentally modifying
        the manager's internal history.
        """

        return [
            dict(message)
            for message in self.message_history
        ]

    def clear_message_history(self) -> None:
        """Clear stored simulated transmissions."""

        self.message_history.clear()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Reset the manager to its initial simulated state.
        """

        self.connected = True

        self.network = (
            DEFAULT_NETWORK
        )

        self.signal_strength = (
            DEFAULT_SIGNAL_STRENGTH
        )

        self.last_message = (
            "Waiting for transmission..."
        )

        self.clear_message_history()

        self.last_emergency_event_id = None

        self.last_emergency_fingerprint = None

        self.emergency_alert_count = 0


# ============================================================
# GLOBAL CELLULAR MANAGER
# ============================================================

_cellular_manager = CellularManager(
    simulated=True
)


# ============================================================
# PUBLIC CONVENIENCE API
# ============================================================

def get_cellular_status() -> Dict[str, Any]:
    """Return the current EC200U status."""

    return _cellular_manager.get_status_dict()


def connect_cellular() -> bool:
    """Connect the global cellular manager."""

    return _cellular_manager.connect()


def disconnect_cellular() -> None:
    """Disconnect the global cellular manager."""

    _cellular_manager.disconnect()


def transmit_data(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Transmit sensor/application data."""

    return _cellular_manager.send_data(
        data
    )


def transmit_emergency_alert(
    alert_data: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Transmit an emergency alert.

    Duplicate alerts are suppressed unless force=True.
    """

    return _cellular_manager.send_emergency_alert(
        alert_data,
        force=force,
    )


def get_signal_strength() -> int:
    """Return the current cellular signal strength."""

    return _cellular_manager.get_signal_strength()


def get_message_history() -> List[Dict[str, Any]]:
    """Return cellular transmission history."""

    return _cellular_manager.get_message_history()


def get_emergency_state() -> Dict[str, Any]:
    """Return emergency transmission state."""

    return _cellular_manager.get_emergency_state()


def clear_emergency_state() -> None:
    """
    Clear the current emergency event identity.

    This allows the next emergency to be treated as a new event.
    """

    _cellular_manager.clear_emergency_state()


def reset_cellular() -> None:
    """Reset the global cellular manager."""

    _cellular_manager.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "CellularStatus",
    "CellularManager",
    "get_cellular_status",
    "connect_cellular",
    "disconnect_cellular",
    "transmit_data",
    "transmit_emergency_alert",
    "get_signal_strength",
    "get_message_history",
    "get_emergency_state",
    "clear_emergency_state",
    "reset_cellular",
]