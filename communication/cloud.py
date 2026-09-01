"""
SAFEBAND AI - Cloud Communication Module

Prototype cloud communication interface.

The current version simulates cloud connectivity and data storage
for demonstration purposes. It is structured so that a real cloud
API/backend can be integrated later.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class CloudStatus:
    """Current cloud connection status."""

    connected: bool
    server: str
    last_sync: Optional[str]
    records_uploaded: int
    last_message: str


class CloudManager:
    """
    SAFEBAND AI cloud communication manager.

    Prototype capabilities:
        - Simulate cloud connection
        - Upload sensor data
        - Upload safety events
        - Store uploaded records
        - Retrieve cloud records
        - Simulate emergency-event synchronization

    Future implementation:
        Replace the internal storage with a REST API, MQTT,
        Firebase, or another selected cloud backend.
    """

    def __init__(self):
        self.server = "SAFEBAND AI Cloud"
        self.connected = True
        self.last_sync = None
        self.last_message = "Cloud connection ready."
        self.records: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------

    def connect(self) -> bool:
        """Simulate connection to the cloud server."""

        self.connected = True
        self.last_message = "Connected to cloud server."

        return True

    def disconnect(self) -> None:
        """Simulate cloud disconnection."""

        self.connected = False
        self.last_message = "Cloud connection disconnected."

    # ---------------------------------------------------------
    # DATA UPLOAD
    # ---------------------------------------------------------

    def upload_data(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload sensor/system data to the simulated cloud.

        Parameters
        ----------
        data : dict
            Sensor and system information.

        Returns
        -------
        dict
            Upload status.
        """

        if not self.connected:
            return {
                "success": False,
                "message": "Cloud connection unavailable.",
                "timestamp": datetime.now().isoformat()
            }

        timestamp = datetime.now().isoformat()

        record = {
            "record_type": "sensor_data",
            "timestamp": timestamp,
            "data": data
        }

        self.records.append(record)
        self.last_sync = timestamp
        self.last_message = "Sensor data uploaded successfully."

        return {
            "success": True,
            "message": self.last_message,
            "timestamp": timestamp,
            "record_id": len(self.records)
        }

    # ---------------------------------------------------------
    # SAFETY EVENT
    # ---------------------------------------------------------

    def upload_event(
        self,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload a detected safety event to the simulated cloud.

        Parameters
        ----------
        event_data : dict
            Event information such as activity, risk score,
            location and alert status.
        """

        if not self.connected:
            return {
                "success": False,
                "message": "Cloud connection unavailable.",
                "timestamp": datetime.now().isoformat()
            }

        timestamp = datetime.now().isoformat()

        event = {
            "record_type": "safety_event",
            "timestamp": timestamp,
            "data": event_data
        }

        self.records.append(event)
        self.last_sync = timestamp
        self.last_message = "Safety event synchronized with cloud."

        return {
            "success": True,
            "message": self.last_message,
            "timestamp": timestamp,
            "record_id": len(self.records)
        }

    # ---------------------------------------------------------
    # EMERGENCY EVENT
    # ---------------------------------------------------------

    def upload_emergency(
        self,
        emergency_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Upload an emergency event to the simulated cloud.

        This is used when the risk engine identifies a critical
        condition or emergency event.
        """

        if not self.connected:
            return {
                "success": False,
                "message": "Emergency data upload failed: cloud unavailable.",
                "timestamp": datetime.now().isoformat()
            }

        timestamp = datetime.now().isoformat()

        emergency = {
            "record_type": "emergency",
            "timestamp": timestamp,
            "priority": "CRITICAL",
            "data": emergency_data
        }

        self.records.append(emergency)
        self.last_sync = timestamp
        self.last_message = "Emergency event synchronized with cloud."

        return {
            "success": True,
            "message": self.last_message,
            "timestamp": timestamp,
            "priority": "CRITICAL",
            "record_id": len(self.records)
        }

    # ---------------------------------------------------------
    # RETRIEVE DATA
    # ---------------------------------------------------------

    def get_records(
        self,
        record_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored cloud records.

        Parameters
        ----------
        record_type : str, optional
            Filter by:
                sensor_data
                safety_event
                emergency
        """

        if record_type is None:
            return self.records.copy()

        return [
            record
            for record in self.records
            if record.get("record_type") == record_type
        ]

    def get_latest_record(self) -> Optional[Dict[str, Any]]:
        """Return the most recently uploaded record."""

        if not self.records:
            return None

        return self.records[-1]

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def get_status(self) -> CloudStatus:
        """Return current cloud status."""

        return CloudStatus(
            connected=self.connected,
            server=self.server,
            last_sync=self.last_sync,
            records_uploaded=len(self.records),
            last_message=self.last_message
        )

    def get_status_dict(self) -> Dict[str, Any]:
        """Return cloud status as a dictionary."""

        status = self.get_status()

        return {
            "connected": status.connected,
            "server": status.server,
            "last_sync": status.last_sync,
            "records_uploaded": status.records_uploaded,
            "last_message": status.last_message
        }


# -------------------------------------------------------------
# GLOBAL CLOUD MANAGER
# -------------------------------------------------------------

_cloud_manager = CloudManager()


# -------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------------------------------------

def get_cloud_status() -> Dict[str, Any]:
    """Return the current simulated cloud status."""

    return _cloud_manager.get_status_dict()


def upload_sensor_data(
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload simulated sensor data."""

    return _cloud_manager.upload_data(data)


def upload_safety_event(
    event_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload a safety event."""

    return _cloud_manager.upload_event(event_data)


def upload_emergency_event(
    emergency_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Upload an emergency event."""

    return _cloud_manager.upload_emergency(emergency_data)


def get_cloud_records(
    record_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieve simulated cloud records."""

    return _cloud_manager.get_records(record_type)


def get_latest_cloud_record() -> Optional[Dict[str, Any]]:
    """Retrieve the latest cloud record."""

    return _cloud_manager.get_latest_record()