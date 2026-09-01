"""
SAFEBAND AI - Cloud Communication Module

Cloud communication abstraction for SAFEBAND AI.

Current implementation:
    In-memory cloud simulation for development and demonstration.

Future implementations:
    REST API
    Firebase
    Supabase
    MQTT
    Custom SAFEBAND backend

Important prototype behaviour:
    Emergency events are idempotent.

    The same emergency event will only be uploaded once.
    This prevents Streamlit reruns from creating duplicate
    emergency records.

Design principle:
    The application and dashboard communicate with CloudManager
    rather than directly depending on a specific cloud provider.

This keeps the project portable and allows the backend to be
changed later without rewriting the application layer.
"""


from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SERVER = "SAFEBAND AI Cloud"

DEFAULT_CONNECTION_MESSAGE = (
    "Cloud connection ready."
)

EMERGENCY_DEDUPLICATION_ENABLED = True


# ============================================================
# CLOUD STATUS
# ============================================================

@dataclass
class CloudStatus:
    """Current cloud connection and synchronization status."""

    connected: bool
    server: str
    last_sync: Optional[str]
    records_uploaded: int
    last_message: str

    def as_dict(self) -> Dict[str, Any]:
        """Return cloud status as a dictionary."""

        return {
            "connected": self.connected,
            "server": self.server,
            "last_sync": self.last_sync,
            "records_uploaded": self.records_uploaded,
            "last_message": self.last_message,
        }


# ============================================================
# CLOUD MANAGER
# ============================================================

class CloudManager:
    """
    SAFEBAND AI cloud communication manager.

    Current capabilities:

        - Simulated cloud connection
        - Sensor-data upload
        - Safety-event upload
        - Emergency-event upload
        - Emergency duplicate protection
        - Record retrieval
        - Record filtering
        - Synchronization status

    The current storage backend is memory-based.

    A future persistent backend can implement the same public
    interface.
    """

    VALID_RECORD_TYPES = {
        "sensor_data",
        "safety_event",
        "emergency",
    }

    def __init__(
        self,
        simulated: bool = True,
        server: str = DEFAULT_SERVER,
    ) -> None:
        """
        Initialize the cloud manager.

        Parameters
        ----------
        simulated:
            Use the local in-memory cloud simulation.

        server:
            Display name of the cloud backend.
        """

        self.simulated = simulated
        self.server = server

        # ----------------------------------------------------
        # Connection state
        # ----------------------------------------------------

        self.connected = True
        self.last_sync: Optional[str] = None

        self.last_message = (
            DEFAULT_CONNECTION_MESSAGE
        )

        # ----------------------------------------------------
        # Stored records
        # ----------------------------------------------------

        self.records: List[
            Dict[str, Any]
        ] = []

        # ----------------------------------------------------
        # Emergency event tracking
        # ----------------------------------------------------

        self.last_emergency_event_id: Optional[str] = None

        self.last_emergency_fingerprint: Optional[str] = None

        self.emergency_event_count = 0

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
        Connect to the cloud backend.

        The prototype implementation connects immediately.
        """

        if not self.simulated:
            return self._connect_backend()

        self.connected = True

        self.last_message = (
            "Connected to cloud server."
        )

        return True

    def disconnect(self) -> None:
        """Disconnect from the cloud backend."""

        if not self.simulated:
            self._disconnect_backend()

        self.connected = False

        self.last_message = (
            "Cloud connection disconnected."
        )

    def is_connected(self) -> bool:
        """Return whether cloud communication is available."""

        return self.connected

    # ========================================================
    # BACKEND PLACEHOLDERS
    # ========================================================

    def _connect_backend(self) -> bool:
        """
        Reserved for real cloud connection logic.

        Future implementations may initialize REST, Firebase,
        Supabase, MQTT or another backend here.
        """

        raise NotImplementedError(
            "Real cloud backend is not implemented yet."
        )

    def _disconnect_backend(self) -> None:
        """Reserved for real backend disconnect logic."""

        return None

    # ========================================================
    # PAYLOAD VALIDATION
    # ========================================================

    @staticmethod
    def _validate_payload(
        payload: Dict[str, Any],
    ) -> bool:
        """Check whether a cloud payload is valid."""

        return isinstance(
            payload,
            dict,
        )

    # ========================================================
    # EMERGENCY EVENT IDENTITY
    # ========================================================

    @staticmethod
    def _get_event_id(
        emergency_data: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract an explicit emergency event ID.

        The application should preferably provide event_id.

        Example:

            {
                "event_id": "FALL-001",
                "activity": "FALL",
                "risk_level": "CRITICAL"
            }
        """

        event_id = emergency_data.get(
            "event_id"
        )

        if event_id is None:
            return None

        event_id = str(
            event_id
        ).strip()

        return event_id or None

    @staticmethod
    def _create_emergency_fingerprint(
        emergency_data: Dict[str, Any],
    ) -> str:
        """
        Create a stable identity for an emergency event.

        Dynamic timestamps are intentionally excluded.

        An explicit event_id takes priority. If one is not
        supplied, important emergency-state fields are used.
        """

        event_id = (
            CloudManager._get_event_id(
                emergency_data
            )
        )

        if event_id:

            return (
                f"EVENT:{event_id}"
            )

        identity_fields = {
            "activity": emergency_data.get(
                "activity"
            ),

            "risk_level": emergency_data.get(
                "risk_level"
            ),

            "emergency": emergency_data.get(
                "emergency"
            ),

            "manual_sos": emergency_data.get(
                "manual_sos"
            ),

            "latitude": emergency_data.get(
                "latitude"
            ),

            "longitude": emergency_data.get(
                "longitude"
            ),
        }

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
    # INTERNAL RECORD STORAGE
    # ========================================================

    def _store_record(
        self,
        record_type: str,
        data: Dict[str, Any],
        timestamp: str,
        priority: Optional[str] = None,
    ) -> int:
        """
        Store a record in the simulated cloud.

        Returns
        -------
        int
            Generated record ID.
        """

        record: Dict[str, Any] = {
            "record_id": len(self.records) + 1,
            "record_type": record_type,
            "timestamp": timestamp,
            "data": dict(data),
        }

        if priority is not None:
            record["priority"] = priority

        self.records.append(
            record
        )

        self.last_sync = timestamp

        return record["record_id"]

    # ========================================================
    # SENSOR DATA
    # ========================================================

    def upload_data(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Upload sensor/system data.

        Sensor telemetry is intentionally NOT deduplicated.

        Every call represents a new sensor sample.
        """

        timestamp = self._timestamp()

        if not self._validate_payload(data):

            return {
                "success": False,
                "record_type": "sensor_data",
                "message": (
                    "Upload failed: "
                    "payload must be a dictionary."
                ),
                "timestamp": timestamp,
            }

        if not self.connected:

            return {
                "success": False,
                "record_type": "sensor_data",
                "message": (
                    "Cloud connection unavailable."
                ),
                "timestamp": timestamp,
            }

        if not self.simulated:

            return self._upload_backend(
                "sensor_data",
                data,
                timestamp,
            )

        record_id = self._store_record(
            record_type="sensor_data",
            data=data,
            timestamp=timestamp,
        )

        self.last_message = (
            "Sensor data uploaded successfully."
        )

        return {
            "success": True,
            "record_type": "sensor_data",
            "message": self.last_message,
            "timestamp": timestamp,
            "record_id": record_id,
        }

    # ========================================================
    # SAFETY EVENT
    # ========================================================

    def upload_event(
        self,
        event_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Upload a detected SAFEBAND safety event.

        Safety events are treated as event records and can
        therefore be called whenever a new safety transition
        is detected.
        """

        timestamp = self._timestamp()

        if not self._validate_payload(
            event_data
        ):

            return {
                "success": False,
                "record_type": "safety_event",
                "message": (
                    "Event upload failed: "
                    "payload must be a dictionary."
                ),
                "timestamp": timestamp,
            }

        if not self.connected:

            return {
                "success": False,
                "record_type": "safety_event",
                "message": (
                    "Cloud connection unavailable."
                ),
                "timestamp": timestamp,
            }

        if not self.simulated:

            return self._upload_backend(
                "safety_event",
                event_data,
                timestamp,
            )

        record_id = self._store_record(
            record_type="safety_event",
            data=event_data,
            timestamp=timestamp,
        )

        self.last_message = (
            "Safety event synchronized with cloud."
        )

        return {
            "success": True,
            "record_type": "safety_event",
            "message": self.last_message,
            "timestamp": timestamp,
            "record_id": record_id,
        }

    # ========================================================
    # EMERGENCY EVENT
    # ========================================================

    def upload_emergency(
        self,
        emergency_data: Dict[str, Any],
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Upload an emergency event exactly once per event.

        Parameters
        ----------
        emergency_data:
            Emergency information.

        force:
            Force another upload even when this appears to be
            the same emergency event.

            Intended mainly for testing.

        Returns
        -------
        dict
            Upload result.

        Duplicate behaviour
        -------------------

        If the same emergency is encountered again:

            success = True
            uploaded = False
            duplicate = True

        This allows Streamlit to rerun safely without creating
        duplicate emergency records.
        """

        timestamp = self._timestamp()

        if not self._validate_payload(
            emergency_data
        ):

            return {
                "success": False,
                "record_type": "emergency",
                "uploaded": False,
                "duplicate": False,
                "message": (
                    "Emergency upload failed: "
                    "payload must be a dictionary."
                ),
                "timestamp": timestamp,
            }

        if not self.connected:

            return {
                "success": False,
                "record_type": "emergency",
                "uploaded": False,
                "duplicate": False,
                "message": (
                    "Emergency upload failed: "
                    "cloud unavailable."
                ),
                "timestamp": timestamp,
            }

        # ----------------------------------------------------
        # Identify the emergency.
        # ----------------------------------------------------

        event_id = (
            self._get_event_id(
                emergency_data
            )
        )

        fingerprint = (
            self._create_emergency_fingerprint(
                emergency_data
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
                "Duplicate emergency upload suppressed."
            )

            return {
                "success": True,
                "record_type": "emergency",
                "uploaded": False,
                "duplicate": True,
                "event_id": event_id,
                "message": self.last_message,
                "timestamp": timestamp,
                "record_id": None,
                "records_uploaded": len(
                    self.records
                ),
            }

        # ----------------------------------------------------
        # Real backend.
        # ----------------------------------------------------

        if not self.simulated:

            result = self._upload_backend(
                "emergency",
                emergency_data,
                timestamp,
                priority="CRITICAL",
            )

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

                self.emergency_event_count += 1

            return result

        # ----------------------------------------------------
        # Simulated cloud upload.
        # ----------------------------------------------------

        record_id = self._store_record(
            record_type="emergency",
            data=emergency_data,
            timestamp=timestamp,
            priority="CRITICAL",
        )

        self.last_emergency_event_id = (
            event_id
        )

        self.last_emergency_fingerprint = (
            fingerprint
        )

        self.emergency_event_count += 1

        self.last_message = (
            "Emergency event synchronized with cloud."
        )

        return {
            "success": True,
            "record_type": "emergency",
            "uploaded": True,
            "duplicate": False,
            "event_id": event_id,
            "message": self.last_message,
            "timestamp": timestamp,
            "priority": "CRITICAL",
            "record_id": record_id,
            "records_uploaded": len(
                self.records
            ),
        }

    # ========================================================
    # REAL BACKEND PLACEHOLDER
    # ========================================================

    def _upload_backend(
        self,
        record_type: str,
        data: Dict[str, Any],
        timestamp: str,
        priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reserved for real cloud upload implementation.

        This method provides the extension point for REST,
        Firebase, Supabase, MQTT or another backend.
        """

        raise NotImplementedError(
            "Real cloud backend upload "
            "is not implemented yet."
        )

    # ========================================================
    # EMERGENCY STATE
    # ========================================================

    def clear_emergency_state(self) -> None:
        """
        Clear the current emergency identity.

        The next emergency will then be considered a new event.
        """

        self.last_emergency_event_id = None
        self.last_emergency_fingerprint = None

        self.last_message = (
            "Emergency cloud state cleared."
        )

    def get_emergency_state(
        self,
    ) -> Dict[str, Any]:
        """Return emergency upload state."""

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

            "emergency_events_uploaded": (
                self.emergency_event_count
            ),
        }

    # ========================================================
    # RECORD RETRIEVAL
    # ========================================================

    def get_records(
        self,
        record_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored cloud records.

        Parameters
        ----------
        record_type:
            Optional filter:

                sensor_data
                safety_event
                emergency
        """

        if record_type is None:

            return [
                dict(record)
                for record in self.records
            ]

        if record_type not in self.VALID_RECORD_TYPES:
            return []

        return [
            dict(record)
            for record in self.records
            if record.get(
                "record_type"
            ) == record_type
        ]

    def get_latest_record(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return the latest uploaded record."""

        if not self.records:
            return None

        return dict(
            self.records[-1]
        )

    def get_record_count(
        self,
        record_type: Optional[str] = None,
    ) -> int:
        """Return the number of stored records."""

        if record_type is None:
            return len(
                self.records
            )

        return len(
            self.get_records(
                record_type
            )
        )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> CloudStatus:
        """Return the current cloud status."""

        return CloudStatus(
            connected=self.connected,
            server=self.server,
            last_sync=self.last_sync,
            records_uploaded=len(
                self.records
            ),
            last_message=self.last_message,
        )

    def get_status_dict(
        self,
    ) -> Dict[str, Any]:
        """Return cloud status as a dictionary."""

        return self.get_status().as_dict()

    # ========================================================
    # RESET
    # ========================================================

    def clear_records(self) -> None:
        """Clear simulated cloud records."""

        self.records.clear()
        self.last_sync = None

    def reset(self) -> None:
        """Reset the cloud manager."""

        self.connected = True

        self.last_sync = None

        self.last_message = (
            DEFAULT_CONNECTION_MESSAGE
        )

        self.clear_records()

        self.last_emergency_event_id = None

        self.last_emergency_fingerprint = None

        self.emergency_event_count = 0


# ============================================================
# GLOBAL CLOUD MANAGER
# ============================================================

_cloud_manager = CloudManager(
    simulated=True
)


# ============================================================
# PUBLIC API
# ============================================================

def get_cloud_status() -> Dict[str, Any]:
    """Return the current cloud status."""

    return _cloud_manager.get_status_dict()


def connect_cloud() -> bool:
    """Connect the global cloud manager."""

    return _cloud_manager.connect()


def disconnect_cloud() -> None:
    """Disconnect the global cloud manager."""

    _cloud_manager.disconnect()


def upload_sensor_data(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Upload sensor/system data."""

    return _cloud_manager.upload_data(
        data
    )


def upload_safety_event(
    event_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Upload a safety event."""

    return _cloud_manager.upload_event(
        event_data
    )


def upload_emergency_event(
    emergency_data: Dict[str, Any],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Upload an emergency event.

    Duplicate events are suppressed unless force=True.
    """

    return _cloud_manager.upload_emergency(
        emergency_data,
        force=force,
    )


def get_cloud_records(
    record_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve simulated cloud records."""

    return _cloud_manager.get_records(
        record_type
    )


def get_latest_cloud_record(
) -> Optional[Dict[str, Any]]:
    """Retrieve the latest cloud record."""

    return _cloud_manager.get_latest_record()


def get_cloud_record_count(
    record_type: Optional[str] = None,
) -> int:
    """Return the number of cloud records."""

    return _cloud_manager.get_record_count(
        record_type
    )


def get_emergency_cloud_state() -> Dict[str, Any]:
    """Return emergency cloud synchronization state."""

    return _cloud_manager.get_emergency_state()


def clear_emergency_cloud_state() -> None:
    """
    Clear the current emergency event identity.
    """

    _cloud_manager.clear_emergency_state()


def reset_cloud() -> None:
    """Reset the global cloud manager."""

    _cloud_manager.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "CloudStatus",
    "CloudManager",
    "get_cloud_status",
    "connect_cloud",
    "disconnect_cloud",
    "upload_sensor_data",
    "upload_safety_event",
    "upload_emergency_event",
    "get_cloud_records",
    "get_latest_cloud_record",
    "get_cloud_record_count",
    "get_emergency_cloud_state",
    "clear_emergency_cloud_state",
    "reset_cloud",
]