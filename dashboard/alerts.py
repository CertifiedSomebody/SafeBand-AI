"""
SAFEBAND AI - Alert Management Module

Central alert-management layer for SAFEBAND AI.

Responsibilities:
    - Safety-status alerts
    - Warning alerts
    - Emergency alerts
    - Manual SOS alerts
    - Caregiver-notification simulation
    - Alert lifecycle management
    - Alert history

IMPORTANT
---------
This module is designed for Streamlit-safe state handling.

Streamlit reruns the application whenever widgets, timers, or
session state change. Therefore, an identical active event must
NOT create a new alert on every rerun.

The alert manager uses active-event deduplication:

    Same active event
        -> reuse existing alert

    New event after previous event is resolved
        -> create a new alert

This allows FALL, HIGH_RISK and SOS testing without filling the
history with duplicate records every few seconds.

The module remains independent of Streamlit so it can later be
used with real hardware, cloud services, cellular communication,
or caregiver applications.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_MAX_HISTORY = 100

ALERT_STATUS_ACTIVE = "ACTIVE"
ALERT_STATUS_RESOLVED = "RESOLVED"

SEVERITY_INFO = "INFO"
SEVERITY_WARNING = "WARNING"
SEVERITY_HIGH = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

ALERT_TYPE_STATUS = "STATUS"
ALERT_TYPE_SAFETY = "SAFETY"
ALERT_TYPE_EMERGENCY = "EMERGENCY"
ALERT_TYPE_SOS = "SOS"


# ============================================================
# ALERT DATA MODEL
# ============================================================

@dataclass
class Alert:
    """
    Represents a SAFEBAND safety alert.
    """

    timestamp: str
    alert_type: str
    severity: str
    title: str
    message: str
    status: str
    risk_score: float
    activity: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert into a serializable dictionary."""

        return asdict(self)


# ============================================================
# ALERT MANAGER
# ============================================================

class AlertManager:
    """
    SAFEBAND AI alert-management system.

    Alert lifecycle:

        CREATE
          ↓
        ACTIVE
          ↓
        RESOLVED

    Streamlit-safe behaviour:

        Repeated identical active events are deduplicated.

        Example:

            FALL detected
                ↓
            alert created
                ↓
            Streamlit rerun
                ↓
            same FALL still active
                ↓
            existing alert reused

        Once the alert is resolved and another FALL occurs,
        a new alert is created.

    Caregiver notification is also sent only once for a
    particular active alert.
    """

    def __init__(
        self,
        max_history: int = DEFAULT_MAX_HISTORY,
        notifications_enabled: bool = True,
    ) -> None:

        self.max_history = max(
            1,
            int(max_history),
        )

        self.notifications_enabled = bool(
            notifications_enabled
        )

        self.alert_history: List[Alert] = []

        self.last_alert: Optional[Alert] = None

        self.notification_sent = False

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
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_severity(
        severity: str,
    ) -> str:
        """Normalize alert severity."""

        return str(
            severity
        ).strip().upper()

    @staticmethod
    def _normalize_alert_type(
        alert_type: str,
    ) -> str:
        """Normalize alert type."""

        return str(
            alert_type
        ).strip().upper()

    @staticmethod
    def _normalize_activity(
        activity: str,
    ) -> str:
        """Normalize activity name."""

        return str(
            activity
        ).strip().upper()

    @staticmethod
    def _normalize_risk_score(
        risk_score: float,
    ) -> float:
        """Clamp risk score to the supported 0-100 range."""

        try:
            score = float(
                risk_score
            )

        except (TypeError, ValueError):
            score = 0.0

        return round(
            max(
                0.0,
                min(
                    100.0,
                    score,
                ),
            ),
            1,
        )

    # ========================================================
    # ACTIVE EVENT DETECTION
    # ========================================================

    def _find_active_duplicate(
        self,
        alert_type: str,
        severity: str,
        activity: str,
    ) -> Optional[Alert]:
        """
        Find an already-active alert representing the same event.

        Deduplication intentionally uses the event category rather
        than the exact message or risk score.

        This is important because sensor values can fluctuate:

            FALL risk = 85
            FALL risk = 93
            FALL risk = 88

        These are still one continuous FALL event, not three
        independent emergencies.
        """

        normalized_type = self._normalize_alert_type(
            alert_type
        )

        normalized_severity = self._normalize_severity(
            severity
        )

        normalized_activity = self._normalize_activity(
            activity
        )

        for alert in self.alert_history:

            if alert.status != ALERT_STATUS_ACTIVE:
                continue

            if alert.alert_type != normalized_type:
                continue

            if alert.severity != normalized_severity:
                continue

            if alert.activity != normalized_activity:
                continue

            return alert

        return None

    # ========================================================
    # UPDATE EXISTING EVENT
    # ========================================================

    def _update_existing_alert(
        self,
        alert: Alert,
        risk_score: float,
        message: str,
    ) -> Alert:
        """
        Update the live values of an existing active event.

        The original timestamp is intentionally preserved because
        it represents when the event started.

        Risk score and message can change as new sensor readings
        arrive.
        """

        alert.risk_score = self._normalize_risk_score(
            risk_score
        )

        alert.message = str(
            message
        )

        self.last_alert = alert

        return alert

    # ========================================================
    # CREATE ALERT
    # ========================================================

    def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        risk_score: float = 0.0,
        activity: str = "UNKNOWN",
    ) -> Alert:
        """
        Create and store a new alert.

        If an equivalent alert is already ACTIVE, that alert is
        returned instead of creating a duplicate.

        This prevents Streamlit reruns from generating repeated
        emergency/SOS records.
        """

        normalized_type = self._normalize_alert_type(
            alert_type
        )

        normalized_severity = self._normalize_severity(
            severity
        )

        normalized_activity = self._normalize_activity(
            activity
        )

        normalized_score = self._normalize_risk_score(
            risk_score
        )

        # ----------------------------------------------------
        # DUPLICATE ACTIVE EVENT CHECK
        # ----------------------------------------------------

        existing_alert = self._find_active_duplicate(
            alert_type=normalized_type,
            severity=normalized_severity,
            activity=normalized_activity,
        )

        if existing_alert is not None:

            return self._update_existing_alert(
                alert=existing_alert,
                risk_score=normalized_score,
                message=message,
            )

        # ----------------------------------------------------
        # CREATE NEW EVENT
        # ----------------------------------------------------

        alert = Alert(
            timestamp=self._timestamp(),

            alert_type=normalized_type,

            severity=normalized_severity,

            title=str(
                title
            ),

            message=str(
                message
            ),

            status=ALERT_STATUS_ACTIVE,

            risk_score=normalized_score,

            activity=normalized_activity,
        )

        self.alert_history.insert(
            0,
            alert,
        )

        if len(
            self.alert_history
        ) > self.max_history:

            self.alert_history = (
                self.alert_history[
                    :self.max_history
                ]
            )

        self.last_alert = alert

        # Only a genuinely new event requires a new notification.
        self.notification_sent = False

        return alert

    # ========================================================
    # SAFETY ALERT
    # ========================================================

    def create_safety_alert(
        self,
        risk_score: float,
        risk_level: str,
        activity: str,
        reason: str,
    ) -> Alert:
        """
        Create an alert from a calculated risk level.

        Mapping:

            LOW       → informational status
            MODERATE  → warning
            HIGH      → high-risk warning
            CRITICAL  → emergency
        """

        level = str(
            risk_level
        ).strip().upper()

        if level == "CRITICAL":

            return self.create_emergency_alert(
                activity=activity,
                risk_score=risk_score,
                reason=reason,
            )

        if level == "HIGH":

            return self.create_alert(
                alert_type=ALERT_TYPE_SAFETY,
                severity=SEVERITY_HIGH,
                title="High Risk Detected",
                message=reason,
                risk_score=risk_score,
                activity=activity,
            )

        if level == "MODERATE":

            return self.create_alert(
                alert_type=ALERT_TYPE_SAFETY,
                severity=SEVERITY_WARNING,
                title="Safety Warning",
                message=reason,
                risk_score=risk_score,
                activity=activity,
            )

        return self.create_alert(
            alert_type=ALERT_TYPE_STATUS,
            severity=SEVERITY_INFO,
            title="System Normal",
            message=(
                "All monitored parameters "
                "are within safe range."
            ),
            risk_score=risk_score,
            activity=activity,
        )

    # ========================================================
    # EMERGENCY ALERT
    # ========================================================

    def create_emergency_alert(
        self,
        activity: str,
        risk_score: float,
        reason: str,
    ) -> Alert:
        """
        Create a critical emergency alert.

        If the same emergency is already active, the existing
        alert is updated instead of creating another record.

        Caregiver notification is therefore sent only once for
        the active emergency event.
        """

        alert = self.create_alert(
            alert_type=ALERT_TYPE_EMERGENCY,
            severity=SEVERITY_CRITICAL,
            title="🚨 EMERGENCY DETECTED",
            message=reason,
            risk_score=risk_score,
            activity=activity,
        )

        # ----------------------------------------------------
        # NOTIFICATION
        # ----------------------------------------------------

        if (
            self.notifications_enabled
            and not self.notification_sent
        ):

            self.send_caregiver_notification(
                alert
            )

        return alert

    # ========================================================
    # SOS ALERT
    # ========================================================

    def create_sos_alert(
        self,
        risk_score: float = 100.0,
        location: str = "Location unavailable",
    ) -> Alert:
        """
        Create an immediate manual SOS alert.

        SOS is always CRITICAL.

        Repeated Streamlit reruns while the SOS remains active
        reuse the same alert.
        """

        alert = self.create_alert(
            alert_type=ALERT_TYPE_SOS,
            severity=SEVERITY_CRITICAL,
            title="🚨 SOS ALERT",
            message=(
                "Manual SOS activated. "
                f"Location: {location}"
            ),
            risk_score=risk_score,
            activity="SOS",
        )

        # ----------------------------------------------------
        # NOTIFICATION
        # ----------------------------------------------------

        if (
            self.notifications_enabled
            and not self.notification_sent
        ):

            self.send_caregiver_notification(
                alert
            )

        return alert

    # ========================================================
    # CAREGIVER NOTIFICATION
    # ========================================================

    def send_caregiver_notification(
        self,
        alert: Alert,
    ) -> Dict[str, Any]:
        """
        Simulate a caregiver notification.

        Only one notification is sent for an active event.
        """

        if not self.notifications_enabled:

            return {
                "success": False,
                "recipient": "Parent / Caregiver",
                "notification_type": alert.alert_type,
                "message": (
                    "Caregiver notifications "
                    "are disabled."
                ),
                "timestamp": self._timestamp(),
            }

        # ----------------------------------------------------
        # DUPLICATE NOTIFICATION PROTECTION
        # ----------------------------------------------------

        if self.notification_sent:

            return {
                "success": True,
                "already_sent": True,
                "recipient": "Parent / Caregiver",
                "notification_type": alert.alert_type,
                "message": (
                    "Caregiver notification already "
                    "sent for this active event."
                ),
                "timestamp": alert.timestamp,
            }

        self.notification_sent = True

        return {
            "success": True,
            "already_sent": False,
            "recipient": "Parent / Caregiver",
            "notification_type": alert.alert_type,
            "message": (
                f"{alert.title}: "
                f"{alert.message}"
            ),
            "timestamp": alert.timestamp,
        }

    # ========================================================
    # ALERT RESOLUTION
    # ========================================================

    def clear_current_alert(self) -> None:
        """
        Mark the current alert as resolved.

        History is preserved.
        """

        if self.last_alert is not None:

            self.last_alert.status = (
                ALERT_STATUS_RESOLVED
            )

        self.notification_sent = False

    def resolve_alert(
        self,
        alert: Optional[Alert] = None,
    ) -> bool:
        """
        Resolve a specific alert.

        If no alert is supplied, resolve the current alert.
        """

        target = (
            alert
            if alert is not None
            else self.last_alert
        )

        if target is None:
            return False

        target.status = (
            ALERT_STATUS_RESOLVED
        )

        if target is self.last_alert:
            self.notification_sent = False

        return True

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(
        self,
    ) -> List[Alert]:
        """
        Return alert history.

        A new list is returned so callers cannot accidentally
        modify the internal history container.
        """

        return list(
            self.alert_history
        )

    def get_history_dict(
        self,
    ) -> List[Dict[str, Any]]:
        """Return alert history as dictionaries."""

        return [
            alert.to_dict()
            for alert in self.alert_history
        ]

    def clear_history(self) -> None:
        """Delete stored alert history."""

        self.alert_history.clear()

        self.last_alert = None

        self.notification_sent = False

    # ========================================================
    # CURRENT ALERT
    # ========================================================

    def get_current_alert(
        self,
    ) -> Optional[Dict[str, Any]]:
        """Return the current alert as a dictionary."""

        if self.last_alert is None:
            return None

        return self.last_alert.to_dict()

    def get_active_alerts(
        self,
    ) -> List[Dict[str, Any]]:
        """Return all currently active alerts."""

        return [
            alert.to_dict()
            for alert in self.alert_history
            if alert.status == ALERT_STATUS_ACTIVE
        ]

    # ========================================================
    # STATUS
    # ========================================================

    def has_active_alert(self) -> bool:
        """Return True if an active alert exists."""

        if self.last_alert is None:
            return False

        return (
            self.last_alert.status
            == ALERT_STATUS_ACTIVE
        )

    def has_active_emergency(self) -> bool:
        """Return True if a critical alert is active."""

        if self.last_alert is None:
            return False

        return (
            self.last_alert.severity
            == SEVERITY_CRITICAL
            and self.last_alert.status
            == ALERT_STATUS_ACTIVE
        )

    def was_notification_sent(self) -> bool:
        """Return whether a caregiver notification was simulated."""

        return self.notification_sent

    def get_statistics(
        self,
    ) -> Dict[str, Any]:
        """Return useful alert statistics."""

        active = sum(
            alert.status == ALERT_STATUS_ACTIVE
            for alert in self.alert_history
        )

        resolved = sum(
            alert.status == ALERT_STATUS_RESOLVED
            for alert in self.alert_history
        )

        emergencies = sum(
            alert.severity == SEVERITY_CRITICAL
            for alert in self.alert_history
        )

        return {
            "total_alerts": len(
                self.alert_history
            ),

            "active_alerts": active,

            "resolved_alerts": resolved,

            "critical_alerts": emergencies,

            "notification_sent": (
                self.notification_sent
            ),
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """Reset alert state and history."""

        self.clear_history()


# ============================================================
# GLOBAL ALERT MANAGER
# ============================================================

_alert_manager = AlertManager()


# ============================================================
# PUBLIC CONVENIENCE API
# ============================================================

def create_safety_alert(
    risk_score: float,
    risk_level: str,
    activity: str,
    reason: str,
) -> Dict[str, Any]:
    """Create a safety alert."""

    alert = _alert_manager.create_safety_alert(
        risk_score=risk_score,
        risk_level=risk_level,
        activity=activity,
        reason=reason,
    )

    return alert.to_dict()


def create_emergency_alert(
    activity: str,
    risk_score: float,
    reason: str,
) -> Dict[str, Any]:
    """Create an emergency alert."""

    alert = _alert_manager.create_emergency_alert(
        activity=activity,
        risk_score=risk_score,
        reason=reason,
    )

    return alert.to_dict()


def create_sos_alert(
    risk_score: float = 100.0,
    location: str = "Location unavailable",
) -> Dict[str, Any]:
    """Create an immediate SOS alert."""

    alert = _alert_manager.create_sos_alert(
        risk_score=risk_score,
        location=location,
    )

    return alert.to_dict()


def get_alert_history() -> List[Dict[str, Any]]:
    """Return all alert history."""

    return _alert_manager.get_history_dict()


def get_current_alert() -> Optional[Dict[str, Any]]:
    """Return the current alert."""

    return _alert_manager.get_current_alert()


def get_active_alerts() -> List[Dict[str, Any]]:
    """Return all active alerts."""

    return _alert_manager.get_active_alerts()


def has_active_alert() -> bool:
    """Return whether an active alert exists."""

    return _alert_manager.has_active_alert()


def has_active_emergency() -> bool:
    """Return whether an active emergency exists."""

    return _alert_manager.has_active_emergency()


def was_notification_sent() -> bool:
    """Return whether a caregiver notification was simulated."""

    return _alert_manager.was_notification_sent()


def get_alert_statistics() -> Dict[str, Any]:
    """Return alert statistics."""

    return _alert_manager.get_statistics()


def clear_alert() -> None:
    """Resolve the current alert."""

    _alert_manager.clear_current_alert()


def clear_alert_history() -> None:
    """Clear all stored alert history."""

    _alert_manager.clear_history()


def reset_alerts() -> None:
    """Reset the global alert manager."""

    _alert_manager.reset()