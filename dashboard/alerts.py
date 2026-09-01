"""
SAFEBAND AI - Alert Management Module

Handles:
- Safety-status messages
- Warning notifications
- Emergency alerts
- Caregiver notification simulation
- Alert history
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class Alert:
    """Represents a SAFEBAND safety alert."""

    timestamp: str
    alert_type: str
    severity: str
    title: str
    message: str
    status: str
    risk_score: float
    activity: str


class AlertManager:
    """
    Prototype alert-management system.

    Alert levels:
        INFO
        WARNING
        EMERGENCY

    The current implementation simulates caregiver notification.
    A real notification service can be connected later.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.alert_history: List[Alert] = []
        self.last_alert: Alert | None = None
        self.notification_sent = False

    # =========================================================
    # CREATE ALERT
    # =========================================================

    def create_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        risk_score: float = 0.0,
        activity: str = "UNKNOWN"
    ) -> Alert:
        """
        Create and store a new alert.
        """

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        alert = Alert(
            timestamp=timestamp,
            alert_type=alert_type.upper(),
            severity=severity.upper(),
            title=title,
            message=message,
            status="ACTIVE",
            risk_score=round(float(risk_score), 1),
            activity=activity.upper()
        )

        self.alert_history.insert(0, alert)

        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[
                :self.max_history
            ]

        self.last_alert = alert

        return alert

    # =========================================================
    # SAFETY ALERT
    # =========================================================

    def create_safety_alert(
        self,
        risk_score: float,
        risk_level: str,
        activity: str,
        reason: str
    ) -> Alert:
        """
        Create an alert based on the calculated risk level.
        """

        risk_level = risk_level.upper()

        if risk_level == "CRITICAL":
            return self.create_emergency_alert(
                activity=activity,
                risk_score=risk_score,
                reason=reason
            )

        if risk_level == "HIGH":
            return self.create_alert(
                alert_type="SAFETY",
                severity="HIGH",
                title="High Risk Detected",
                message=reason,
                risk_score=risk_score,
                activity=activity
            )

        if risk_level == "MODERATE":
            return self.create_alert(
                alert_type="SAFETY",
                severity="WARNING",
                title="Safety Warning",
                message=reason,
                risk_score=risk_score,
                activity=activity
            )

        return self.create_alert(
            alert_type="STATUS",
            severity="INFO",
            title="System Normal",
            message="All monitored parameters are within safe range.",
            risk_score=risk_score,
            activity=activity
        )

    # =========================================================
    # EMERGENCY ALERT
    # =========================================================

    def create_emergency_alert(
        self,
        activity: str,
        risk_score: float,
        reason: str
    ) -> Alert:
        """
        Create an emergency alert.
        """

        alert = self.create_alert(
            alert_type="EMERGENCY",
            severity="CRITICAL",
            title="🚨 EMERGENCY DETECTED",
            message=reason,
            risk_score=risk_score,
            activity=activity
        )

        self.send_caregiver_notification(alert)

        return alert

    # =========================================================
    # SOS ALERT
    # =========================================================

    def create_sos_alert(
        self,
        risk_score: float = 100.0,
        location: str = "Location unavailable"
    ) -> Alert:
        """
        Create an immediate SOS emergency alert.
        """

        alert = self.create_alert(
            alert_type="SOS",
            severity="CRITICAL",
            title="🚨 SOS ALERT",
            message=(
                f"Manual SOS activated. "
                f"Location: {location}"
            ),
            risk_score=risk_score,
            activity="SOS"
        )

        self.send_caregiver_notification(alert)

        return alert

    # =========================================================
    # CAREGIVER NOTIFICATION
    # =========================================================

    def send_caregiver_notification(
        self,
        alert: Alert
    ) -> Dict[str, Any]:
        """
        Simulate sending an emergency notification
        to a parent/caregiver.
        """

        self.notification_sent = True

        return {
            "success": True,
            "recipient": "Parent / Caregiver",
            "notification_type": alert.alert_type,
            "message": (
                f"{alert.title}: {alert.message}"
            ),
            "timestamp": alert.timestamp
        }

    # =========================================================
    # CLEAR ALERT
    # =========================================================

    def clear_current_alert(self) -> None:
        """Mark the current alert as resolved."""

        if self.last_alert is not None:
            self.last_alert.status = "RESOLVED"

        self.notification_sent = False

    # =========================================================
    # HISTORY
    # =========================================================

    def get_history(self) -> List[Alert]:
        """Return alert history."""

        return self.alert_history.copy()

    def get_history_dict(self) -> List[Dict[str, Any]]:
        """Return alert history as dictionaries."""

        return [
            {
                "timestamp": alert.timestamp,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "status": alert.status,
                "risk_score": alert.risk_score,
                "activity": alert.activity
            }
            for alert in self.alert_history
        ]

    # =========================================================
    # STATUS
    # =========================================================

    def get_current_alert(self) -> Dict[str, Any] | None:
        """Return the current active alert."""

        if self.last_alert is None:
            return None

        alert = self.last_alert

        return {
            "timestamp": alert.timestamp,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "title": alert.title,
            "message": alert.message,
            "status": alert.status,
            "risk_score": alert.risk_score,
            "activity": alert.activity
        }

    def has_active_emergency(self) -> bool:
        """Return True if an emergency alert is currently active."""

        if self.last_alert is None:
            return False

        return (
            self.last_alert.severity == "CRITICAL"
            and self.last_alert.status == "ACTIVE"
        )

    def was_notification_sent(self) -> bool:
        """Return whether a caregiver notification was simulated."""

        return self.notification_sent


# =============================================================
# GLOBAL ALERT MANAGER
# =============================================================

_alert_manager = AlertManager()


# =============================================================
# CONVENIENCE FUNCTIONS
# =============================================================

def create_safety_alert(
    risk_score: float,
    risk_level: str,
    activity: str,
    reason: str
) -> Dict[str, Any]:
    """Create a safety alert."""

    alert = _alert_manager.create_safety_alert(
        risk_score=risk_score,
        risk_level=risk_level,
        activity=activity,
        reason=reason
    )

    return {
        "timestamp": alert.timestamp,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "status": alert.status,
        "risk_score": alert.risk_score,
        "activity": alert.activity
    }


def create_emergency_alert(
    activity: str,
    risk_score: float,
    reason: str
) -> Dict[str, Any]:
    """Create an emergency alert."""

    alert = _alert_manager.create_emergency_alert(
        activity=activity,
        risk_score=risk_score,
        reason=reason
    )

    return {
        "timestamp": alert.timestamp,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "status": alert.status,
        "risk_score": alert.risk_score,
        "activity": alert.activity
    }


def create_sos_alert(
    risk_score: float = 100.0,
    location: str = "Location unavailable"
) -> Dict[str, Any]:
    """Create an SOS alert."""

    alert = _alert_manager.create_sos_alert(
        risk_score=risk_score,
        location=location
    )

    return {
        "timestamp": alert.timestamp,
        "alert_type": alert.alert_type,
        "severity": alert.severity,
        "title": alert.title,
        "message": alert.message,
        "status": alert.status,
        "risk_score": alert.risk_score,
        "activity": alert.activity
    }


def get_alert_history() -> List[Dict[str, Any]]:
    """Return all alert history."""

    return _alert_manager.get_history_dict()


def get_current_alert() -> Dict[str, Any] | None:
    """Return the current alert."""

    return _alert_manager.get_current_alert()


def clear_alert() -> None:
    """Clear the current active alert."""

    _alert_manager.clear_current_alert()