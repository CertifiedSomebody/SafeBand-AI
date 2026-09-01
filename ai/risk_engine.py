"""
SAFEBAND AI - Risk Assessment Engine

Prototype risk engine for combining activity, physiological,
motion and environmental information into a single safety score.

The current implementation is rule-based and intended for
demonstration. It can later be replaced or enhanced with a
trained AI/TinyML model.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RiskResult:
    """Result produced by the risk assessment engine."""

    risk_score: float
    risk_level: str
    status: str
    reason: str
    emergency: bool
    alert_required: bool


class RiskEngine:
    """
    SAFEBAND AI prototype risk assessment engine.

    Risk levels:
        0-29   -> LOW
        30-59  -> MODERATE
        60-79  -> HIGH
        80-100 -> CRITICAL
    """

    def __init__(self):
        self.last_result = RiskResult(
            risk_score=0.0,
            risk_level="LOW",
            status="SAFE",
            reason="No abnormal event detected.",
            emergency=False,
            alert_required=False
        )

    @staticmethod
    def _number(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely retrieve a numeric value."""
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def calculate_risk(
        self,
        sensor_data: Dict[str, Any],
        activity_result: Dict[str, Any]
    ) -> RiskResult:
        """
        Calculate overall safety risk from sensor and activity data.

        Parameters
        ----------
        sensor_data : dict
            Current sensor readings.

        activity_result : dict
            Output from the activity-recognition engine.

        Returns
        -------
        RiskResult
            Overall risk assessment.
        """

        score = 0.0
        reasons = []

        # ---------------------------------------------------------
        # SENSOR VALUES
        # ---------------------------------------------------------

        heart_rate = self._number(
            sensor_data,
            "heart_rate",
            75.0
        )

        temperature = self._number(
            sensor_data,
            "temperature",
            25.0
        )

        motion_intensity = self._number(
            sensor_data,
            "motion_intensity",
            0.0
        )

        orientation = self._number(
            sensor_data,
            "orientation",
            0.0
        )

        oxygen = self._number(
            sensor_data,
            "spo2",
            98.0
        )

        # ---------------------------------------------------------
        # ACTIVITY INFORMATION
        # ---------------------------------------------------------

        activity = str(
            activity_result.get("activity", "UNKNOWN")
        ).upper()

        activity_emergency = bool(
            activity_result.get("emergency", False)
        )

        # ---------------------------------------------------------
        # 1. FALL / EMERGENCY EVENT
        # ---------------------------------------------------------

        if activity == "FALL" or activity_emergency:
            score += 65
            reasons.append("Abnormal fall/emergency event detected")

        # ---------------------------------------------------------
        # 2. HEART RATE ASSESSMENT
        # ---------------------------------------------------------

        if heart_rate >= 130:
            score += 15
            reasons.append("Elevated heart rate")

        elif heart_rate >= 110:
            score += 8
            reasons.append("Increased heart rate")

        elif heart_rate < 50:
            score += 12
            reasons.append("Low heart rate")

        # ---------------------------------------------------------
        # 3. SPO2 ASSESSMENT
        # ---------------------------------------------------------

        if oxygen < 90:
            score += 20
            reasons.append("Low blood oxygen level")

        elif oxygen < 94:
            score += 10
            reasons.append("Reduced blood oxygen level")

        # ---------------------------------------------------------
        # 4. MOTION ASSESSMENT
        # ---------------------------------------------------------

        if motion_intensity >= 2.0:
            score += 10
            reasons.append("High abnormal motion")

        elif motion_intensity >= 1.2:
            score += 5
            reasons.append("Increased motion intensity")

        # ---------------------------------------------------------
        # 5. ORIENTATION ASSESSMENT
        # ---------------------------------------------------------

        if abs(orientation) >= 75:
            score += 10
            reasons.append("Abnormal body orientation")

        elif abs(orientation) >= 60:
            score += 5
            reasons.append("Unusual body orientation")

        # ---------------------------------------------------------
        # 6. ENVIRONMENTAL ASSESSMENT
        # ---------------------------------------------------------

        if temperature >= 45:
            score += 10
            reasons.append("High environmental temperature")

        elif temperature <= 5:
            score += 5
            reasons.append("Low environmental temperature")

        # ---------------------------------------------------------
        # 7. ACTIVITY CONTEXT
        # ---------------------------------------------------------

        if activity == "RUNNING":
            # Running itself is not an emergency.
            # Only add a small contextual contribution.
            score += 2

        elif activity == "UNKNOWN":
            score += 3
            reasons.append("Activity could not be confidently classified")

        # ---------------------------------------------------------
        # LIMIT SCORE
        # ---------------------------------------------------------

        score = min(100.0, max(0.0, score))

        # ---------------------------------------------------------
        # RISK CLASSIFICATION
        # ---------------------------------------------------------

        if score >= 80:
            risk_level = "CRITICAL"
            status = "EMERGENCY"

        elif score >= 60:
            risk_level = "HIGH"
            status = "WARNING"

        elif score >= 30:
            risk_level = "MODERATE"
            status = "WARNING"

        else:
            risk_level = "LOW"
            status = "SAFE"

        # ---------------------------------------------------------
        # EMERGENCY DECISION
        # ---------------------------------------------------------

        emergency = (
            activity == "FALL"
            or activity_emergency
            or score >= 80
        )

        alert_required = (
            emergency
            or score >= 60
        )

        # ---------------------------------------------------------
        # REASON
        # ---------------------------------------------------------

        if reasons:
            reason = "; ".join(reasons)
        else:
            reason = "All monitored parameters are within safe range."

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------

        result = RiskResult(
            risk_score=round(score, 1),
            risk_level=risk_level,
            status=status,
            reason=reason,
            emergency=emergency,
            alert_required=alert_required
        )

        self.last_result = result

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return the latest risk assessment."""

        return {
            "risk_score": self.last_result.risk_score,
            "risk_level": self.last_result.risk_level,
            "status": self.last_result.status,
            "reason": self.last_result.reason,
            "emergency": self.last_result.emergency,
            "alert_required": self.last_result.alert_required
        }


def assess_risk(
    sensor_data: Dict[str, Any],
    activity_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function for the SAFEBAND AI application.

    Example
    -------
    result = assess_risk(
        {
            "heart_rate": 112,
            "spo2": 96,
            "temperature": 28,
            "motion_intensity": 2.5,
            "orientation": 80
        },
        {
            "activity": "FALL",
            "emergency": True
        }
    )
    """

    engine = RiskEngine()

    result = engine.calculate_risk(
        sensor_data,
        activity_result
    )

    return {
        "risk_score": result.risk_score,
        "risk_level": result.risk_level,
        "status": result.status,
        "reason": result.reason,
        "emergency": result.emergency,
        "alert_required": result.alert_required
    }