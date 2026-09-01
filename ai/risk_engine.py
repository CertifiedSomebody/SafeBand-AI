"""
SAFEBAND AI - Risk Assessment Engine

Combines physiological, motion, activity, and environmental
information into a normalized SAFEBAND safety-risk assessment.

Current implementation:
    Deterministic rule-based assessment.

Architecture:
    Sensor data
        ↓
    Activity recognition
        ↓
    Risk assessment
        ↓
    Risk level + alert decision

Important sensor distinction:

    temperature
        BME680 environmental temperature

    body_temperature
        MAX30208 body temperature

    heart_rate
        MAX30102

    spo2
        MAX30102

    motion / orientation
        BNO055

Manual SOS is treated as an explicit emergency input.

The engine intentionally does NOT use the demonstration
scenario name to calculate risk.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List


# ============================================================
# RISK CONFIGURATION
# ============================================================

class RiskConfig:
    """Centralized thresholds and scoring weights."""

    # --------------------------------------------------------
    # RISK LEVEL BOUNDARIES
    # --------------------------------------------------------

    MODERATE_THRESHOLD = 30.0
    HIGH_THRESHOLD = 60.0
    CRITICAL_THRESHOLD = 80.0

    # --------------------------------------------------------
    # EMERGENCY EVENTS
    # --------------------------------------------------------

    # Base contribution for an automatically detected fall.
    FALL_SCORE = 65.0

    # Manual SOS is an explicit user emergency action.
    SOS_SCORE = 100.0

    # Emergency activity score floor.
    # This ensures a detected fall remains CRITICAL even if
    # other sensor values happen to be normal.
    EMERGENCY_MINIMUM_SCORE = 80.0

    # --------------------------------------------------------
    # HEART RATE
    # --------------------------------------------------------

    HEART_RATE_HIGH = 130.0
    HEART_RATE_ELEVATED = 110.0
    HEART_RATE_LOW = 50.0

    HEART_RATE_HIGH_SCORE = 15.0
    HEART_RATE_ELEVATED_SCORE = 8.0
    HEART_RATE_LOW_SCORE = 12.0

    # --------------------------------------------------------
    # SpO2
    # --------------------------------------------------------

    SPO2_CRITICAL = 90.0
    SPO2_REDUCED = 94.0

    SPO2_CRITICAL_SCORE = 20.0
    SPO2_REDUCED_SCORE = 10.0

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    MOTION_HIGH = 2.0
    MOTION_ELEVATED = 1.2

    MOTION_HIGH_SCORE = 10.0
    MOTION_ELEVATED_SCORE = 5.0

    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    ORIENTATION_HIGH = 75.0
    ORIENTATION_UNUSUAL = 60.0

    ORIENTATION_HIGH_SCORE = 10.0
    ORIENTATION_UNUSUAL_SCORE = 5.0

    # --------------------------------------------------------
    # ENVIRONMENTAL TEMPERATURE
    # --------------------------------------------------------

    ENVIRONMENT_HIGH = 45.0
    ENVIRONMENT_LOW = 5.0

    ENVIRONMENT_HIGH_SCORE = 10.0
    ENVIRONMENT_LOW_SCORE = 5.0

    # --------------------------------------------------------
    # BODY TEMPERATURE
    # --------------------------------------------------------

    BODY_TEMPERATURE_HIGH = 38.0
    BODY_TEMPERATURE_CRITICAL = 39.5

    BODY_TEMPERATURE_LOW = 35.0
    BODY_TEMPERATURE_CRITICAL_LOW = 34.0

    BODY_TEMPERATURE_HIGH_SCORE = 8.0
    BODY_TEMPERATURE_CRITICAL_SCORE = 12.0
    BODY_TEMPERATURE_LOW_SCORE = 8.0
    BODY_TEMPERATURE_CRITICAL_LOW_SCORE = 12.0

    # --------------------------------------------------------
    # ACTIVITY CONTEXT
    # --------------------------------------------------------

    RUNNING_SCORE = 2.0
    HIGH_RISK_ACTIVITY_SCORE = 30.0
    UNKNOWN_ACTIVITY_SCORE = 3.0


# ============================================================
# RISK RESULT
# ============================================================

@dataclass
class RiskResult:
    """Result produced by the risk assessment engine."""

    risk_score: float
    risk_level: str
    status: str
    reason: str
    emergency: bool
    alert_required: bool

    # Structured reasons are useful for dashboard,
    # logging and future ML explainability.
    reasons: List[str] = field(
        default_factory=list
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return the result in application-friendly format."""

        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "status": self.status,
            "reason": self.reason,
            "emergency": self.emergency,
            "alert_required": self.alert_required,
            "reasons": list(self.reasons),
        }


# ============================================================
# RISK ENGINE
# ============================================================

class RiskEngine:
    """
    SAFEBAND AI risk assessment engine.

    Risk levels:

        0-29    LOW
        30-59   MODERATE
        60-79   HIGH
        80-100  CRITICAL

    The engine uses deterministic multi-factor scoring.

    Emergency conditions:
        - Manual SOS
        - Detected FALL
        - Activity-recognition emergency flag
        - Risk score >= CRITICAL_THRESHOLD
    """

    def __init__(self) -> None:
        """Initialize the risk engine."""

        self.last_result = self._safe_result()

    # ========================================================
    # DEFAULT / SAFE STATE
    # ========================================================

    @staticmethod
    def _safe_result() -> RiskResult:
        """Return a default safe result."""

        return RiskResult(
            risk_score=0.0,
            risk_level="LOW",
            status="SAFE",
            reason="No abnormal event detected.",
            emergency=False,
            alert_required=False,
            reasons=[],
        )

    # ========================================================
    # INPUT HELPERS
    # ========================================================

    @staticmethod
    def _number(
        data: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        """Safely retrieve a numeric value."""

        try:
            value = data.get(
                key,
                default,
            )

            if value is None:
                return default

            return float(value)

        except (TypeError, ValueError):
            return default

    @staticmethod
    def _boolean(
        data: Dict[str, Any],
        key: str,
        default: bool = False,
    ) -> bool:
        """Safely retrieve a boolean value."""

        value = data.get(
            key,
            default,
        )

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            return value.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
            }

        return bool(value)

    # ========================================================
    # SENSOR ASSESSMENT
    # ========================================================

    def _assess_heart_rate(
        self,
        heart_rate: float,
        reasons: List[str],
    ) -> float:
        """Assess heart-rate contribution to risk."""

        if heart_rate >= RiskConfig.HEART_RATE_HIGH:

            reasons.append(
                "Elevated heart rate"
            )

            return RiskConfig.HEART_RATE_HIGH_SCORE

        if heart_rate >= RiskConfig.HEART_RATE_ELEVATED:

            reasons.append(
                "Increased heart rate"
            )

            return RiskConfig.HEART_RATE_ELEVATED_SCORE

        if heart_rate < RiskConfig.HEART_RATE_LOW:

            reasons.append(
                "Low heart rate"
            )

            return RiskConfig.HEART_RATE_LOW_SCORE

        return 0.0

    def _assess_spo2(
        self,
        oxygen: float,
        reasons: List[str],
    ) -> float:
        """Assess blood-oxygen contribution to risk."""

        if oxygen < RiskConfig.SPO2_CRITICAL:

            reasons.append(
                "Low blood oxygen level"
            )

            return RiskConfig.SPO2_CRITICAL_SCORE

        if oxygen < RiskConfig.SPO2_REDUCED:

            reasons.append(
                "Reduced blood oxygen level"
            )

            return RiskConfig.SPO2_REDUCED_SCORE

        return 0.0

    def _assess_motion(
        self,
        motion_intensity: float,
        reasons: List[str],
    ) -> float:
        """Assess motion contribution to risk."""

        if motion_intensity >= RiskConfig.MOTION_HIGH:

            reasons.append(
                "High abnormal motion"
            )

            return RiskConfig.MOTION_HIGH_SCORE

        if motion_intensity >= RiskConfig.MOTION_ELEVATED:

            reasons.append(
                "Increased motion intensity"
            )

            return RiskConfig.MOTION_ELEVATED_SCORE

        return 0.0

    def _assess_orientation(
        self,
        orientation: float,
        reasons: List[str],
    ) -> float:
        """Assess body-orientation contribution to risk."""

        orientation_abs = abs(
            orientation
        )

        if orientation_abs >= RiskConfig.ORIENTATION_HIGH:

            reasons.append(
                "Abnormal body orientation"
            )

            return RiskConfig.ORIENTATION_HIGH_SCORE

        if orientation_abs >= RiskConfig.ORIENTATION_UNUSUAL:

            reasons.append(
                "Unusual body orientation"
            )

            return RiskConfig.ORIENTATION_UNUSUAL_SCORE

        return 0.0

    def _assess_environment(
        self,
        ambient_temperature: float,
        reasons: List[str],
    ) -> float:
        """
        Assess BME680 environmental temperature.

        This is deliberately separate from MAX30208 body
        temperature.
        """

        if ambient_temperature >= RiskConfig.ENVIRONMENT_HIGH:

            reasons.append(
                "High environmental temperature"
            )

            return RiskConfig.ENVIRONMENT_HIGH_SCORE

        if ambient_temperature <= RiskConfig.ENVIRONMENT_LOW:

            reasons.append(
                "Low environmental temperature"
            )

            return RiskConfig.ENVIRONMENT_LOW_SCORE

        return 0.0

    def _assess_body_temperature(
        self,
        body_temperature: float,
        reasons: List[str],
    ) -> float:
        """
        Assess MAX30208 body temperature.

        This signal is kept separate from BME680 environmental
        temperature.
        """

        if (
            body_temperature
            >= RiskConfig.BODY_TEMPERATURE_CRITICAL
        ):

            reasons.append(
                "Critically elevated body temperature"
            )

            return (
                RiskConfig.BODY_TEMPERATURE_CRITICAL_SCORE
            )

        if (
            body_temperature
            >= RiskConfig.BODY_TEMPERATURE_HIGH
        ):

            reasons.append(
                "Elevated body temperature"
            )

            return RiskConfig.BODY_TEMPERATURE_HIGH_SCORE

        if (
            body_temperature
            <= RiskConfig.BODY_TEMPERATURE_CRITICAL_LOW
        ):

            reasons.append(
                "Critically low body temperature"
            )

            return (
                RiskConfig.BODY_TEMPERATURE_CRITICAL_LOW_SCORE
            )

        if (
            body_temperature
            <= RiskConfig.BODY_TEMPERATURE_LOW
        ):

            reasons.append(
                "Low body temperature"
            )

            return RiskConfig.BODY_TEMPERATURE_LOW_SCORE

        return 0.0

    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_score(
        score: float,
        emergency: bool = False,
    ) -> Dict[str, str]:
        """
        Convert numeric score into risk classification.

        Explicit emergency events always receive CRITICAL
        classification.
        """

        if (
            emergency
            or score >= RiskConfig.CRITICAL_THRESHOLD
        ):

            return {
                "risk_level": "CRITICAL",
                "status": "EMERGENCY",
            }

        if score >= RiskConfig.HIGH_THRESHOLD:

            return {
                "risk_level": "HIGH",
                "status": "WARNING",
            }

        if score >= RiskConfig.MODERATE_THRESHOLD:

            return {
                "risk_level": "MODERATE",
                "status": "WARNING",
            }

        return {
            "risk_level": "LOW",
            "status": "SAFE",
        }

    # ========================================================
    # MAIN CALCULATION
    # ========================================================

    def calculate_risk(
        self,
        sensor_data: Dict[str, Any],
        activity_result: Dict[str, Any],
    ) -> RiskResult:
        """
        Calculate overall SAFEBAND safety risk.

        Parameters
        ----------
        sensor_data:
            Current sensor readings.

        activity_result:
            Output from the activity-recognition engine.

        Returns
        -------
        RiskResult:
            Normalized safety assessment.
        """

        if not isinstance(
            sensor_data,
            dict,
        ):

            sensor_data = {}

        if not isinstance(
            activity_result,
            dict,
        ):

            activity_result = {}

        score = 0.0

        reasons: List[str] = []

        # ====================================================
        # SENSOR VALUES
        # ====================================================

        heart_rate = self._number(
            sensor_data,
            "heart_rate",
            75.0,
        )

        oxygen = self._number(
            sensor_data,
            "spo2",
            98.0,
        )

        # BME680 environmental temperature.
        ambient_temperature = self._number(
            sensor_data,
            "temperature",
            25.0,
        )

        # MAX30208 body temperature.
        body_temperature = self._number(
            sensor_data,
            "body_temperature",
            36.7,
        )

        motion_intensity = self._number(
            sensor_data,
            "motion_intensity",
            0.0,
        )

        orientation = self._number(
            sensor_data,
            "orientation",
            0.0,
        )

        # ====================================================
        # ACTIVITY
        # ====================================================

        activity = str(
            activity_result.get(
                "activity",
                "UNKNOWN",
            )
        ).upper()

        activity_emergency = self._boolean(
            activity_result,
            "emergency",
            False,
        )

        # ====================================================
        # MANUAL SOS
        # ====================================================

        manual_sos = self._boolean(
            sensor_data,
            "manual_sos",
            False,
        )

        # Also accept an explicit SOS flag from the activity
        # result if another application layer supplies one.
        activity_sos = (
            activity == "SOS"
        )

        if manual_sos or activity_sos:

            score = RiskConfig.SOS_SCORE

            reasons.append(
                "Manual SOS emergency activated"
            )

            emergency = True

            classification = self._classify_score(
                score,
                emergency=True,
            )

            reason = "; ".join(
                reasons
            )

            result = RiskResult(
                risk_score=100.0,
                risk_level="CRITICAL",
                status="EMERGENCY",
                reason=reason,
                emergency=True,
                alert_required=True,
                reasons=reasons,
            )

            self.last_result = result

            return result

        # ====================================================
        # FALL / AUTOMATIC EMERGENCY
        # ====================================================

        fall_detected = (
            activity == "FALL"
            or activity_emergency
        )

        if fall_detected:

            score += RiskConfig.FALL_SCORE

            reasons.append(
                "Abnormal fall/emergency event detected"
            )

        # ====================================================
        # HEART RATE
        # ====================================================

        score += self._assess_heart_rate(
            heart_rate,
            reasons,
        )

        # ====================================================
        # SPO2
        # ====================================================

        score += self._assess_spo2(
            oxygen,
            reasons,
        )

        # ====================================================
        # MOTION
        # ====================================================

        score += self._assess_motion(
            motion_intensity,
            reasons,
        )

        # ====================================================
        # ORIENTATION
        # ====================================================

        score += self._assess_orientation(
            orientation,
            reasons,
        )

        # ====================================================
        # ENVIRONMENT
        # ====================================================

        score += self._assess_environment(
            ambient_temperature,
            reasons,
        )

        # ====================================================
        # BODY TEMPERATURE
        # ====================================================

        score += self._assess_body_temperature(
            body_temperature,
            reasons,
        )

        # ====================================================
        # ACTIVITY CONTEXT
        # ====================================================

        if activity == "HIGH_RISK":

            score += RiskConfig.HIGH_RISK_ACTIVITY_SCORE

            reasons.append(
                "Activity recognition indicates high-risk condition"
            )

        elif activity == "RUNNING":

            score += RiskConfig.RUNNING_SCORE

        elif activity == "UNKNOWN":

            score += RiskConfig.UNKNOWN_ACTIVITY_SCORE

            reasons.append(
                "Activity could not be confidently classified"
            )

        # ====================================================
        # EMERGENCY DECISION
        # ====================================================

        emergency = (
            fall_detected
            or score >= RiskConfig.CRITICAL_THRESHOLD
        )

        # A detected FALL is an emergency even if the current
        # additive sensor score is below 80.
        if fall_detected:

            score = max(
                score,
                RiskConfig.EMERGENCY_MINIMUM_SCORE,
            )

        # ====================================================
        # LIMIT SCORE
        # ====================================================

        score = min(
            100.0,
            max(
                0.0,
                score,
            ),
        )

        # ====================================================
        # RISK LEVEL
        # ====================================================

        classification = self._classify_score(
            score,
            emergency=emergency,
        )

        risk_level = classification[
            "risk_level"
        ]

        status = classification[
            "status"
        ]

        # ====================================================
        # ALERT DECISION
        # ====================================================

        alert_required = (
            emergency
            or score >= RiskConfig.HIGH_THRESHOLD
        )

        # ====================================================
        # REASON
        # ====================================================

        if reasons:

            reason = "; ".join(
                reasons
            )

        else:

            reason = (
                "All monitored parameters "
                "are within safe range."
            )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = RiskResult(
            risk_score=round(
                score,
                1,
            ),
            risk_level=risk_level,
            status=status,
            reason=reason,
            emergency=emergency,
            alert_required=alert_required,
            reasons=reasons,
        )

        self.last_result = result

        return result

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return the latest risk assessment."""

        return self.last_result.as_dict()

    def reset(
        self,
    ) -> None:
        """Reset the engine to a safe initial state."""

        self.last_result = self._safe_result()


# ============================================================
# GLOBAL ENGINE
# ============================================================

_risk_engine = RiskEngine()


# ============================================================
# PUBLIC CONVENIENCE API
# ============================================================

def assess_risk(
    sensor_data: Dict[str, Any],
    activity_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assess SAFEBAND risk using the shared risk engine.

    Example:

        result = assess_risk(
            {
                "heart_rate": 112,
                "spo2": 96,
                "temperature": 28,
                "body_temperature": 36.8,
                "motion_intensity": 2.5,
                "orientation": 80,
            },
            {
                "activity": "FALL",
                "emergency": True,
            },
        )
    """

    result = _risk_engine.calculate_risk(
        sensor_data,
        activity_result,
    )

    return result.as_dict()


def get_risk_status() -> Dict[str, Any]:
    """Return the latest global risk status."""

    return _risk_engine.get_status()


def reset_risk_engine() -> None:
    """Reset the global risk engine."""

    _risk_engine.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "RiskConfig",
    "RiskResult",
    "RiskEngine",
    "assess_risk",
    "get_risk_status",
    "reset_risk_engine",
]