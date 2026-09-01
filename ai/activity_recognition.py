"""
SAFEBAND AI - Activity Recognition Engine

Rule-based activity recognition for the SAFEBAND AI prototype.

This module classifies activity from sensor measurements only.

IMPORTANT
---------
The classifier MUST NOT use the simulation profile/scenario name
to determine activity.

The scenario field may exist in demonstration data, but it is
treated as metadata only.

Current classifier:
    Rule-based multi-sensor classification

Future classifier:
    TinyML / machine-learning inference can replace the internal
    classification method without changing the public interface.

Expected sensor inputs:

    BNO055
        acceleration_x
        acceleration_y
        acceleration_z
        motion_intensity
        orientation

    MAX30102
        heart_rate
        spo2

    MAX30208
        body_temperature

Activities:

    NORMAL
    SITTING
    STANDING
    WALKING
    RUNNING
    HIGH_RISK
    FALL
    UNKNOWN
"""


from dataclasses import dataclass
import math
from typing import Any, Dict


# ============================================================
# ACTIVITY CONFIGURATION
# ============================================================

class ActivityConfig:
    """
    Thresholds used by the prototype rule-based classifier.

    Classification is based entirely on sensor measurements.
    """

    # --------------------------------------------------------
    # FALL DETECTION
    # --------------------------------------------------------

    FALL_ACCELERATION = 2.50
    FALL_ORIENTATION = 60.0

    # --------------------------------------------------------
    # HIGH-RISK DETECTION
    # --------------------------------------------------------

    HIGH_RISK_HEART_RATE = 120.0
    HIGH_RISK_SPO2 = 94.0
    HIGH_RISK_BODY_TEMPERATURE = 38.0
    HIGH_RISK_ORIENTATION = 45.0
    HIGH_RISK_MOTION = 1.60

    # Minimum number of abnormal physiological/physical
    # indicators required for HIGH_RISK classification.
    HIGH_RISK_INDICATORS = 2

    # --------------------------------------------------------
    # RUNNING DETECTION
    # --------------------------------------------------------

    RUNNING_MOTION = 1.20
    RUNNING_ACCELERATION = 1.55

    # --------------------------------------------------------
    # WALKING DETECTION
    # --------------------------------------------------------

    WALKING_MOTION = 0.25
    WALKING_ACCELERATION_MIN = 1.05
    WALKING_ACCELERATION_MAX = 1.50

    # --------------------------------------------------------
    # STANDING DETECTION
    # --------------------------------------------------------

    STANDING_MOTION = 0.25
    STANDING_ACCELERATION_MIN = 0.85
    STANDING_ACCELERATION_MAX = 1.15

    # --------------------------------------------------------
    # SITTING DETECTION
    # --------------------------------------------------------

    SITTING_MOTION = 0.15
    SITTING_HEART_RATE_MAX = 100.0
    SITTING_ORIENTATION_MAX = 45.0

    # --------------------------------------------------------
    # NORMAL STATE
    # --------------------------------------------------------

    NORMAL_MOTION_MAX = 0.25
    NORMAL_ACCELERATION_MIN = 0.85
    NORMAL_ACCELERATION_MAX = 1.15
    NORMAL_HEART_RATE_MIN = 50.0
    NORMAL_HEART_RATE_MAX = 100.0
    NORMAL_SPO2_MIN = 95.0
    NORMAL_BODY_TEMPERATURE_MIN = 35.0
    NORMAL_BODY_TEMPERATURE_MAX = 37.8


# ============================================================
# ACTIVITY RESULT
# ============================================================

@dataclass
class ActivityResult:
    """Result returned by the activity-recognition engine."""

    activity: str
    confidence: float
    description: str
    emergency: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return the result as a dictionary."""

        return {
            "activity": self.activity,
            "confidence": round(
                self.confidence,
                2,
            ),
            "description": self.description,
            "emergency": self.emergency,
        }


# ============================================================
# ACTIVITY RECOGNIZER
# ============================================================

class ActivityRecognizer:
    """
    SAFEBAND activity-recognition engine.

    Classification is performed exclusively from sensor data.

    The simulation scenario name is deliberately ignored.

    Activities:

        NORMAL
        SITTING
        STANDING
        WALKING
        RUNNING
        HIGH_RISK
        FALL
        UNKNOWN
    """

    ACTIVITIES = {
        "NORMAL": (
            "User is stationary with physiological parameters "
            "within the normal range."
        ),

        "SITTING": (
            "User appears stationary and seated."
        ),

        "STANDING": (
            "User is upright with low movement."
        ),

        "WALKING": (
            "User is walking normally."
        ),

        "RUNNING": (
            "User is performing high-intensity movement."
        ),

        "HIGH_RISK": (
            "Multiple abnormal physiological or motion "
            "parameters indicate elevated safety risk."
        ),

        "FALL": (
            "Sudden abnormal motion and orientation are "
            "consistent with a possible fall."
        ),

        "UNKNOWN": (
            "Activity could not be classified."
        ),
    }

    def __init__(self) -> None:
        """Initialize the activity recognizer."""

        self.last_result = ActivityResult(
            activity="UNKNOWN",
            confidence=0.0,
            description=self.ACTIVITIES["UNKNOWN"],
            emergency=False,
        )

    # ========================================================
    # INPUT HELPERS
    # ========================================================

    @staticmethod
    def _get(
        data: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        """
        Safely retrieve a numeric sensor value.

        Invalid, missing, None, or non-numeric values fall back
        to the supplied default.
        """

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

    @classmethod
    def _calculate_acceleration_magnitude(
        cls,
        data: Dict[str, Any],
    ) -> float:
        """
        Calculate resultant acceleration magnitude.

        Formula:

            A = sqrt(Ax² + Ay² + Az²)
        """

        ax = cls._get(
            data,
            "acceleration_x",
        )

        ay = cls._get(
            data,
            "acceleration_y",
        )

        az = cls._get(
            data,
            "acceleration_z",
        )

        return math.sqrt(
            ax ** 2
            + ay ** 2
            + az ** 2
        )

    # ========================================================
    # RESULT CREATION
    # ========================================================

    @classmethod
    def _result(
        cls,
        activity: str,
        confidence: float,
        emergency: bool = False,
    ) -> ActivityResult:
        """Create a normalized activity result."""

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        return ActivityResult(
            activity=activity,
            confidence=confidence,
            description=cls.ACTIVITIES.get(
                activity,
                cls.ACTIVITIES["UNKNOWN"],
            ),
            emergency=emergency,
        )

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    def _classify(
        self,
        sensor_data: Dict[str, Any],
    ) -> ActivityResult:
        """
        Perform multi-sensor rule-based activity classification.

        IMPORTANT
        ---------
        The `scenario` field is intentionally never accessed.

        Classification is based only on sensor measurements.
        """

        # ----------------------------------------------------
        # SENSOR VALUES
        # ----------------------------------------------------

        acceleration = (
            self._calculate_acceleration_magnitude(
                sensor_data
            )
        )

        motion_intensity = self._get(
            sensor_data,
            "motion_intensity",
            abs(acceleration - 1.0),
        )

        heart_rate = self._get(
            sensor_data,
            "heart_rate",
            75.0,
        )

        spo2 = self._get(
            sensor_data,
            "spo2",
            98.0,
        )

        body_temperature = self._get(
            sensor_data,
            "body_temperature",
            36.7,
        )

        orientation = self._get(
            sensor_data,
            "orientation",
            0.0,
        )

        orientation_abs = abs(
            orientation
        )

        # ====================================================
        # 1. FALL
        # ====================================================

        # Fall requires BOTH:
        #
        #   - unusually high acceleration
        #   - abnormal body orientation
        #
        # This prevents ordinary running from being interpreted
        # as a fall.

        if (
            acceleration
            >= ActivityConfig.FALL_ACCELERATION
            and orientation_abs
            >= ActivityConfig.FALL_ORIENTATION
        ):

            return self._result(
                activity="FALL",
                confidence=0.96,
                emergency=True,
            )

        # ====================================================
        # 2. HIGH RISK
        # ====================================================

        # HIGH_RISK is inferred from multiple independent
        # abnormal measurements.
        #
        # This is deliberately checked BEFORE RUNNING because
        # high-risk physiological conditions may also contain
        # high motion.

        high_risk_indicators = 0

        # Elevated heart rate.
        if heart_rate >= ActivityConfig.HIGH_RISK_HEART_RATE:
            high_risk_indicators += 1

        # Reduced oxygen saturation.
        if spo2 <= ActivityConfig.HIGH_RISK_SPO2:
            high_risk_indicators += 1

        # Elevated physiological/body temperature.
        if (
            body_temperature
            >= ActivityConfig.HIGH_RISK_BODY_TEMPERATURE
        ):
            high_risk_indicators += 1

        # Abnormal body orientation.
        if (
            orientation_abs
            >= ActivityConfig.HIGH_RISK_ORIENTATION
        ):
            high_risk_indicators += 1

        # Very high motion without satisfying the fall pattern.
        if (
            motion_intensity
            >= ActivityConfig.HIGH_RISK_MOTION
        ):
            high_risk_indicators += 1

        if (
            high_risk_indicators
            >= ActivityConfig.HIGH_RISK_INDICATORS
        ):

            confidence = min(
                0.97,
                0.72
                + (
                    high_risk_indicators
                    * 0.05
                ),
            )

            return self._result(
                activity="HIGH_RISK",
                confidence=confidence,
                emergency=False,
            )

        # ====================================================
        # 3. RUNNING
        # ====================================================

        if (
            motion_intensity
            >= ActivityConfig.RUNNING_MOTION
            or acceleration
            >= ActivityConfig.RUNNING_ACCELERATION
        ):

            motion_component = min(
                1.0,
                motion_intensity / 2.0,
            )

            acceleration_component = min(
                1.0,
                acceleration / 2.0,
            )

            confidence = min(
                0.95,
                0.70
                + (
                    0.15
                    * motion_component
                )
                + (
                    0.10
                    * acceleration_component
                ),
            )

            return self._result(
                activity="RUNNING",
                confidence=confidence,
            )

        # ====================================================
        # 4. WALKING
        # ====================================================

        walking_motion = (
            motion_intensity
            >= ActivityConfig.WALKING_MOTION
        )

        walking_acceleration = (
            ActivityConfig.WALKING_ACCELERATION_MIN
            <= acceleration
            <= ActivityConfig.WALKING_ACCELERATION_MAX
        )

        if (
            walking_motion
            or walking_acceleration
        ):

            confidence = min(
                0.95,
                0.78
                + (
                    min(
                        motion_intensity,
                        1.0,
                    )
                    / 10.0
                ),
            )

            return self._result(
                activity="WALKING",
                confidence=confidence,
            )

        # ====================================================
        # 5. SITTING
        # ====================================================

        if (
            motion_intensity
            < ActivityConfig.SITTING_MOTION
            and heart_rate
            < ActivityConfig.SITTING_HEART_RATE_MAX
            and orientation_abs
            < ActivityConfig.SITTING_ORIENTATION_MAX
            and acceleration
            < ActivityConfig.STANDING_ACCELERATION_MIN
        ):

            return self._result(
                activity="SITTING",
                confidence=0.88,
            )

        # ====================================================
        # 6. NORMAL
        # ====================================================

        # NORMAL is a general healthy stationary state.
        #
        # It is intentionally based on sensor values rather than
        # the simulator's NORMAL scenario name.

        normal_physiology = (
            ActivityConfig.NORMAL_HEART_RATE_MIN
            <= heart_rate
            <= ActivityConfig.NORMAL_HEART_RATE_MAX
            and spo2
            >= ActivityConfig.NORMAL_SPO2_MIN
            and ActivityConfig.NORMAL_BODY_TEMPERATURE_MIN
            <= body_temperature
            <= ActivityConfig.NORMAL_BODY_TEMPERATURE_MAX
        )

        normal_motion = (
            motion_intensity
            <= ActivityConfig.NORMAL_MOTION_MAX
        )

        normal_acceleration = (
            ActivityConfig.NORMAL_ACCELERATION_MIN
            <= acceleration
            <= ActivityConfig.NORMAL_ACCELERATION_MAX
        )

        if (
            normal_physiology
            and normal_motion
            and normal_acceleration
        ):

            return self._result(
                activity="NORMAL",
                confidence=0.92,
            )

        # ====================================================
        # 7. STANDING
        # ====================================================

        if (
            motion_intensity
            < ActivityConfig.STANDING_MOTION
            and ActivityConfig.STANDING_ACCELERATION_MIN
            <= acceleration
            <= ActivityConfig.STANDING_ACCELERATION_MAX
        ):

            return self._result(
                activity="STANDING",
                confidence=0.88,
            )

        # ====================================================
        # 8. UNKNOWN
        # ====================================================

        return self._result(
            activity="UNKNOWN",
            confidence=0.50,
        )

    # ========================================================
    # PUBLIC RECOGNITION API
    # ========================================================

    def recognize(
        self,
        sensor_data: Dict[str, Any],
    ) -> ActivityResult:
        """
        Classify the current sensor state.

        Parameters
        ----------
        sensor_data:
            Dictionary containing current sensor readings.

        Returns
        -------
        ActivityResult:
            Current activity, confidence, description and
            emergency status.
        """

        if not isinstance(
            sensor_data,
            dict,
        ):

            result = self._result(
                activity="UNKNOWN",
                confidence=0.0,
            )

            self._store_result(
                result
            )

            return result

        result = self._classify(
            sensor_data
        )

        self._store_result(
            result
        )

        return result

    # ========================================================
    # STATE MANAGEMENT
    # ========================================================

    def _store_result(
        self,
        result: ActivityResult,
    ) -> None:
        """Store the latest recognition result."""

        self.last_result = result

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return the current recognition status."""

        return self.last_result.as_dict()

    def reset(
        self,
    ) -> None:
        """Reset the recognizer to an unknown state."""

        self.last_result = self._result(
            activity="UNKNOWN",
            confidence=0.0,
        )


# ============================================================
# GLOBAL RECOGNIZER
# ============================================================

_activity_recognizer = ActivityRecognizer()


# ============================================================
# PUBLIC CONVENIENCE API
# ============================================================

def recognize_activity(
    sensor_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recognize activity using the shared SAFEBAND recognizer.

    The classification is based exclusively on sensor values.

    Example
    -------

    result = recognize_activity({
        "acceleration_x": 0.2,
        "acceleration_y": 0.1,
        "acceleration_z": 1.05,
        "heart_rate": 82,
        "spo2": 98,
        "body_temperature": 36.7,
        "orientation": 10,
        "motion_intensity": 0.4,
    })
    """

    result = _activity_recognizer.recognize(
        sensor_data
    )

    return result.as_dict()


def get_activity_status() -> Dict[str, Any]:
    """Return the latest activity-recognition status."""

    return _activity_recognizer.get_status()


def reset_activity_recognition() -> None:
    """Reset the shared activity recognizer."""

    _activity_recognizer.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "ActivityConfig",
    "ActivityResult",
    "ActivityRecognizer",
    "recognize_activity",
    "get_activity_status",
    "reset_activity_recognition",
]