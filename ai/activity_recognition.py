"""
SAFEBAND AI - Activity Recognition Engine

Prototype activity-recognition module.

This version uses rule-based classification on simulated sensor
values. It is intentionally structured so that a trained TinyML/AI
model can replace the classification logic later without changing
the rest of the application.
"""

from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass
class ActivityResult:
    """Result returned by the activity-recognition engine."""

    activity: str
    confidence: float
    description: str
    emergency: bool = False


class ActivityRecognizer:
    """
    Prototype activity-recognition engine.

    Expected sensor inputs:
        acceleration_x
        acceleration_y
        acceleration_z
        heart_rate
        orientation
        motion_intensity

    The current implementation uses deterministic rules rather than
    a trained machine-learning model.
    """

    ACTIVITIES = {
        "SITTING": "User appears stationary.",
        "STANDING": "User is upright with low movement.",
        "WALKING": "User is walking normally.",
        "RUNNING": "User is performing high-intensity movement.",
        "FALL": "Sudden abnormal motion consistent with a fall.",
        "UNKNOWN": "Activity could not be classified."
    }

    def __init__(self):
        self.last_activity = "UNKNOWN"
        self.last_confidence = 0.0

    @staticmethod
    def _get(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely retrieve a numeric sensor value."""
        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _calculate_acceleration_magnitude(data: Dict[str, Any]) -> float:
        """Calculate resultant acceleration from X, Y and Z axes."""
        ax = ActivityRecognizer._get(data, "acceleration_x")
        ay = ActivityRecognizer._get(data, "acceleration_y")
        az = ActivityRecognizer._get(data, "acceleration_z")

        return math.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    def recognize(self, sensor_data: Dict[str, Any]) -> ActivityResult:
        """
        Classify the user's current activity.

        Parameters
        ----------
        sensor_data : dict
            Sensor values collected from the simulated or real
            SAFEBAND sensor layer.

        Returns
        -------
        ActivityResult
            Classified activity, confidence and emergency status.
        """

        acceleration = self._calculate_acceleration_magnitude(sensor_data)

        motion_intensity = self._get(
            sensor_data,
            "motion_intensity",
            abs(acceleration - 1.0)
        )

        heart_rate = self._get(
            sensor_data,
            "heart_rate",
            75.0
        )

        orientation = self._get(
            sensor_data,
            "orientation",
            0.0
        )

        # ---------------------------------------------------------
        # 1. FALL DETECTION
        # ---------------------------------------------------------
        #
        # A large acceleration spike combined with an abnormal
        # orientation is treated as a possible fall.
        #
        if acceleration >= 2.5 and (
            orientation >= 60 or orientation <= -60
        ):
            result = ActivityResult(
                activity="FALL",
                confidence=0.96,
                description=self.ACTIVITIES["FALL"],
                emergency=True
            )

            self._store_result(result)
            return result

        # ---------------------------------------------------------
        # 2. RUNNING
        # ---------------------------------------------------------

        if motion_intensity >= 1.2 or acceleration >= 1.8:
            confidence = min(
                0.95,
                0.75 + (motion_intensity / 10.0)
            )

            result = ActivityResult(
                activity="RUNNING",
                confidence=confidence,
                description=self.ACTIVITIES["RUNNING"]
            )

            self._store_result(result)
            return result

        # ---------------------------------------------------------
        # 3. WALKING
        # ---------------------------------------------------------

        if motion_intensity >= 0.25 or 1.05 <= acceleration <= 1.5:
            confidence = min(
                0.95,
                0.80 + (motion_intensity / 10.0)
            )

            result = ActivityResult(
                activity="WALKING",
                confidence=confidence,
                description=self.ACTIVITIES["WALKING"]
            )

            self._store_result(result)
            return result

        # ---------------------------------------------------------
        # 4. STANDING
        # ---------------------------------------------------------

        if motion_intensity < 0.25 and 0.85 <= acceleration <= 1.15:
            result = ActivityResult(
                activity="STANDING",
                confidence=0.91,
                description=self.ACTIVITIES["STANDING"]
            )

            self._store_result(result)
            return result

        # ---------------------------------------------------------
        # 5. SITTING
        # ---------------------------------------------------------

        if (
            motion_intensity < 0.15
            and heart_rate < 100
            and abs(orientation) < 45
        ):
            result = ActivityResult(
                activity="SITTING",
                confidence=0.88,
                description=self.ACTIVITIES["SITTING"]
            )

            self._store_result(result)
            return result

        # ---------------------------------------------------------
        # 6. UNKNOWN
        # ---------------------------------------------------------

        result = ActivityResult(
            activity="UNKNOWN",
            confidence=0.50,
            description=self.ACTIVITIES["UNKNOWN"]
        )

        self._store_result(result)
        return result

    def _store_result(self, result: ActivityResult) -> None:
        """Store the latest recognition result."""
        self.last_activity = result.activity
        self.last_confidence = result.confidence

    def get_status(self) -> Dict[str, Any]:
        """Return the current recognition status."""
        return {
            "activity": self.last_activity,
            "confidence": self.last_confidence
        }


def recognize_activity(sensor_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for the rest of the SAFEBAND application.

    Example:
        result = recognize_activity({
            "acceleration_x": 0.2,
            "acceleration_y": 0.1,
            "acceleration_z": 1.05,
            "heart_rate": 82,
            "orientation": 10,
            "motion_intensity": 0.4
        })
    """

    recognizer = ActivityRecognizer()
    result = recognizer.recognize(sensor_data)

    return {
        "activity": result.activity,
        "confidence": round(result.confidence, 2),
        "description": result.description,
        "emergency": result.emergency
    }