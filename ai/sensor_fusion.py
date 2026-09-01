"""
SAFEBAND AI - Sensor Fusion Engine

Prototype sensor-fusion module.

Combines information from:
- BNO055  : motion and orientation
- MAX30102: heart rate and SpO2
- BME680  : environmental conditions
- INMP441 : acoustic activity

The current implementation uses weighted evidence and rule-based
fusion for demonstration. It is designed so that a trained
sensor-fusion / TinyML model can replace the logic later.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class FusionResult:
    """Result produced by the sensor-fusion engine."""

    fusion_score: float
    confidence: float
    condition: str
    abnormal: bool
    evidence: Dict[str, Any]


class SensorFusion:
    """
    Multi-sensor fusion engine for SAFEBAND AI.

    The engine does not make the final emergency decision.
    It combines sensor evidence and passes the resulting information
    to the risk engine.
    """

    def __init__(self):
        self.last_result = FusionResult(
            fusion_score=0.0,
            confidence=0.0,
            condition="NORMAL",
            abnormal=False,
            evidence={}
        )

    @staticmethod
    def _number(
        data: Dict[str, Any],
        key: str,
        default: float = 0.0
    ) -> float:
        """Safely retrieve a numeric sensor value."""

        try:
            return float(data.get(key, default))
        except (TypeError, ValueError):
            return default

    def process(
        self,
        sensor_data: Dict[str, Any],
        activity_result: Dict[str, Any] | None = None
    ) -> FusionResult:
        """
        Fuse multiple sensor readings into a combined condition.

        Parameters
        ----------
        sensor_data : dict
            Current sensor readings.

        activity_result : dict, optional
            Activity-recognition result.

        Returns
        -------
        FusionResult
            Combined sensor-fusion assessment.
        """

        # ---------------------------------------------------------
        # SENSOR INPUTS
        # ---------------------------------------------------------

        heart_rate = self._number(
            sensor_data,
            "heart_rate",
            75.0
        )

        spo2 = self._number(
            sensor_data,
            "spo2",
            98.0
        )

        temperature = self._number(
            sensor_data,
            "temperature",
            25.0
        )

        humidity = self._number(
            sensor_data,
            "humidity",
            50.0
        )

        pressure = self._number(
            sensor_data,
            "pressure",
            1013.0
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

        audio_level = self._number(
            sensor_data,
            "audio_level",
            0.0
        )

        # ---------------------------------------------------------
        # INDIVIDUAL SENSOR EVIDENCE
        # ---------------------------------------------------------

        motion_evidence = 0.0
        physiological_evidence = 0.0
        environmental_evidence = 0.0
        audio_evidence = 0.0

        # ---------------------------------------------------------
        # MOTION / ORIENTATION
        # BNO055
        # ---------------------------------------------------------

        if motion_intensity >= 2.0:
            motion_evidence += 35

        elif motion_intensity >= 1.2:
            motion_evidence += 15

        if abs(orientation) >= 75:
            motion_evidence += 30

        elif abs(orientation) >= 60:
            motion_evidence += 15

        motion_evidence = min(40.0, motion_evidence)

        # ---------------------------------------------------------
        # PHYSIOLOGICAL DATA
        # MAX30102
        # ---------------------------------------------------------

        if heart_rate >= 130:
            physiological_evidence += 25

        elif heart_rate >= 110:
            physiological_evidence += 12

        elif heart_rate < 50:
            physiological_evidence += 20

        if spo2 < 90:
            physiological_evidence += 30

        elif spo2 < 94:
            physiological_evidence += 15

        physiological_evidence = min(
            35.0,
            physiological_evidence
        )

        # ---------------------------------------------------------
        # ENVIRONMENTAL DATA
        # BME680
        # ---------------------------------------------------------

        if temperature >= 45:
            environmental_evidence += 15

        elif temperature <= 5:
            environmental_evidence += 10

        if humidity >= 90:
            environmental_evidence += 5

        elif humidity <= 20:
            environmental_evidence += 5

        # Pressure is retained as part of the fused environmental
        # information. Extreme values can be incorporated later
        # when calibrated field data is available.
        if pressure < 950 or pressure > 1050:
            environmental_evidence += 5

        environmental_evidence = min(
            20.0,
            environmental_evidence
        )

        # ---------------------------------------------------------
        # AUDIO DATA
        # INMP441
        # ---------------------------------------------------------

        if audio_level >= 0.85:
            audio_evidence += 10

        elif audio_level >= 0.65:
            audio_evidence += 5

        audio_evidence = min(10.0, audio_evidence)

        # ---------------------------------------------------------
        # ACTIVITY INFORMATION
        # ---------------------------------------------------------

        activity = "UNKNOWN"
        activity_emergency = False
        activity_confidence = 0.0

        if activity_result:
            activity = str(
                activity_result.get(
                    "activity",
                    "UNKNOWN"
                )
            ).upper()

            activity_emergency = bool(
                activity_result.get(
                    "emergency",
                    False
                )
            )

            activity_confidence = self._number(
                activity_result,
                "confidence",
                0.0
            )

        # A detected fall is strong combined evidence.
        if activity == "FALL":
            motion_evidence += 15

        # ---------------------------------------------------------
        # SENSOR FUSION
        # ---------------------------------------------------------

        base_score = (
            motion_evidence
            + physiological_evidence
            + environmental_evidence
            + audio_evidence
        )

        # Multiple abnormal sensor groups increase confidence.
        abnormal_groups = 0

        if motion_evidence >= 15:
            abnormal_groups += 1

        if physiological_evidence >= 12:
            abnormal_groups += 1

        if environmental_evidence >= 10:
            abnormal_groups += 1

        if audio_evidence >= 5:
            abnormal_groups += 1

        # Cross-sensor reinforcement.
        fusion_bonus = 0.0

        if abnormal_groups >= 2:
            fusion_bonus += 10

        if abnormal_groups >= 3:
            fusion_bonus += 10

        if activity_emergency:
            fusion_bonus += 20

        fusion_score = min(
            100.0,
            base_score + fusion_bonus
        )

        # ---------------------------------------------------------
        # CONFIDENCE
        # ---------------------------------------------------------

        confidence = 0.50

        if abnormal_groups == 0:
            confidence = 0.90

        elif abnormal_groups == 1:
            confidence = 0.72

        elif abnormal_groups == 2:
            confidence = 0.84

        elif abnormal_groups >= 3:
            confidence = 0.94

        if activity_confidence > 0:
            confidence = (
                confidence * 0.7
                + activity_confidence * 0.3
            )

        confidence = min(
            0.99,
            max(0.0, confidence)
        )

        # ---------------------------------------------------------
        # CONDITION CLASSIFICATION
        # ---------------------------------------------------------

        if activity == "FALL" or activity_emergency:
            condition = "EMERGENCY"

        elif fusion_score >= 60:
            condition = "HIGH RISK"

        elif fusion_score >= 30:
            condition = "ABNORMAL"

        else:
            condition = "NORMAL"

        abnormal = condition != "NORMAL"

        # ---------------------------------------------------------
        # EVIDENCE SUMMARY
        # ---------------------------------------------------------

        evidence = {
            "motion_evidence": round(
                motion_evidence,
                1
            ),
            "physiological_evidence": round(
                physiological_evidence,
                1
            ),
            "environmental_evidence": round(
                environmental_evidence,
                1
            ),
            "audio_evidence": round(
                audio_evidence,
                1
            ),
            "abnormal_sensor_groups": abnormal_groups,
            "activity": activity
        }

        # ---------------------------------------------------------
        # FINAL RESULT
        # ---------------------------------------------------------

        result = FusionResult(
            fusion_score=round(
                fusion_score,
                1
            ),
            confidence=round(
                confidence,
                2
            ),
            condition=condition,
            abnormal=abnormal,
            evidence=evidence
        )

        self.last_result = result

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return the latest sensor-fusion status."""

        return {
            "fusion_score": self.last_result.fusion_score,
            "confidence": self.last_result.confidence,
            "condition": self.last_result.condition,
            "abnormal": self.last_result.abnormal,
            "evidence": self.last_result.evidence
        }


def fuse_sensor_data(
    sensor_data: Dict[str, Any],
    activity_result: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Convenience function for the SAFEBAND AI application.

    Example
    -------
    result = fuse_sensor_data(
        {
            "heart_rate": 112,
            "spo2": 96,
            "temperature": 28,
            "humidity": 60,
            "pressure": 1013,
            "motion_intensity": 2.5,
            "orientation": 80,
            "audio_level": 0.4
        },
        {
            "activity": "FALL",
            "confidence": 0.96,
            "emergency": True
        }
    )
    """

    fusion_engine = SensorFusion()

    result = fusion_engine.process(
        sensor_data,
        activity_result
    )

    return {
        "fusion_score": result.fusion_score,
        "confidence": result.confidence,
        "condition": result.condition,
        "abnormal": result.abnormal,
        "evidence": result.evidence
    }