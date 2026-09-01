"""
SAFEBAND AI - Sensor Fusion Engine

Combines readings from SAFEBAND's physiological, motion,
environmental, body-temperature, and acoustic sensors into
a unified condition.

Current implementation:
    Weighted rule-based sensor fusion.

Future implementation:
    A trained AI/TinyML fusion model can replace or augment
    the rule-based fusion logic without changing the public API.

Important architecture rule:

    Sensor Fusion
        ↓
    Combined evidence / condition
        ↓
    Risk Engine
        ↓
    Final safety / emergency decision


Sensor groups
-------------

BNO055
    Motion and orientation.

MAX30102
    Heart rate and SpO2.

MAX30208
    Body temperature.

BME680
    Environmental temperature, humidity and pressure.

INMP441
    Acoustic activity.

GPS
    Location is retained in the sensor pipeline but does not
    directly contribute to safety fusion.

Important sensor distinction:

    temperature
        BME680 environmental temperature

    body_temperature
        MAX30208 body temperature
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# FUSION CONFIGURATION
# ============================================================

class FusionConfig:
    """Thresholds and weights used by the fusion engine."""

    # --------------------------------------------------------
    # MOTION
    # --------------------------------------------------------

    MOTION_HIGH = 2.0
    MOTION_ELEVATED = 1.2

    MOTION_HIGH_SCORE = 35.0
    MOTION_ELEVATED_SCORE = 15.0

    # --------------------------------------------------------
    # ORIENTATION
    # --------------------------------------------------------

    ORIENTATION_HIGH = 75.0
    ORIENTATION_UNUSUAL = 60.0

    ORIENTATION_HIGH_SCORE = 30.0
    ORIENTATION_UNUSUAL_SCORE = 15.0

    # Additional evidence when activity recognition detects
    # a FALL.
    FALL_BONUS = 15.0

    # --------------------------------------------------------
    # PHYSIOLOGICAL
    # --------------------------------------------------------

    HEART_RATE_HIGH = 130.0
    HEART_RATE_ELEVATED = 110.0
    HEART_RATE_LOW = 50.0

    HEART_RATE_HIGH_SCORE = 25.0
    HEART_RATE_ELEVATED_SCORE = 12.0
    HEART_RATE_LOW_SCORE = 20.0

    SPO2_CRITICAL = 90.0
    SPO2_REDUCED = 94.0

    SPO2_CRITICAL_SCORE = 30.0
    SPO2_REDUCED_SCORE = 15.0

    # --------------------------------------------------------
    # ENVIRONMENTAL
    # --------------------------------------------------------

    # BME680 environmental temperature.
    AMBIENT_TEMPERATURE_HIGH = 45.0
    AMBIENT_TEMPERATURE_LOW = 5.0

    AMBIENT_TEMPERATURE_HIGH_SCORE = 15.0
    AMBIENT_TEMPERATURE_LOW_SCORE = 10.0

    HUMIDITY_HIGH = 90.0
    HUMIDITY_LOW = 20.0
    HUMIDITY_SCORE = 5.0

    PRESSURE_LOW = 950.0
    PRESSURE_HIGH = 1050.0
    PRESSURE_SCORE = 5.0

    # --------------------------------------------------------
    # BODY TEMPERATURE
    # --------------------------------------------------------

    # MAX30208 body-temperature thresholds.
    #
    # These are prototype fusion thresholds. They should be
    # validated during the hardware/physiological validation
    # phase before being treated as clinical rules.

    BODY_TEMPERATURE_HIGH = 38.0
    BODY_TEMPERATURE_CRITICAL = 39.5

    BODY_TEMPERATURE_LOW = 35.0
    BODY_TEMPERATURE_CRITICAL_LOW = 34.0

    BODY_TEMPERATURE_HIGH_SCORE = 8.0
    BODY_TEMPERATURE_CRITICAL_SCORE = 12.0

    BODY_TEMPERATURE_LOW_SCORE = 8.0
    BODY_TEMPERATURE_CRITICAL_LOW_SCORE = 12.0

    BODY_TEMPERATURE_ENABLED = True

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    AUDIO_HIGH = 0.85
    AUDIO_ELEVATED = 0.65

    AUDIO_HIGH_SCORE = 10.0
    AUDIO_ELEVATED_SCORE = 5.0

    # --------------------------------------------------------
    # FUSION BONUSES
    # --------------------------------------------------------

    TWO_GROUP_BONUS = 10.0
    THREE_GROUP_BONUS = 10.0

    # Emergency activity reinforcement.
    EMERGENCY_BONUS = 20.0

    # Manual SOS is an explicit emergency input.
    SOS_SCORE = 100.0

    # --------------------------------------------------------
    # CONDITION THRESHOLDS
    # --------------------------------------------------------

    HIGH_RISK_THRESHOLD = 60.0
    ABNORMAL_THRESHOLD = 30.0


# ============================================================
# FUSION RESULT
# ============================================================

@dataclass
class FusionResult:
    """Result produced by the sensor-fusion engine."""

    fusion_score: float
    confidence: float
    condition: str
    abnormal: bool

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return the result as a dictionary."""

        return {
            "fusion_score": self.fusion_score,
            "confidence": self.confidence,
            "condition": self.condition,
            "abnormal": self.abnormal,
            "evidence": self.evidence,
        }


# ============================================================
# SENSOR FUSION ENGINE
# ============================================================

class SensorFusion:
    """
    Multi-sensor fusion engine for SAFEBAND AI.

    The engine combines evidence from independent sensor
    groups rather than making the final emergency decision.

    Sensor groups:

        motion
        physiological
        body_temperature
        environmental
        audio

    The final emergency/risk decision remains the responsibility
    of the risk engine.
    """

    def __init__(self) -> None:
        """Initialize the fusion engine."""

        self.last_result = FusionResult(
            fusion_score=0.0,
            confidence=0.0,
            condition="NORMAL",
            abnormal=False,
            evidence={},
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
    # CONFIDENCE NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_confidence(
        confidence: float,
    ) -> float:
        """
        Normalize activity-recognition confidence.

        Accepts either:

            0.91
            91

        and converts both to:

            0.91
        """

        try:
            confidence = float(
                confidence
            )

        except (TypeError, ValueError):

            return 0.0

        if confidence > 1.0:
            confidence /= 100.0

        return min(
            1.0,
            max(
                0.0,
                confidence,
            ),
        )

    # ========================================================
    # MOTION FUSION
    # ========================================================

    @staticmethod
    def _motion_evidence(
        motion_intensity: float,
        orientation: float,
    ) -> float:
        """
        Calculate BNO055 motion/orientation evidence.

        Maximum normal motion-group contribution:

            40 points
        """

        score = 0.0

        if (
            motion_intensity
            >= FusionConfig.MOTION_HIGH
        ):

            score += (
                FusionConfig.MOTION_HIGH_SCORE
            )

        elif (
            motion_intensity
            >= FusionConfig.MOTION_ELEVATED
        ):

            score += (
                FusionConfig.MOTION_ELEVATED_SCORE
            )

        orientation_abs = abs(
            orientation
        )

        if (
            orientation_abs
            >= FusionConfig.ORIENTATION_HIGH
        ):

            score += (
                FusionConfig.ORIENTATION_HIGH_SCORE
            )

        elif (
            orientation_abs
            >= FusionConfig.ORIENTATION_UNUSUAL
        ):

            score += (
                FusionConfig.ORIENTATION_UNUSUAL_SCORE
            )

        return min(
            40.0,
            score,
        )

    # ========================================================
    # PHYSIOLOGICAL FUSION
    # ========================================================

    @staticmethod
    def _physiological_evidence(
        heart_rate: float,
        spo2: float,
    ) -> float:
        """Calculate MAX30102 physiological evidence."""

        score = 0.0

        if (
            heart_rate
            >= FusionConfig.HEART_RATE_HIGH
        ):

            score += (
                FusionConfig.HEART_RATE_HIGH_SCORE
            )

        elif (
            heart_rate
            >= FusionConfig.HEART_RATE_ELEVATED
        ):

            score += (
                FusionConfig.HEART_RATE_ELEVATED_SCORE
            )

        elif (
            heart_rate
            < FusionConfig.HEART_RATE_LOW
        ):

            score += (
                FusionConfig.HEART_RATE_LOW_SCORE
            )

        if (
            spo2
            < FusionConfig.SPO2_CRITICAL
        ):

            score += (
                FusionConfig.SPO2_CRITICAL_SCORE
            )

        elif (
            spo2
            < FusionConfig.SPO2_REDUCED
        ):

            score += (
                FusionConfig.SPO2_REDUCED_SCORE
            )

        return min(
            35.0,
            score,
        )

    # ========================================================
    # ENVIRONMENTAL FUSION
    # ========================================================

    @staticmethod
    def _environmental_evidence(
        ambient_temperature: float,
        humidity: float,
        pressure: float,
    ) -> float:
        """
        Calculate BME680 environmental evidence.

        `ambient_temperature` is BME680 environmental
        temperature, not body temperature.
        """

        score = 0.0

        if (
            ambient_temperature
            >= FusionConfig.AMBIENT_TEMPERATURE_HIGH
        ):

            score += (
                FusionConfig
                .AMBIENT_TEMPERATURE_HIGH_SCORE
            )

        elif (
            ambient_temperature
            <= FusionConfig.AMBIENT_TEMPERATURE_LOW
        ):

            score += (
                FusionConfig
                .AMBIENT_TEMPERATURE_LOW_SCORE
            )

        if (
            humidity
            >= FusionConfig.HUMIDITY_HIGH
        ):

            score += (
                FusionConfig.HUMIDITY_SCORE
            )

        elif (
            humidity
            <= FusionConfig.HUMIDITY_LOW
        ):

            score += (
                FusionConfig.HUMIDITY_SCORE
            )

        if (
            pressure
            < FusionConfig.PRESSURE_LOW
            or pressure
            > FusionConfig.PRESSURE_HIGH
        ):

            score += (
                FusionConfig.PRESSURE_SCORE
            )

        return min(
            20.0,
            score,
        )

    # ========================================================
    # BODY TEMPERATURE
    # ========================================================

    @staticmethod
    def _body_temperature_evidence(
        body_temperature: float,
    ) -> float:
        """
        Calculate MAX30208 body-temperature evidence.

        This signal is deliberately independent of BME680
        environmental temperature.
        """

        if not FusionConfig.BODY_TEMPERATURE_ENABLED:

            return 0.0

        if (
            body_temperature
            >= FusionConfig.BODY_TEMPERATURE_CRITICAL
        ):

            return (
                FusionConfig
                .BODY_TEMPERATURE_CRITICAL_SCORE
            )

        if (
            body_temperature
            >= FusionConfig.BODY_TEMPERATURE_HIGH
        ):

            return (
                FusionConfig
                .BODY_TEMPERATURE_HIGH_SCORE
            )

        if (
            body_temperature
            <= FusionConfig.BODY_TEMPERATURE_CRITICAL_LOW
        ):

            return (
                FusionConfig
                .BODY_TEMPERATURE_CRITICAL_LOW_SCORE
            )

        if (
            body_temperature
            <= FusionConfig.BODY_TEMPERATURE_LOW
        ):

            return (
                FusionConfig
                .BODY_TEMPERATURE_LOW_SCORE
            )

        return 0.0

    # ========================================================
    # AUDIO FUSION
    # ========================================================

    @staticmethod
    def _audio_evidence(
        audio_level: float,
    ) -> float:
        """Calculate INMP441 acoustic evidence."""

        if (
            audio_level
            >= FusionConfig.AUDIO_HIGH
        ):

            return (
                FusionConfig.AUDIO_HIGH_SCORE
            )

        if (
            audio_level
            >= FusionConfig.AUDIO_ELEVATED
        ):

            return (
                FusionConfig.AUDIO_ELEVATED_SCORE
            )

        return 0.0

    # ========================================================
    # CONDITION CLASSIFICATION
    # ========================================================

    @staticmethod
    def _classify_condition(
        fusion_score: float,
        activity: str,
        activity_emergency: bool,
        manual_sos: bool,
    ) -> str:
        """
        Convert fused evidence into a combined condition.

        Explicit emergency inputs take priority over numerical
        fusion thresholds.
        """

        if (
            manual_sos
            or activity == "SOS"
        ):

            return "EMERGENCY"

        if (
            activity == "FALL"
            or activity_emergency
        ):

            return "EMERGENCY"

        if (
            fusion_score
            >= FusionConfig.HIGH_RISK_THRESHOLD
        ):

            return "HIGH RISK"

        if (
            fusion_score
            >= FusionConfig.ABNORMAL_THRESHOLD
        ):

            return "ABNORMAL"

        return "NORMAL"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    @staticmethod
    def _calculate_confidence(
        abnormal_groups: int,
        activity_confidence: float,
    ) -> float:
        """
        Calculate confidence in the fused assessment.

        More independently abnormal sensor groups produce
        stronger fusion confidence.
        """

        if abnormal_groups == 0:

            confidence = 0.90

        elif abnormal_groups == 1:

            confidence = 0.72

        elif abnormal_groups == 2:

            confidence = 0.84

        else:

            confidence = 0.94

        activity_confidence = (
            SensorFusion._normalize_confidence(
                activity_confidence
            )
        )

        if activity_confidence > 0.0:

            confidence = (
                confidence * 0.70
                + activity_confidence * 0.30
            )

        return min(
            0.99,
            max(
                0.0,
                confidence,
            ),
        )

    # ========================================================
    # MAIN PROCESSING
    # ========================================================

    def process(
        self,
        sensor_data: Dict[str, Any],
        activity_result: Optional[
            Dict[str, Any]
        ] = None,
    ) -> FusionResult:
        """
        Fuse current SAFEBAND sensor readings.

        The method uses actual sensor values for evidence
        calculation. The activity result provides context and
        confidence, but the sensor values remain the basis of
        the fusion score.
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

        # ====================================================
        # SENSOR VALUES
        # ====================================================

        heart_rate = self._number(
            sensor_data,
            "heart_rate",
            75.0,
        )

        spo2 = self._number(
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

        humidity = self._number(
            sensor_data,
            "humidity",
            50.0,
        )

        pressure = self._number(
            sensor_data,
            "pressure",
            1013.0,
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

        audio_level = self._number(
            sensor_data,
            "audio_level",
            0.0,
        )

        # ====================================================
        # ACTIVITY INFORMATION
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

        activity_confidence = self._normalize_confidence(
            self._number(
                activity_result,
                "confidence",
                0.0,
            )
        )

        # Explicit emergency input.
        manual_sos = self._boolean(
            sensor_data,
            "manual_sos",
            False,
        )

        # ====================================================
        # INDIVIDUAL SENSOR GROUPS
        # ====================================================

        motion_evidence = (
            self._motion_evidence(
                motion_intensity,
                orientation,
            )
        )

        physiological_evidence = (
            self._physiological_evidence(
                heart_rate,
                spo2,
            )
        )

        environmental_evidence = (
            self._environmental_evidence(
                ambient_temperature,
                humidity,
                pressure,
            )
        )

        body_temperature_evidence = (
            self._body_temperature_evidence(
                body_temperature,
            )
        )

        audio_evidence = (
            self._audio_evidence(
                audio_level,
            )
        )

        # ====================================================
        # FALL REINFORCEMENT
        # ====================================================

        if activity == "FALL":

            motion_evidence = min(
                40.0,
                motion_evidence
                + FusionConfig.FALL_BONUS,
            )

        # ====================================================
        # EMERGENCY OVERRIDE
        # ====================================================

        if (
            manual_sos
            or activity == "SOS"
        ):

            fusion_score = (
                FusionConfig.SOS_SCORE
            )

            abnormal_groups = 5

            confidence = 0.99

        else:

            # =================================================
            # ABNORMAL SENSOR GROUPS
            # =================================================

            abnormal_groups = 0

            if motion_evidence >= 15.0:

                abnormal_groups += 1

            if physiological_evidence >= 12.0:

                abnormal_groups += 1

            if environmental_evidence >= 10.0:

                abnormal_groups += 1

            if body_temperature_evidence >= 5.0:

                abnormal_groups += 1

            if audio_evidence >= 5.0:

                abnormal_groups += 1

            # ================================================
            # BASE FUSION SCORE
            # ================================================

            base_score = (
                motion_evidence
                + physiological_evidence
                + environmental_evidence
                + body_temperature_evidence
                + audio_evidence
            )

            # ================================================
            # CROSS-SENSOR REINFORCEMENT
            # ================================================

            fusion_bonus = 0.0

            if abnormal_groups >= 2:

                fusion_bonus += (
                    FusionConfig.TWO_GROUP_BONUS
                )

            if abnormal_groups >= 3:

                fusion_bonus += (
                    FusionConfig.THREE_GROUP_BONUS
                )

            if activity_emergency:

                fusion_bonus += (
                    FusionConfig.EMERGENCY_BONUS
                )

            fusion_score = min(
                100.0,
                base_score + fusion_bonus,
            )

            confidence = (
                self._calculate_confidence(
                    abnormal_groups,
                    activity_confidence,
                )
            )

        # ====================================================
        # CONDITION
        # ====================================================

        condition = self._classify_condition(
            fusion_score,
            activity,
            activity_emergency,
            manual_sos,
        )

        abnormal = (
            condition != "NORMAL"
        )

        # ====================================================
        # EVIDENCE SUMMARY
        # ====================================================

        evidence = {
            "motion_evidence": round(
                motion_evidence,
                1,
            ),

            "physiological_evidence": round(
                physiological_evidence,
                1,
            ),

            "environmental_evidence": round(
                environmental_evidence,
                1,
            ),

            "body_temperature_evidence": round(
                body_temperature_evidence,
                1,
            ),

            "audio_evidence": round(
                audio_evidence,
                1,
            ),

            "abnormal_sensor_groups": (
                abnormal_groups
            ),

            "activity": activity,

            "activity_confidence": round(
                activity_confidence,
                2,
            ),

            "manual_sos": manual_sos,

            "sensor_inputs": {
                "heart_rate": heart_rate,
                "spo2": spo2,
                "temperature": ambient_temperature,
                "body_temperature": body_temperature,
                "humidity": humidity,
                "pressure": pressure,
                "motion_intensity": motion_intensity,
                "orientation": orientation,
                "audio_level": audio_level,
            },
        }

        # ====================================================
        # FINAL RESULT
        # ====================================================

        result = FusionResult(
            fusion_score=round(
                fusion_score,
                1,
            ),

            confidence=round(
                confidence,
                2,
            ),

            condition=condition,

            abnormal=abnormal,

            evidence=evidence,
        )

        self.last_result = result

        return result

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """Return the latest sensor-fusion status."""

        return self.last_result.as_dict()

    def reset(
        self,
    ) -> None:
        """Reset the global sensor-fusion engine."""

        self.last_result = FusionResult(
            fusion_score=0.0,
            confidence=0.0,
            condition="NORMAL",
            abnormal=False,
            evidence={},
        )


# ============================================================
# GLOBAL FUSION ENGINE
# ============================================================

_fusion_engine = SensorFusion()


# ============================================================
# PUBLIC CONVENIENCE API
# ============================================================

def fuse_sensor_data(
    sensor_data: Dict[str, Any],
    activity_result: Optional[
        Dict[str, Any]
    ] = None,
) -> Dict[str, Any]:
    """
    Fuse SAFEBAND sensor data using the shared engine.
    """

    result = _fusion_engine.process(
        sensor_data,
        activity_result,
    )

    return result.as_dict()


def get_fusion_status() -> Dict[str, Any]:
    """Return the latest global sensor-fusion status."""

    return _fusion_engine.get_status()


def reset_sensor_fusion() -> None:
    """Reset the global sensor-fusion engine."""

    _fusion_engine.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "FusionConfig",
    "FusionResult",
    "SensorFusion",
    "fuse_sensor_data",
    "get_fusion_status",
    "reset_sensor_fusion",
]