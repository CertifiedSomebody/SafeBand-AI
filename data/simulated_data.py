"""
SAFEBAND AI - Simulated Sensor Data Engine

Generates coherent simulated sensor data for the SAFEBAND AI
prototype.

The simulator represents the expected real sensor layer:

    MAX30102
        Heart rate
        SpO2

    MAX30208
        Body temperature

    BNO055
        Acceleration
        Motion intensity
        Orientation

    BME680
        Environmental temperature
        Humidity
        Pressure

    INMP441
        Audio level

    GPS
        Latitude
        Longitude

Simulation profiles are development/testing tools only.

IMPORTANT
---------
The selected simulation profile describes the sensor conditions
to generate. It does NOT directly determine the activity shown
by the AI engine.

The activity-recognition layer must infer activity from the
generated sensor data.

PROFILE SWITCHING
-----------------
Each profile switch creates a clean sensor-state boundary.

This prevents transient information from a previous profile from
leaking into the next profile.

Examples:

    NORMAL -> FALL
        FALL sensor values + manual_sos=False

    FALL -> NORMAL
        NORMAL sensor values + manual_sos=False

    NORMAL -> SOS
        SOS sensor values + manual_sos=True

    SOS -> NORMAL
        NORMAL sensor values + manual_sos=False

The simulator itself does not retain an emergency condition after
the selected profile changes.
"""

from datetime import datetime
import random
from typing import Any, Dict, Optional


from config.settings import (
    DEFAULT_SENSOR_VALUES,
    DEMO_CONFIG,
    get_default_scenario,
)

from data.demo_scenarios import (
    get_scenario,
    get_available_profiles,
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_SCENARIO = get_default_scenario()

MIN_HEART_RATE = 30.0
MAX_HEART_RATE = 220.0

MIN_SPO2 = 70.0
MAX_SPO2 = 100.0

MIN_HUMIDITY = 0.0
MAX_HUMIDITY = 100.0

MIN_AUDIO_LEVEL = 0.0
MAX_AUDIO_LEVEL = 1.0

MIN_BODY_TEMPERATURE = 30.0
MAX_BODY_TEMPERATURE = 45.0

MIN_ENVIRONMENTAL_TEMPERATURE = -40.0
MAX_ENVIRONMENTAL_TEMPERATURE = 85.0

MIN_ORIENTATION = -180.0
MAX_ORIENTATION = 180.0


# ============================================================
# SIMULATED SENSOR ENGINE
# ============================================================

class SimulatedSensorData:
    """
    SAFEBAND AI simulated sensor-data generator.

    Produces one coherent sensor sample at a time.

    Features:
        - Deterministic demonstration profiles
        - Realistic sensor noise
        - Physiological simulation
        - Body-temperature simulation
        - Environmental simulation
        - Motion simulation
        - Audio simulation
        - GPS movement
        - Manual SOS simulation
        - Sample numbering
        - Profile transition reset
        - Full simulator reset

    The class is intentionally independent of the dashboard,
    AI engine and communication layers.
    """

    def __init__(
        self,
        default_scenario: str = DEFAULT_SCENARIO,
        seed: Optional[int] = None,
    ) -> None:

        # ----------------------------------------------------
        # RANDOM GENERATOR
        # ----------------------------------------------------

        if seed is not None:
            self._random = random.Random(seed)
        else:
            self._random = random.Random()

        # ----------------------------------------------------
        # INITIAL PROFILE
        # ----------------------------------------------------

        scenario_name = (
            str(default_scenario)
            .strip()
            .upper()
        )

        # Validate the initial profile.
        profile = get_scenario(
            scenario_name
        )

        self.current_scenario = scenario_name

        self.data = profile

        # ----------------------------------------------------
        # SIMULATOR STATE
        # ----------------------------------------------------

        self.start_time = datetime.now()

        self.sample_count = 0

        self.previous_latitude = float(
            self.data.get(
                "latitude",
                DEFAULT_SENSOR_VALUES["latitude"],
            )
        )

        self.previous_longitude = float(
            self.data.get(
                "longitude",
                DEFAULT_SENSOR_VALUES["longitude"],
            )
        )

    # ========================================================
    # PROFILE CONTROL
    # ========================================================

    def set_scenario(
        self,
        scenario: str,
    ) -> Dict[str, Any]:
        """
        Select a deterministic simulation profile.

        A profile change creates a clean simulation-state
        boundary.

        Important behaviour:

            Previous profile:
                FALL / SOS

            New profile:
                NORMAL

            Result:
                manual_sos=False

        No emergency flag, sensor value, or GPS position from the
        previous profile is carried into the new profile.

        Returns
        -------
        dict
            Copy of the newly selected profile.
        """

        scenario_name = (
            str(scenario)
            .strip()
            .upper()
        )

        # ----------------------------------------------------
        # VALIDATE FIRST
        # ----------------------------------------------------

        new_profile = get_scenario(
            scenario_name
        )

        # ----------------------------------------------------
        # UPDATE ACTIVE PROFILE
        # ----------------------------------------------------

        self.current_scenario = scenario_name

        # Always create an independent profile dictionary.
        self.data = new_profile

        # ----------------------------------------------------
        # RESET PROFILE-SPECIFIC TRANSIENT STATE
        # ----------------------------------------------------

        # Re-anchor GPS to the newly selected profile.
        self.previous_latitude = float(
            self.data.get(
                "latitude",
                DEFAULT_SENSOR_VALUES["latitude"],
            )
        )

        self.previous_longitude = float(
            self.data.get(
                "longitude",
                DEFAULT_SENSOR_VALUES["longitude"],
            )
        )

        # Reset sample numbering at a profile boundary.
        #
        # This makes demonstrations easier to understand and
        # prevents samples from the previous scenario appearing
        # to belong to the new scenario.
        self.sample_count = 0

        # New profile gets a fresh simulation start time.
        self.start_time = datetime.now()

        return self.data.copy()

    def get_scenario(self) -> str:
        """Return the currently active simulation profile."""

        return self.current_scenario

    def get_profile(self) -> str:
        """Alias for get_scenario()."""

        return self.current_scenario

    def get_available_profiles(self) -> list:
        """Return available deterministic simulation profiles."""

        return get_available_profiles()

    # ========================================================
    # RANDOM / NUMERIC HELPERS
    # ========================================================

    def _noise(
        self,
        value: float,
        variation: float,
    ) -> float:
        """Add bounded random variation to a value."""

        return (
            value
            + self._random.uniform(
                -variation,
                variation,
            )
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """Clamp a numeric value to a specified range."""

        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    def _base_value(
        self,
        key: str,
        default: float,
    ) -> float:
        """Safely retrieve a numeric value from the profile."""

        try:
            return float(
                self.data.get(
                    key,
                    default,
                )
            )

        except (TypeError, ValueError):
            return float(default)

    # ========================================================
    # MAX30102 - HEART RATE
    # ========================================================

    def _simulate_heart_rate(self) -> float:
        """Generate simulated MAX30102 heart-rate data."""

        base = self._base_value(
            "heart_rate",
            DEFAULT_SENSOR_VALUES["heart_rate"],
        )

        variation = {
            "NORMAL": 2.0,
            "WALKING": 4.0,
            "RUNNING": 6.0,
            "FALL": 5.0,
            "HIGH_RISK": 5.0,
            "SOS": 4.0,
        }.get(
            self.current_scenario,
            3.0,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            self._clamp(
                value,
                MIN_HEART_RATE,
                MAX_HEART_RATE,
            ),
            1,
        )

    # ========================================================
    # MAX30102 - SPO2
    # ========================================================

    def _simulate_spo2(self) -> float:
        """Generate simulated MAX30102 SpO2 data."""

        base = self._base_value(
            "spo2",
            DEFAULT_SENSOR_VALUES["spo2"],
        )

        variation = {
            "NORMAL": 0.3,
            "WALKING": 0.4,
            "RUNNING": 0.5,
            "FALL": 0.4,
            "HIGH_RISK": 0.4,
            "SOS": 0.3,
        }.get(
            self.current_scenario,
            0.5,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            self._clamp(
                value,
                MIN_SPO2,
                MAX_SPO2,
            ),
            1,
        )

    # ========================================================
    # MAX30208 - BODY TEMPERATURE
    # ========================================================

    def _simulate_body_temperature(self) -> float:
        """
        Generate simulated MAX30208 body temperature.

        IMPORTANT:
            This is physiological/body temperature.

        BME680 environmental temperature is generated separately
        by _simulate_environmental_temperature().
        """

        base = self._base_value(
            "body_temperature",
            DEFAULT_SENSOR_VALUES["body_temperature"],
        )

        variation = {
            "NORMAL": 0.08,
            "WALKING": 0.12,
            "RUNNING": 0.18,
            "FALL": 0.12,
            "HIGH_RISK": 0.15,
            "SOS": 0.10,
        }.get(
            self.current_scenario,
            0.10,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            self._clamp(
                value,
                MIN_BODY_TEMPERATURE,
                MAX_BODY_TEMPERATURE,
            ),
            2,
        )

    # ========================================================
    # BNO055 - ACCELERATION
    # ========================================================

    def _simulate_acceleration(
        self,
    ) -> Dict[str, float]:
        """Generate simulated three-axis BNO055 acceleration."""

        ax = self._noise(
            self._base_value(
                "acceleration_x",
                DEFAULT_SENSOR_VALUES["acceleration_x"],
            ),
            0.06,
        )

        ay = self._noise(
            self._base_value(
                "acceleration_y",
                DEFAULT_SENSOR_VALUES["acceleration_y"],
            ),
            0.06,
        )

        az = self._noise(
            self._base_value(
                "acceleration_z",
                DEFAULT_SENSOR_VALUES["acceleration_z"],
            ),
            0.06,
        )

        return {
            "acceleration_x": round(
                ax,
                3,
            ),
            "acceleration_y": round(
                ay,
                3,
            ),
            "acceleration_z": round(
                az,
                3,
            ),
        }

    # ========================================================
    # BNO055 - MOTION
    # ========================================================

    def _simulate_motion(self) -> float:
        """Generate simulated BNO055 motion intensity."""

        base = self._base_value(
            "motion_intensity",
            DEFAULT_SENSOR_VALUES["motion_intensity"],
        )

        variation = {
            "NORMAL": 0.03,
            "WALKING": 0.08,
            "RUNNING": 0.12,
            "FALL": 0.08,
            "HIGH_RISK": 0.10,
            "SOS": 0.04,
        }.get(
            self.current_scenario,
            0.05,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            max(
                0.0,
                value,
            ),
            2,
        )

    # ========================================================
    # BNO055 - ORIENTATION
    # ========================================================

    def _simulate_orientation(self) -> float:
        """Generate simulated BNO055 orientation."""

        base = self._base_value(
            "orientation",
            DEFAULT_SENSOR_VALUES["orientation"],
        )

        variation = {
            "NORMAL": 2.0,
            "WALKING": 3.0,
            "RUNNING": 4.0,
            "FALL": 2.0,
            "HIGH_RISK": 3.0,
            "SOS": 2.0,
        }.get(
            self.current_scenario,
            2.0,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            self._clamp(
                value,
                MIN_ORIENTATION,
                MAX_ORIENTATION,
            ),
            1,
        )

    # ========================================================
    # BME680 - ENVIRONMENTAL TEMPERATURE
    # ========================================================

    def _simulate_environmental_temperature(
        self,
    ) -> float:
        """
        Generate simulated BME680 environmental temperature.

        IMPORTANT:
            This is environmental temperature.

        It must never be confused with MAX30208
        body_temperature.
        """

        base = self._base_value(
            "temperature",
            DEFAULT_SENSOR_VALUES["temperature"],
        )

        value = self._noise(
            base,
            0.25,
        )

        return round(
            self._clamp(
                value,
                MIN_ENVIRONMENTAL_TEMPERATURE,
                MAX_ENVIRONMENTAL_TEMPERATURE,
            ),
            1,
        )

    # ========================================================
    # BME680 - HUMIDITY
    # ========================================================

    def _simulate_humidity(self) -> float:
        """Generate simulated BME680 humidity."""

        base = self._base_value(
            "humidity",
            DEFAULT_SENSOR_VALUES["humidity"],
        )

        value = self._noise(
            base,
            0.8,
        )

        return round(
            self._clamp(
                value,
                MIN_HUMIDITY,
                MAX_HUMIDITY,
            ),
            1,
        )

    # ========================================================
    # BME680 - PRESSURE
    # ========================================================

    def _simulate_pressure(self) -> float:
        """Generate simulated BME680 atmospheric pressure."""

        base = self._base_value(
            "pressure",
            DEFAULT_SENSOR_VALUES["pressure"],
        )

        value = self._noise(
            base,
            1.2,
        )

        return round(
            value,
            1,
        )

    # ========================================================
    # INMP441 - AUDIO
    # ========================================================

    def _simulate_audio(self) -> float:
        """Generate simulated INMP441 audio level."""

        base = self._base_value(
            "audio_level",
            DEFAULT_SENSOR_VALUES["audio_level"],
        )

        variation = {
            "NORMAL": 0.03,
            "WALKING": 0.04,
            "RUNNING": 0.06,
            "FALL": 0.08,
            "HIGH_RISK": 0.06,
            "SOS": 0.04,
        }.get(
            self.current_scenario,
            0.04,
        )

        value = self._noise(
            base,
            variation,
        )

        return round(
            self._clamp(
                value,
                MIN_AUDIO_LEVEL,
                MAX_AUDIO_LEVEL,
            ),
            2,
        )

    # ========================================================
    # GPS
    # ========================================================

    def _simulate_gps(
        self,
    ) -> Dict[str, float]:
        """
        Generate simulated GPS coordinates.

        Walking and running profiles receive small positional
        changes to demonstrate live location movement.

        Every profile transition re-anchors the GPS position to
        the newly selected profile.
        """

        latitude = self._base_value(
            "latitude",
            DEFAULT_SENSOR_VALUES["latitude"],
        )

        longitude = self._base_value(
            "longitude",
            DEFAULT_SENSOR_VALUES["longitude"],
        )

        if self.current_scenario in (
            "WALKING",
            "RUNNING",
        ):

            latitude += self._random.uniform(
                -0.0001,
                0.0002,
            )

            longitude += self._random.uniform(
                -0.0001,
                0.0002,
            )

        self.previous_latitude = latitude

        self.previous_longitude = longitude

        return {
            "latitude": round(
                latitude,
                6,
            ),
            "longitude": round(
                longitude,
                6,
            ),
        }

    # ========================================================
    # COMPLETE SENSOR SAMPLE
    # ========================================================

    def generate(
        self,
    ) -> Dict[str, Any]:
        """
        Generate one complete simulated sensor sample.

        The returned dictionary is the common sensor-data
        contract consumed by the AI, fusion, risk and dashboard
        layers.

        IMPORTANT
        ---------
        The sample is generated exclusively from the currently
        selected profile.

        In particular:

            manual_sos

        is read from the current profile every time.

        Therefore a previous SOS profile cannot remain active
        after switching to another profile.
        """

        self.sample_count += 1

        acceleration = (
            self._simulate_acceleration()
        )

        gps = self._simulate_gps()

        # ----------------------------------------------------
        # READ CURRENT PROFILE EMERGENCY STATE
        # ----------------------------------------------------

        manual_sos = bool(
            self.data.get(
                "manual_sos",
                False,
            )
        )

        # ----------------------------------------------------
        # COMPLETE SAMPLE
        # ----------------------------------------------------

        sample = {

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),

            "sample_id": self.sample_count,

            "scenario": self.current_scenario,

            "simulated": True,

            # ------------------------------------------------
            # MAX30102
            # ------------------------------------------------

            "heart_rate": (
                self._simulate_heart_rate()
            ),

            "spo2": (
                self._simulate_spo2()
            ),

            # ------------------------------------------------
            # MAX30208
            # ------------------------------------------------

            "body_temperature": (
                self._simulate_body_temperature()
            ),

            # ------------------------------------------------
            # BNO055
            # ------------------------------------------------

            "acceleration_x": (
                acceleration["acceleration_x"]
            ),

            "acceleration_y": (
                acceleration["acceleration_y"]
            ),

            "acceleration_z": (
                acceleration["acceleration_z"]
            ),

            "motion_intensity": (
                self._simulate_motion()
            ),

            "orientation": (
                self._simulate_orientation()
            ),

            # ------------------------------------------------
            # BME680
            # ------------------------------------------------

            "temperature": (
                self._simulate_environmental_temperature()
            ),

            "humidity": (
                self._simulate_humidity()
            ),

            "pressure": (
                self._simulate_pressure()
            ),

            # ------------------------------------------------
            # INMP441
            # ------------------------------------------------

            "audio_level": (
                self._simulate_audio()
            ),

            # ------------------------------------------------
            # GPS
            # ------------------------------------------------

            "latitude": gps["latitude"],

            "longitude": gps["longitude"],

            # ------------------------------------------------
            # Explicit emergency input
            # ------------------------------------------------

            "manual_sos": manual_sos,
        }

        return sample

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
    ) -> None:
        """
        Reset the simulator to the configured default profile.

        This is a complete state reset:
            - profile
            - sensor profile data
            - sample counter
            - GPS anchor
            - simulation start time
        """

        default_profile = get_scenario(
            DEFAULT_SCENARIO
        )

        self.current_scenario = (
            DEFAULT_SCENARIO
        )

        self.data = default_profile

        self.sample_count = 0

        self.start_time = datetime.now()

        self.previous_latitude = float(
            self.data.get(
                "latitude",
                DEFAULT_SENSOR_VALUES["latitude"],
            )
        )

        self.previous_longitude = float(
            self.data.get(
                "longitude",
                DEFAULT_SENSOR_VALUES["longitude"],
            )
        )


# ============================================================
# GLOBAL SIMULATOR
# ============================================================

_simulator = SimulatedSensorData()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def generate_sensor_data(
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate one simulated sensor sample.

    Parameters
    ----------
    scenario:
        Optional development profile.

        When supplied, that profile becomes active.

        When omitted, the currently active profile continues.

    Notes
    -----
    The scenario parameter is retained for testing and backward
    compatibility.

    The normal dashboard pipeline should call this function
    without supplying an activity state.
    """

    if scenario is not None:

        _simulator.set_scenario(
            scenario
        )

    return _simulator.generate()


def set_simulation_scenario(
    scenario: str,
) -> Dict[str, Any]:
    """
    Set the active development simulation profile.

    Intended for:
        - testing
        - demonstrations
        - automated validation

    A profile switch resets profile-specific transient state.
    """

    return _simulator.set_scenario(
        scenario
    )


def get_simulation_scenario() -> str:
    """Return the currently active simulation profile."""

    return _simulator.get_scenario()


def get_simulation_profiles() -> list:
    """Return available development profiles."""

    return _simulator.get_available_profiles()


def reset_simulator() -> None:
    """Reset the simulator to the configured default profile."""

    _simulator.reset()


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "SimulatedSensorData",
    "generate_sensor_data",
    "set_simulation_scenario",
    "get_simulation_scenario",
    "get_simulation_profiles",
    "reset_simulator",
]