from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tools.feature_engineering.feature_extraction import (
    FEATURE_COLUMNS,
    extract_features,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_index": [0, 1, 2],
            "elapsed_ms": [0, 10, 20],
            "accel_x_m_s2": [0.0, 2.0, 4.0],
            "accel_y_m_s2": [0.0, 0.0, 0.0],
            "accel_z_m_s2": [0.0, 0.0, 0.0],
            "gyro_x_rad_s": [0.0, 0.0, 0.0],
            "gyro_y_rad_s": [0.0, 0.0, 0.0],
            "gyro_z_rad_s": [0.0, 0.0, 0.0],
            "heading_deg": [359.0, 0.0, 1.0],
            "roll_deg": [0.0, 0.0, 0.0],
            "pitch_deg": [0.0, 0.0, 0.0],
            "temperature_C": [-999.0, -999.0, -999.0],
            "humidity_pct": [-999.0, -999.0, -999.0],
            "pressure_hPa": [-999.0, -999.0, -999.0],
            "gas_kOhm": [-999.0, -999.0, -999.0],
            "bme_valid": [0, 0, 0],
            "mic_rms": [0.0, 0.0, 0.0],
            "mic_dc": [0.0, 0.0, 0.0],
            "mic_average_rms": [0.0, 0.0, 0.0],
            "mic_max_rms": [0.0, 0.0, 0.0],
            "mic_blocks": [0, 0, 0],
            "max_ir": [100.0, 100.0, 100.0],
            "max_red": [90.0, 90.0, 90.0],
            "finger_detected": [1, 1, 1],
            "raw_hr": [-999.0, -999.0, 90.0],
            "raw_hr_valid": [0, 0, 1],
            "raw_spo2": [-999.0, -999.0, 98.0],
            "raw_spo2_valid": [0, 0, 1],
            "final_hr": [150.0, 150.0, 150.0],
            "final_hr_valid": [1, 1, 1],
            "final_spo2": [99.0, 99.0, 99.0],
            "final_spo2_valid": [1, 1, 1],
            "battery_voltage_V": [4.0, 4.0, 4.0],
            "battery_percent": [75, 75, 75],
            "charger_connected": [0, 0, 0],
        }
    )


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_contract_is_fixed_and_finite(self) -> None:
        features = extract_features(sample_frame())

        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertTrue(np.isfinite(features.to_numpy()).all())

    def test_invalid_ppg_is_masked_and_valid_ppg_is_preserved(self) -> None:
        features = extract_features(sample_frame())

        self.assertEqual(features.loc[0, "raw_hr_bpm"], 0.0)
        self.assertEqual(features.loc[0, "raw_hr_available"], 0.0)
        self.assertEqual(features.loc[2, "raw_hr_bpm"], 90.0)
        self.assertEqual(features.loc[2, "raw_hr_available"], 1.0)

    def test_ppg_signal_is_masked_when_contact_is_absent(self) -> None:
        frame = sample_frame()
        frame.loc[2, "finger_detected"] = 0

        features = extract_features(frame)

        self.assertEqual(features.loc[2, "ppg_max_ir"], 0.0)
        self.assertEqual(features.loc[2, "ppg_available"], 0.0)
        self.assertEqual(features.loc[2, "raw_hr_available"], 0.0)

    def test_valid_sentinel_is_rejected(self) -> None:
        frame = sample_frame()
        frame.loc[2, "raw_hr"] = -999.0

        with self.assertRaisesRegex(ValueError, "raw_hr_valid=1"):
            extract_features(frame)

    def test_invalid_environment_is_masked(self) -> None:
        features = extract_features(sample_frame())

        self.assertTrue((features["temperature_C"] == 0.0).all())
        self.assertTrue((features["environment_available"] == 0.0).all())

    def test_rolling_features_are_causal_and_timing_aware(self) -> None:
        features = extract_features(sample_frame())

        self.assertAlmostEqual(
            features.loc[1, "rolling_accel_mean_1s_m_s2"],
            1.0,
        )
        self.assertAlmostEqual(
            features.loc[2, "rolling_accel_mean_1s_m_s2"],
            2.0,
        )
        self.assertAlmostEqual(features.loc[1, "heading_rate_deg_s"], 100.0)

    def test_timing_stall_is_explicit(self) -> None:
        frame = sample_frame()
        frame["elapsed_ms"] = [0, 10, 1510]

        features = extract_features(frame)

        self.assertEqual(features.loc[2, "timing_interval_ms"], 1500.0)
        self.assertEqual(features.loc[2, "timing_stall"], 1.0)

    def test_missing_optional_sensors_have_clear_masks(self) -> None:
        features = extract_features(
            pd.DataFrame(
                {
                    "sample_index": [0, 1],
                    "elapsed_ms": [0, 10],
                }
            )
        )

        self.assertTrue((features["accel_available"] == 0.0).all())
        self.assertTrue((features["gyro_available"] == 0.0).all())
        self.assertTrue((features["audio_available"] == 0.0).all())

    def test_non_monotonic_timestamps_are_rejected(self) -> None:
        frame = sample_frame()
        frame["elapsed_ms"] = [0, 20, 10]

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            extract_features(frame)


if __name__ == "__main__":
    unittest.main()
