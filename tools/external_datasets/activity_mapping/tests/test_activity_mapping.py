from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from tools.external_datasets.activity_mapping.adapters.aicos_har_adapter import (
    _read_accelerometer,
    iter_recordings,
)
from tools.external_datasets.activity_mapping.adapters.common import (
    Recording,
    resample_segment,
    rolling_magnitude_statistics,
    safe_band_activity,
    validate_recording,
    window_features,
)
from tools.external_datasets.activity_mapping.adapters.sisfall_adapter import (
    G_PER_COUNT_ADXL345,
    STANDARD_GRAVITY_M_S2,
    _parse_recording,
)
from tools.external_datasets.activity_mapping.build_activity_dataset import (
    MODEL_FEATURES,
    OUTPUT_FIELDS,
)
from tools.external_datasets.activity_mapping.evaluate_activity_benchmarks import (
    grouped_partitions,
)
from tools.feature_engineering.feature_extraction import extract_features


class ActivityMappingTests(unittest.TestCase):
    def test_sisfall_acceleration_counts_are_converted_to_m_s2(self) -> None:
        raw = b"256,0,0,0,0,0,0,0,0;\n"
        result = _parse_recording(raw, "D01_SA01_R01.txt")
        self.assertEqual(result.shape, (1, 3))
        self.assertAlmostEqual(
            result[0, 0],
            256 * G_PER_COUNT_ADXL345 * STANDARD_GRAVITY_M_S2,
        )

    def test_aicos_timestamp_ns_is_read_as_seconds(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            for delimiter in (",", "|"):
                with self.subTest(delimiter=delimiter):
                    source = Path(temp_dir) / "acc.txt"
                    source.write_text(
                        delimiter.join(("5000000000", "1", "2", "3")) + "\n"
                        + delimiter.join(("5010000000", "4", "5", "6")) + "\n"
                        + delimiter.join(("5020000000", "7", "8", "9")) + "\n",
                        encoding="utf-8",
                    )
                    timestamps, values = _read_accelerometer(source)
                    np.testing.assert_allclose(timestamps, [5.0, 5.01, 5.02])
                    self.assertEqual(values.shape, (3, 3))
                    self.assertEqual(values[1].tolist(), [4.0, 5.0, 6.0])

    def test_aicos_unique_se2_path_fallback_is_recorded(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            acquisition_dir = root / "S1" / "Walking_1"
            sensor_file = acquisition_dir / "AppleiPhoneSE2_Hand" / "Accelerometer.txt"
            sensor_file.parent.mkdir(parents=True)
            sensor_file.write_text(
                "0,1,2,3\n10000000,4,5,6\n20000000,7,8,9\n",
                encoding="utf-8",
            )
            metadata = root / "metadata.csv"
            metadata.write_text(
                "AcquisitionID\nS1/Walking_1/AppleiPhone_Hand\n",
                encoding="utf-8",
            )

            recordings = list(iter_recordings(root, metadata))

        self.assertEqual(len(recordings), 1)
        self.assertEqual(recordings[0].source_path_resolution, "unique_se2_same_placement_fallback")
        self.assertEqual(Path(recordings[0].source_path).name, "Accelerometer.txt")

    def test_aicos_ambiguous_path_fallback_is_rejected(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            acquisition_dir = root / "S1" / "Walking_1"
            for device in ("AppleiPhoneSE2_Hand", "OtherPhone_Hand"):
                sensor_file = acquisition_dir / device / "Accelerometer.txt"
                sensor_file.parent.mkdir(parents=True)
                sensor_file.write_text("0,1,2,3\n", encoding="utf-8")
            metadata = root / "metadata.csv"
            metadata.write_text(
                "AcquisitionID\nS1/Walking_1/AppleiPhone_Hand\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "AICOS accelerometer is missing"):
                list(iter_recordings(root, metadata))

    def test_resampling_is_monotonic_and_keeps_source_boundaries(self) -> None:
        source_t = np.array([0.0, 0.01, 0.02, 0.03])
        source_x = np.column_stack((source_t, source_t * 2, source_t * 0))
        target_t, target_x = resample_segment(source_t, source_x, target_hz=50)
        self.assertTrue(np.all(np.diff(target_t) > 0))
        self.assertEqual(target_t[0], 0)
        self.assertLessEqual(target_t[-1], source_t[-1])
        self.assertEqual(target_x.shape, (len(target_t), 3))

    def test_label_mapping(self) -> None:
        cases = [
            ("aicos_har", "Sitting", "STATIC"),
            ("aicos_har", "Running", "RUNNING"),
            ("aicos_har", "LiftUp", "UNMAPPED"),
            ("sisfall", "D12", "TRANSITION"),
            ("sisfall", "D18", "HIGH_MOTION"),
            ("sisfall", "F05", "FALL_OR_HAZARD"),
        ]
        for dataset, source, expected in cases:
            with self.subTest(dataset=dataset, source=source):
                self.assertEqual(safe_band_activity(dataset, source)[0], expected)

    def test_recording_validation_rejects_nonfinite_values(self) -> None:
        recording = Recording(
            dataset_id="test",
            subject_id="s1",
            session_id="s1-r1",
            source_label="walking",
            timestamps_s=np.array([0.0, 0.02]),
            accel_xyz_m_s2=np.array([[0.0, 0.0, 9.8], [np.nan, 0.0, 9.8]]),
            sample_rate_hz=50,
            source_unit="m/s^2",
            timestamp_source_unit="s",
            source_path="fixture",
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_recording(recording)

    def test_window_features_use_runtime_compatible_acceleration_and_timing(self) -> None:
        times = np.arange(50, dtype=float) / 50
        acceleration = np.column_stack((
            np.zeros(50),
            np.zeros(50),
            9.80665 + np.sin(times * 2 * np.pi),
        ))
        magnitude = np.linalg.norm(acceleration, axis=1)
        rolling = rolling_magnitude_statistics(times, magnitude)
        features = window_features(
            times,
            acceleration,
            rolling_mean=rolling[0],
            rolling_std=rolling[1],
            rolling_max=rolling[2],
        )
        self.assertEqual(set(features), set(MODEL_FEATURES))
        self.assertEqual(features["feature__accel_available_fraction"], 1)
        self.assertEqual(features["feature__gyro_available_fraction"], 0)
        self.assertAlmostEqual(features["feature__timing_interval_ms_mean"], 20)
        self.assertNotIn("dataset_id", MODEL_FEATURES)
        self.assertFalse(set(MODEL_FEATURES) & set(OUTPUT_FIELDS[:13]))

    def test_external_magnitude_and_rolling_values_match_runtime_extractor(self) -> None:
        times = np.arange(60, dtype=float) / 50
        acceleration = np.column_stack((
            np.sin(times),
            np.cos(times * 0.5),
            np.full(len(times), 9.80665),
        ))
        runtime = extract_features(pd.DataFrame({
            "elapsed_ms": times * 1000,
            "sample_index": np.arange(len(times)),
            "accel_x_m_s2": acceleration[:, 0],
            "accel_y_m_s2": acceleration[:, 1],
            "accel_z_m_s2": acceleration[:, 2],
        }))
        magnitude = np.linalg.norm(acceleration, axis=1)
        rolling = rolling_magnitude_statistics(times, magnitude)
        np.testing.assert_allclose(
            runtime["accel_magnitude_m_s2"].to_numpy(), magnitude
        )
        np.testing.assert_allclose(
            runtime["rolling_accel_mean_1s_m_s2"].to_numpy(),
            rolling[0],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            runtime["rolling_accel_std_1s_m_s2"].to_numpy(),
            rolling[1],
            atol=1e-8,
        )

    def test_grouped_split_has_no_subject_leakage(self) -> None:
        labels = np.tile(np.array([0, 1]), 20)
        groups = np.repeat(np.array([f"s{i}" for i in range(20)]), 2)
        train, validation, test = grouped_partitions(labels, groups)
        train_groups = set(groups[train])
        validation_groups = set(groups[validation])
        test_groups = set(groups[test])
        self.assertTrue(train_groups.isdisjoint(validation_groups))
        self.assertTrue(train_groups.isdisjoint(test_groups))
        self.assertTrue(validation_groups.isdisjoint(test_groups))


if __name__ == "__main__":
    unittest.main()
