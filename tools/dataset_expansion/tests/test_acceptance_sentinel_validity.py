import unittest

import pandas as pd

from tools.dataset_expansion.acceptance_sentinel_validity import (
    validate_measurement_sentinels,
)


class MeasurementSentinelValidationTests(unittest.TestCase):
    def valid_frame(self):
        return pd.DataFrame(
            {
                "raw_hr": [-999.0, 72.0],
                "raw_hr_valid": [0, 1],
                "raw_spo2": [-999.0, 98.0],
                "raw_spo2_valid": [0, 1],
                "final_hr": [-999.0, 72.0],
                "final_hr_valid": [0, 1],
                "final_spo2": [-999.0, 98.0],
                "final_spo2_valid": [0, 1],
            }
        )

    def test_valid_readings_must_not_be_sentinels(self):
        frame = self.valid_frame()
        valid, errors = validate_measurement_sentinels(frame)

        self.assertTrue(valid, errors)

    def test_zero_validity_reading_may_be_non_sentinel(self):
        frame = self.valid_frame()
        frame.loc[0, "raw_hr"] = 65.0

        valid, errors = validate_measurement_sentinels(frame)

        self.assertTrue(valid, errors)

    def test_sentinel_with_valid_flag_is_rejected(self):
        frame = self.valid_frame()
        frame.loc[0, "raw_hr_valid"] = 1

        valid, errors = validate_measurement_sentinels(frame)

        self.assertFalse(valid)
        self.assertTrue(
            any("must have validity flag 0" in error for error in errors)
        )

    def test_valid_nan_is_rejected(self):
        frame = self.valid_frame()
        frame.loc[1, "raw_hr"] = float("nan")

        valid, errors = validate_measurement_sentinels(frame)

        self.assertFalse(valid)
        self.assertTrue(
            any("valid rows contain NaN" in error for error in errors)
        )


if __name__ == "__main__":
    unittest.main()
