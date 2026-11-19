import hashlib

import numpy as np
import pandas as pd

from tools.generate_realistic_synthetic_sensor_v2 import (
    build_session,
    generate_elapsed_ms,
    real_calibration,
    session_seed,
)


def test_session_seeds_are_independent():
    seeds = [session_seed(20261119, i) for i in range(1, 25)]
    assert len(set(seeds)) == 24


def test_elapsed_ms_is_strictly_increasing_and_irregular():
    calibration = real_calibration()
    elapsed = generate_elapsed_ms(np.random.default_rng(17), 400, calibration)
    dt = np.diff(elapsed)
    assert np.all(dt > 0)
    assert np.std(dt) > 0
    assert np.max(dt) > 1000


def test_generated_session_is_physically_plausible():
    df, provenance = build_session(1, 20261119, real_calibration())
    assert provenance["source_type"] == "synthetic"
    assert provenance["real_row_reuse"] is False
    acc = df[["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]].to_numpy()
    magnitude = np.linalg.norm(acc, axis=1)
    assert np.min(magnitude) > 5.0
    assert np.max(magnitude) < 30.0
    assert df["elapsed_ms"].is_monotonic_increasing


def test_ppg_validity_has_sentinel_consistency():
    df, _ = build_session(2, 20261119, real_calibration())
    for value_column, validity_column in (
        ("raw_hr", "raw_hr_valid"),
        ("raw_spo2", "raw_spo2_valid"),
        ("final_hr", "final_hr_valid"),
        ("final_spo2", "final_spo2_valid"),
    ):
        valid = df[value_column][df[validity_column] == 1]
        invalid = df[value_column][df[validity_column] == 0]
        assert not np.any(np.isclose(valid.to_numpy(), -999))
        assert np.all(np.isclose(invalid.to_numpy(), -999))
