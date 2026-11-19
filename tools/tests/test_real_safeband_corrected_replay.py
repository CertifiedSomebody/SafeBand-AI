import numpy as np
import pandas as pd

from tools.evaluate_real_safeband_corrected_replay import (
    elapsed_windows,
    normalize_evaluation_activity,
    observed_metrics,
)


def test_label_normalization_preserves_original():
    result = normalize_evaluation_activity("STATIONARY")
    assert result["original_activity"] == "STATIONARY"
    assert result["evaluation_activity"] == "STATIC"

    walking = normalize_evaluation_activity("WALKING")
    assert walking["original_activity"] == "WALKING"
    assert walking["evaluation_activity"] == "WALKING"


def test_elapsed_windowing_is_irregular_not_row_count():
    slow = pd.DataFrame({"elapsed_ms": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]})
    fast = pd.DataFrame({"elapsed_ms": [0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]})
    slow_windows = elapsed_windows(slow, 1000)
    fast_windows = elapsed_windows(fast, 1000)

    assert len(slow_windows) == 1
    assert len(fast_windows) == 8
    # Irregular 400 ms spacing produces windows from different elapsed spans
    # instead of assuming the same 10-row/1000 ms relationship.
    assert fast_windows[0]["actual_duration_ms"] == 800.0


def test_elapsed_window_durations_use_elapsed_ms():
    df = pd.DataFrame({"elapsed_ms": [0, 390, 780, 1170, 1560, 1950, 2340, 2730]})
    windows = elapsed_windows(df, 1000)
    assert windows
    assert all(window["start_timestamp_ms"] <= window["prediction_timestamp_ms"] for window in windows)
    assert all(window["actual_duration_ms"] >= 0 for window in windows)
    assert all(window["actual_duration_ms"] <= 1000.5 for window in windows)


def test_stall_detection_flags_large_gap():
    df = pd.DataFrame({"elapsed_ms": [0, 200, 400, 1600, 2000, 2400]})
    windows = elapsed_windows(df, 1000)
    assert windows
    assert any(window["contains_timing_stall"] for window in windows)


def test_causal_semantics_never_uses_future_samples():
    df = pd.DataFrame({"elapsed_ms": [0, 400, 800, 1200, 1600, 2000, 2400]})
    windows = elapsed_windows(df, 1000)
    for window in windows:
        assert window["start_timestamp_ms"] >= window["prediction_timestamp_ms"] - 1000
        assert window["end_timestamp_ms"] <= window["prediction_timestamp_ms"]
        assert max(df.iloc[window["selected_original_rows"]]["elapsed_ms"]) <= window["prediction_timestamp_ms"]


def test_absent_class_is_not_evaluable():
    result = observed_metrics(["STATIC"] * 5, ["STATIC"] * 5)
    static = next(item for item in result["per_class"] if item["class"] == "STATIC")
    walking = next(item for item in result["per_class"] if item["class"] == "WALKING")
    running = next(item for item in result["per_class"] if item["class"] == "RUNNING")
    assert static["evaluation_status"] == "OBSERVED_CLASS_PERFORMANCE"
    assert walking["evaluation_status"] == "ABSENT_CLASS_NOT_EVALUABLE"
    assert walking["precision"] is None
    assert running["evaluation_status"] == "ABSENT_CLASS_NOT_EVALUABLE"
