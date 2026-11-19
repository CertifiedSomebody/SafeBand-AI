import numpy as np
from tools.evaluate_real_safeband_validation import coarse_label, metrics, sliding_windows
import pandas as pd


def test_coarse_label_mapping():
    assert coarse_label("STATIONARY") == "STATIC"
    assert coarse_label("SITTING") == "STATIC"
    assert coarse_label("WALKING") == "WALKING"
    assert coarse_label("RUNNING") == "RUNNING"


def test_sliding_windows_are_contiguous_and_overlap():
    df = pd.DataFrame({"x": range(10)})
    windows = sliding_windows(df, 4)
    assert len(windows) == 4
    assert len(windows[0]) == 4
    assert windows[0].iloc[-1]["x"] == 3
    assert windows[1].iloc[0]["x"] == 2


def test_metrics_support_empty_predicted_classes():
    result = metrics(["STATIC"] * 4, ["STATIC"] * 4, ["STATIC", "WALKING"])
    assert result["accuracy"] == 1.0
    assert result["confusion_matrix"] == [[4, 0], [0, 0]]
