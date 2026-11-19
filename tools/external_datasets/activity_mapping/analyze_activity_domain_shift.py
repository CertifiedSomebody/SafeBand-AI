from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.external_datasets.activity_mapping.build_activity_dataset import (  # noqa: E402
    MODEL_FEATURES,
)
from tools.external_datasets.activity_mapping.adapters.common import (  # noqa: E402
    rolling_magnitude_statistics,
    window_features,
)

SAFE_RECORDINGS = {
    "stationary.csv": "datasets/real sensor data/stationary.csv",
    "stationary_2.csv": "datasets/real sensor data/stationary_2.csv",
    "walking.csv": "datasets/real sensor data/walking.csv",
}


def _safe_band_windows(path: Path, label: str) -> tuple[list[dict[str, float]], dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {
        "elapsed_ms", "sample_index", "accel_x_m_s2",
        "accel_y_m_s2", "accel_z_m_s2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing SafeBand anchor columns: {sorted(missing)}")
    timestamps = pd.to_numeric(frame["elapsed_ms"], errors="coerce").to_numpy(float) / 1000.0
    acceleration = frame[
        ["accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    available = np.isfinite(timestamps) & np.isfinite(acceleration).all(axis=1)
    available &= ~np.isclose(acceleration, -999.0, atol=1e-6, rtol=0.0).any(axis=1)
    if "accel_valid" in frame.columns:
        flags = pd.to_numeric(frame["accel_valid"], errors="coerce").fillna(0).to_numpy()
        available &= flags == 1
    timestamps = timestamps[available]
    acceleration = acceleration[available]
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    acceleration = acceleration[order]
    unique = np.r_[True, np.diff(timestamps) > 0]
    timestamps = timestamps[unique]
    acceleration = acceleration[unique]
    if len(timestamps) < 2:
        return [], {"source": str(path), "rows": len(frame), "usable_samples": len(timestamps)}

    gaps = np.diff(timestamps)
    median_interval = float(np.median(gaps))
    cuts = np.flatnonzero(gaps > max(0.25, 5 * median_interval)) + 1
    bounds = np.concatenate(([0], cuts, [len(timestamps)]))
    output: list[dict[str, float]] = []
    for left, right in zip(bounds[:-1], bounds[1:]):
        segment_t = timestamps[left:right]
        segment_a = acceleration[left:right]
        if len(segment_t) < 2:
            continue
        magnitude = np.linalg.norm(segment_a, axis=1)
        rolling_mean, rolling_std, rolling_max = rolling_magnitude_statistics(
            segment_t, magnitude
        )
        start_time = segment_t[0]
        while start_time + 1.0 <= segment_t[-1] + 1e-9:
            mask = (segment_t >= start_time) & (segment_t <= start_time + 1.0)
            if mask.sum() >= 2:
                output.append(window_features(
                    segment_t[mask],
                    segment_a[mask],
                    rolling_mean=rolling_mean[mask],
                    rolling_std=rolling_std[mask],
                    rolling_max=rolling_max[mask],
                ))
            start_time += 0.5
    return output, {
        "source": str(path),
        "activity": label,
        "rows": int(len(frame)),
        "usable_samples": int(len(timestamps)),
        "one_second_windows": int(len(output)),
        "session_independence": "not established for stationary files; descriptive anchor only",
    }


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare external activity features with SafeBand real-recording anchors.")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "datasets" / "external_processed" / "activity" / "activity_windows.csv.gz",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "external_evaluation" / "activity_domain_shift.json",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Processed external activity data missing: {args.input}")
    external = pd.read_csv(args.input, compression="gzip")
    external = external[
        external["dataset_id"].eq("aicos_har")
        & external["window_seconds"].astype(float).eq(1.0)
        & external["activity_safeband"].isin(["STATIC", "WALKING", "RUNNING"])
    ]
    anchors: dict[str, list[dict[str, float]]] = {}
    anchor_metadata = {}
    for name, relative_path in SAFE_RECORDINGS.items():
        path = args.repo_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"SafeBand anchor recording missing: {path}")
        values, metadata = _safe_band_windows(path, "WALKING" if name == "walking.csv" else "STATIC")
        anchors[name] = values
        anchor_metadata[name] = metadata

    comparisons: dict[str, Any] = {}
    for source_name, windows in anchors.items():
        if not windows:
            comparisons[source_name] = {
                "status": "INSUFFICIENT_VALID_WINDOWS",
                "anchor": anchor_metadata[source_name],
            }
            continue
        anchor_frame = pd.DataFrame(windows)
        results: dict[str, Any] = {}
        for dataset_id in sorted(external["dataset_id"].unique()):
            source_frame = external[external["dataset_id"].eq(dataset_id)]
            feature_results = {}
            for feature in MODEL_FEATURES:
                left = pd.to_numeric(anchor_frame[feature], errors="coerce").dropna().to_numpy(float)
                right = pd.to_numeric(source_frame[feature], errors="coerce").dropna().to_numpy(float)
                if not len(left) or not len(right):
                    feature_results[feature] = {
                        "anchor": _distribution(left),
                        "external": _distribution(right),
                        "ks_statistic": None,
                        "wasserstein_distance": None,
                    }
                    continue
                feature_results[feature] = {
                    "anchor": _distribution(left),
                    "external": _distribution(right),
                    "ks_statistic": float(ks_2samp(left, right).statistic),
                    "wasserstein_distance": float(wasserstein_distance(left, right)),
                }
            results[dataset_id] = {
                "window_seconds": 1.0,
                "rows": int(len(source_frame)),
                "features": feature_results,
            }
        comparisons[source_name] = {
            "status": "DESCRIPTIVE_ONLY",
            "anchor": anchor_metadata[source_name],
            "external_comparisons": results,
        }

    report = {
        "evaluation_scope": "DEVELOPMENT / COMPONENT-LEVEL EVALUATION",
        "interpretation": (
            "Descriptive domain-shift analysis only. It does not estimate SafeBand model performance. "
            "The two stationary files are not treated as independent sessions."
        ),
        "window_seconds": 1.0,
        "external_source": "AICOS-HAR (ACC, resampled from documented timestamps and units)",
        "features": list(MODEL_FEATURES),
        "anchors": comparisons,
        "known_limitations": [
            "SafeBand recordings are short per-file sensor summaries and lack independent provenance for the two stationary files.",
            "The external source uses heterogeneous devices and placements; rotation-invariant ACC features reduce, but do not eliminate, domain shift.",
            "KS and Wasserstein statistics are descriptive on correlated temporal windows; they are not inferential clinical evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "anchor_status": {name: result["status"] for name, result in comparisons.items()},
        "external_rows": int(len(external)),
    }, indent=2))


if __name__ == "__main__":
    main()
