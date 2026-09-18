"""
SafeBand AI - BNO055 Synthetic Dataset Visualization V2

Purpose
-------
Generate deterministic, representative temporal plots from the BNO055-shaped
synthetic dataset before any ML training is performed.

Input
-----
datasets/synthetic/bno055/bno055_synthetic_v2.csv

Output
------
reports/figures/bno055_synthetic/
    activity_<label>_window.png
    selected_windows.csv

The script is deliberately read-only with respect to the input dataset.

Important
---------
This dataset is synthetic. These plots are for pipeline/integration sanity
checking and preliminary experimentation, not real BNO055 hardware validation.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository-root import path. Kept for consistency with SafeBand tools.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


ACTIVITIES = [
    "FALL",
    "LYING",
    "RUNNING",
    "SITTING",
    "SIT_TO_STAND",
    "STAIRS",
    "STANDING",
    "STAND_TO_SIT",
    "WALKING",
]

ACCEL_COLS = [
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
]

GYRO_COLS = [
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
]

MAG_COLS = [
    "mag_x_uT",
    "mag_y_uT",
    "mag_z_uT",
]

REQUIRED_COLS = [
    "subject_id",
    "session_id",
    "timestamp_s",
    "activity_label",
    *ACCEL_COLS,
    *GYRO_COLS,
    *MAG_COLS,
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize representative BNO055 synthetic sensor windows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "datasets" / "synthetic" / "bno055"
        / "bno055_synthetic_v2.csv",
        help="Input BNO055 synthetic CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "figures" / "bno055_synthetic",
        help="Directory for generated figures and selected-window metadata.",
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="Optional subject ID to visualize. If omitted, use the first "
             "subject containing every requested activity.",
    )
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help="Optional session ID to visualize.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Window duration in seconds. Default: 2.0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic selection seed. Default: 42.",
    )
    return parser.parse_args()


def validate_schema(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required columns:\n  - "
            + "\n  - ".join(missing)
        )

    if df.empty:
        raise ValueError("Input CSV is empty.")

    numeric_cols = [
        "subject_id",
        "session_id",
        "timestamp_s",
        *ACCEL_COLS,
        *GYRO_COLS,
        *MAG_COLS,
    ]

    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be numeric.")

        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains non-finite values.")

    if df["activity_label"].isna().any():
        raise ValueError("Column 'activity_label' contains missing values.")


def choose_subject_session(df: pd.DataFrame, requested_subject, requested_session):
    """Choose one subject/session containing all nine activities."""
    candidates = df

    if requested_subject is not None:
        candidates = candidates[candidates["subject_id"] == requested_subject]

    if requested_session is not None:
        candidates = candidates[candidates["session_id"] == requested_session]

    if candidates.empty:
        raise ValueError("No rows match the requested subject/session filter.")

    available = (
        candidates.groupby(["subject_id", "session_id"])["activity_label"]
        .agg(lambda s: set(s.astype(str)))
    )

    required = set(ACTIVITIES)
    complete = available[available.apply(lambda s: required.issubset(s))]

    if complete.empty:
        raise ValueError(
            "Could not find one subject/session containing all nine activities. "
            "Use --subject/--session only if the requested recording is known "
            "to contain all activities."
        )

    # Deterministic: choose lowest subject/session pair.
    subject_id, session_id = sorted(complete.index.tolist())[0]
    return int(subject_id), int(session_id)


def select_activity_window(df, subject_id, session_id, activity, duration, rng):
    """Select a contiguous window from one activity without crossing boundaries."""
    part = df[
        (df["subject_id"] == subject_id)
        & (df["session_id"] == session_id)
        & (df["activity_label"] == activity)
    ].copy()

    if part.empty:
        raise ValueError(
            f"No samples found for activity '{activity}' "
            f"for subject={subject_id}, session={session_id}."
        )

    part = part.sort_values("timestamp_s").reset_index(drop=True)

    timestamps = part["timestamp_s"].to_numpy(dtype=float)

    if len(part) < 2:
        raise ValueError(f"Not enough samples for activity '{activity}'.")

    dt = np.diff(timestamps)
    positive_dt = dt[dt > 0]
    if positive_dt.size == 0:
        raise ValueError(f"Invalid timestamps for activity '{activity}'.")

    sample_period = float(np.median(positive_dt))
    expected_samples = max(2, int(round(duration / sample_period)))

    if len(part) < expected_samples:
        raise ValueError(
            f"Activity '{activity}' has only {len(part)} samples, "
            f"but {expected_samples} are required for {duration:.2f}s."
        )

    # Build candidate starts only where the requested duration remains within
    # the same activity recording and timestamps stay reasonably regular.
    max_start = len(part) - expected_samples
    if max_start == 0:
        starts = np.array([0], dtype=int)
    else:
        starts = np.arange(max_start + 1, dtype=int)

    # Require a contiguous timestamp sequence. The synthetic generator is
    # expected to be regular, but this protects the plotting tool from
    # accidentally crossing gaps.
    valid_starts = []
    max_gap = sample_period * 1.5

    for start in starts:
        end = start + expected_samples
        local_dt = np.diff(timestamps[start:end])
        if local_dt.size == 0 or np.all(local_dt > 0):
            if np.max(local_dt, initial=0.0) <= max_gap:
                valid_starts.append(start)

    if not valid_starts:
        raise ValueError(
            f"No contiguous {duration:.2f}s window found for '{activity}'."
        )

    start = int(rng.choice(valid_starts))
    end = start + expected_samples

    window = part.iloc[start:end].copy()

    # Re-zero time for easier visual comparison.
    window["time_rel_s"] = window["timestamp_s"] - window["timestamp_s"].iloc[0]

    return window


def add_derived_signals(window):
    window = window.copy()

    ax = window[ACCEL_COLS].to_numpy(dtype=float)
    gx = window[GYRO_COLS].to_numpy(dtype=float)

    window["acc_mag"] = np.linalg.norm(ax, axis=1)
    window["gyro_mag"] = np.linalg.norm(gx, axis=1)

    return window


def plot_window(window, activity, subject_id, session_id, output_path):
    t = window["time_rel_s"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    # Accelerometer
    axes[0].plot(t, window["accel_x_mps2"], label="X")
    axes[0].plot(t, window["accel_y_mps2"], label="Y")
    axes[0].plot(t, window["accel_z_mps2"], label="Z")
    axes[0].plot(t, window["acc_mag"], label="Magnitude", linewidth=2)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title(
        f"{activity} | Subject {subject_id} | Session {session_id}"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(ncol=4)

    # Gyroscope
    axes[1].plot(t, window["gyro_x_dps"], label="X")
    axes[1].plot(t, window["gyro_y_dps"], label="Y")
    axes[1].plot(t, window["gyro_z_dps"], label="Z")
    axes[1].plot(t, window["gyro_mag"], label="Magnitude", linewidth=2)
    axes[1].set_ylabel("Angular rate (°/s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(ncol=4)

    # Magnetometer
    axes[2].plot(t, window["mag_x_uT"], label="X")
    axes[2].plot(t, window["mag_y_uT"], label="Y")
    axes[2].plot(t, window["mag_z_uT"], label="Z")
    axes[2].set_ylabel("Magnetic field (µT)")
    axes[2].set_xlabel("Time from window start (s)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(ncol=3)

    fig.suptitle(
        "SafeBand AI — BNO055 Synthetic Sensor Window",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_dir}")

    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded {len(df):,} rows.")

    validate_schema(df)

    subject_id, session_id = choose_subject_session(
        df, args.subject, args.session
    )

    print(
        f"[INFO] Selected subject={subject_id}, session={session_id} "
        f"for deterministic visualization."
    )

    rng = np.random.default_rng(args.seed)

    selected_rows = []

    for activity in ACTIVITIES:
        window = select_activity_window(
            df,
            subject_id,
            session_id,
            activity,
            args.duration,
            rng,
        )
        window = add_derived_signals(window)

        safe_name = activity.lower()
        figure_path = output_dir / f"activity_{safe_name}_window.png"

        plot_window(
            window,
            activity,
            subject_id,
            session_id,
            figure_path,
        )

        selected_rows.append(
            {
                "activity_label": activity,
                "subject_id": subject_id,
                "session_id": session_id,
                "start_timestamp_s": float(window["timestamp_s"].iloc[0]),
                "end_timestamp_s": float(window["timestamp_s"].iloc[-1]),
                "samples": int(len(window)),
                "duration_s": float(
                    window["timestamp_s"].iloc[-1]
                    - window["timestamp_s"].iloc[0]
                ),
                "acc_mag_mean": float(window["acc_mag"].mean()),
                "acc_mag_std": float(window["acc_mag"].std()),
                "gyro_mag_mean": float(window["gyro_mag"].mean()),
                "gyro_mag_std": float(window["gyro_mag"].std()),
            }
        )

        print(f"[OK] {activity:15s} -> {figure_path.name}")

    summary_path = output_dir / "selected_windows.csv"
    pd.DataFrame(selected_rows).to_csv(summary_path, index=False)

    print()
    print("[PASS] Visualization generation completed successfully.")
    print(f"[PASS] Figures : {output_dir}")
    print(f"[PASS] Summary : {summary_path}")


if __name__ == "__main__":
    main()
