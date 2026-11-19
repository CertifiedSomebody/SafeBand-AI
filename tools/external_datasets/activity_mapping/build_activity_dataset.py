from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.external_datasets.activity_mapping.adapters import (  # noqa: E402
    aicos_har_adapter,
    bits2_adapter,
    forth_trace_adapter,
    pamap2_adapter,
    sisfall_adapter,
)
from tools.external_datasets.activity_mapping.adapters.common import (  # noqa: E402
    Recording,
    contiguous_segments,
    resample_segment,
    rolling_magnitude_statistics,
    safe_band_activity,
    validate_recording,
    window_features,
)

TARGET_HZ = 50.0
WINDOW_SECONDS = (0.5, 1.0, 2.0, 3.0)
MAX_WINDOWS_PER_SESSION = 8
G_PER_COUNT_ADXL345 = sisfall_adapter.G_PER_COUNT_ADXL345
STANDARD_GRAVITY_M_S2 = sisfall_adapter.STANDARD_GRAVITY_M_S2

SISFALL_DESCRIPTIONS = {
    "D01": "Walking slowly",
    "D02": "Walking quickly",
    "D03": "Jogging slowly",
    "D04": "Jogging quickly",
    "D05": "Walking upstairs and downstairs slowly",
    "D06": "Walking upstairs and downstairs quickly",
    "D07": "Slowly sit in a half height chair, wait, and up slowly",
    "D08": "Quickly sit in a half height chair, wait, and up quickly",
    "D09": "Slowly sit in a low height chair, wait, and up slowly",
    "D10": "Quickly sit in a low height chair, wait, and up quickly",
    "D11": "Sit, try to get up, and collapse into a chair",
    "D12": "Sit, lie slowly, wait, and sit again",
    "D13": "Sit, lie quickly, wait, and sit again",
    "D14": "Change from supine to lateral position and back",
    "D15": "Stand, bend at knees, and get up",
    "D16": "Stand, bend without knees, and get up",
    "D17": "Get into a car, remain seated, and get out",
    "D18": "Stumble while walking",
    "D19": "Gently jump without falling",
    "F01": "Fall forward while walking caused by a slip",
    "F02": "Fall backward while walking caused by a slip",
    "F03": "Lateral fall while walking caused by a slip",
    "F04": "Fall forward while walking caused by a trip",
    "F05": "Fall forward while jogging caused by a trip",
    "F06": "Vertical fall while walking caused by fainting",
    "F07": "Fall while walking with hands used to dampen fall, caused by fainting",
    "F08": "Fall forward when trying to get up",
    "F09": "Lateral fall when trying to get up",
    "F10": "Fall forward while trying to sit down",
    "F11": "Fall backward while trying to sit down",
    "F12": "Lateral fall while trying to sit down",
    "F13": "Fall forward while sitting, caused by fainting or falling asleep",
    "F14": "Fall backward while sitting, caused by fainting or falling asleep",
    "F15": "Lateral fall while sitting, caused by fainting or falling asleep",
}

SISFALL_RECORDING_RATE_HZ = 200.0
SISFALL_BLOCK_SIZE = 4
SISFALL_LABEL_IDS = (
    {f"D{i:02d}" for i in range(1, 20)}
    | {f"F{i:02d}" for i in range(1, 16)}
)

MODEL_FEATURES = (
    "feature__accel_magnitude_mean_m_s2",
    "feature__accel_magnitude_std_m_s2",
    "feature__accel_magnitude_max_m_s2",
    "feature__accel_magnitude_range_m_s2",
    "feature__accel_jerk_abs_mean_m_s3",
    "feature__accel_jerk_abs_max_m_s3",
    "feature__rolling_accel_mean_1s_m_s2",
    "feature__rolling_accel_std_1s_m_s2",
    "feature__rolling_accel_max_1s_m_s2",
    "feature__timing_interval_ms_mean",
    "feature__timing_interval_ms_std",
    "feature__timing_interval_ms_max",
    "feature__accel_available_fraction",
    "feature__gyro_available_fraction",
)

OUTPUT_FIELDS = (
    "dataset_id",
    "subject_id",
    "session_id",
    "source_path",
    "source_path_resolution",
    "sample_index",
    "timestamp",
    "window_end_timestamp",
    "window_seconds",
    "activity_original",
    "source_activity_detail",
    "activity_safeband",
    "mapping_status",
    "hazard_label",
    "accel_original_unit",
    "accel_normalized_unit",
    "accel_conversion_applied",
    "gyro_original_unit",
    "gyro_normalized_unit",
    "gyro_conversion_applied",
    "timestamp_original_unit",
    "timestamp_normalized_unit",
    "timestamp_conversion_applied",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
    "accel_available",
    "gyro_available",
    "orientation_available",
    "timing_interval_ms",
    *MODEL_FEATURES,
)


def _compatibility_rows(raw_root: Path) -> list[dict[str, str]]:
    aicos_root = raw_root / "AICOS-HAR" / "AICOS-HAR"
    forth_root = raw_root / "FORTH_TRACE" / "FORTH_TRACE_DATASET-master"
    pamap_path = raw_root / "PAMAP2" / "dataset2.csv"
    bits_path = raw_root / "BITS-2" / "full_dataset.zip"

    forth_files = list(forth_root.glob("part*/part*dev1.csv"))
    pamap_labels = pamap2_adapter.observed_labels(pamap_path)
    with __import__("pandas").read_csv(
        pamap_path, usecols=["PeopleId"], chunksize=250_000
    ) as chunks:
        pamap_people: set[str] = set()
        for chunk in chunks:
            pamap_people.update(chunk["PeopleId"].dropna().astype(str).unique())

    aicos_labels = aicos_har_adapter.observed_labels(aicos_root / "metadata.csv")
    aicos_row = {
        "dataset_id": "aicos_har",
        "source_sensor": "Accelerometer stream; separate gyro streams are not merged because cross-stream synchronization is not established by this adapter.",
        "accelerometer_available": "yes",
        "gyroscope_available": "recorded separately; not used",
        "orientation_available": "no",
        "timestamp_available": "yes; timestamp column in each raw sensor file",
        "sampling_rate": "irregular raw timestamps; resampled at 50 Hz within contiguous segments",
        "accelerometer_units": "m/s^2 (local AICOS README)",
        "gyroscope_units": "rad/s (local AICOS README); not used",
        "coordinate_convention": "source axes preserved; no transformation documented",
        "sensor_placement": "heterogeneous; acquisition-specific position in metadata",
        "subject_count": "106 in local README",
        "session_count": "5,231 acquisitions in local README; acquisition ID is grouping key",
        "activity_label_quality": "activity and trial in acquisition path; labels: " + "; ".join(aicos_labels),
        "safe_band_compatibility": "MEDIUM",
        "compatibility_notes": "ACC SI units and ns timestamps are documented. Only rotation-invariant ACC magnitude/jerk and timing features enter models because device and placement axes vary. Gyro is intentionally not fused.",
        "decision": "INCLUDE",
    }

    sisfall_row = {
        "dataset_id": "sisfall",
        "source_sensor": "ADXL345 accelerometer counts selected from raw nine-column records",
        "accelerometer_available": "yes",
        "gyroscope_available": "not used; raw ITG3200 rotation counts lack verified angular unit conversion in the local docs",
        "orientation_available": "no",
        "timestamp_available": "sample index; elapsed time derived from documented 200 Hz fixed sampling rate",
        "sampling_rate": "200 Hz raw; block-averaged to 50 Hz",
        "accelerometer_units": "ADXL345 +/-16g, 13-bit counts -> g -> m/s^2 using documented calibration",
        "gyroscope_units": "not established; do not convert",
        "coordinate_convention": "source axes retained; only rotation-invariant magnitude-based features are used",
        "sensor_placement": "waist sensor location documented in archive README",
        "subject_count": "38 observed in repository audit",
        "session_count": "recording file path is session key",
        "activity_label_quality": "D01-D19 ADL and F01-F15 staged-fall descriptions in local archive README",
        "safe_band_compatibility": "MEDIUM",
        "compatibility_notes": "ADL codes map to activity classes; staged falls remain a separate hazard target. No raw-data changes.",
        "decision": "INCLUDE",
    }

    rows = [
        aicos_row,
        bits2_adapter.compatibility(),
        forth_trace_adapter.compatibility(len(forth_files)),
        pamap2_adapter.compatibility(len(pamap_people), pamap_labels),
        sisfall_row,
    ]
    if not bits_path.is_file():
        raise FileNotFoundError(f"BITS-2 source archive missing: {bits_path}")
    return rows


def _mapping_row(
    dataset: str,
    source_label: str,
    description: str,
    normalized: str,
    confidence: str,
    reason: str,
    sensor_basis: str,
    decision: str,
    notes: str = "",
) -> dict[str, str]:
    included_primary = decision == "INCLUDED" and normalized not in {
        "FALL_OR_HAZARD", "UNKNOWN", "UNMAPPED",
    }
    return {
        "source_dataset": dataset,
        "source_label": source_label,
        "source_label_description": description,
        "safeband_activity": normalized,
        "mapping_confidence": confidence,
        "mapping_reason": reason,
        "included_primary": "YES" if included_primary else "NO",
        "included_secondary": "YES" if decision in {"INCLUDED", "SECONDARY_ONLY"} else "NO",
        "sensor_basis": sensor_basis,
        "notes": notes,
        "decision": decision,
    }


def _aicos_mapping_rows(raw_root: Path) -> list[dict[str, str]]:
    labels = aicos_har_adapter.observed_labels(
        raw_root / "AICOS-HAR" / "AICOS-HAR" / "metadata.csv"
    )
    rows = []
    for label in labels:
        normalized, detail = safe_band_activity("aicos_har", label)
        decision = "INCLUDED" if normalized not in {"UNMAPPED", "UNKNOWN"} else "UNMAPPED"
        reason = (
            "Low-motion posture labels are combined into STATIC; original detail is retained."
            if normalized == "STATIC"
            else "Walking/ramp/stair locomotion is retained as WALKING."
            if normalized == "WALKING"
            else "Explicit running label."
            if normalized == "RUNNING"
            else "Elevator/lift travel is not a defensible IMU activity class in this taxonomy."
        )
        rows.append(_mapping_row(
            "aicos_har", label, label, normalized, "HIGH", reason,
            "timestamped ACC in documented m/s^2; rotation-invariant ACC features",
            decision,
            "Source label preserved in activity_original/source_activity_detail."
        ))
    return rows


def _sisfall_mapping_rows() -> list[dict[str, str]]:
    rows = []
    for label in sorted(SISFALL_LABEL_IDS):
        normalized, _ = safe_band_activity("sisfall", label)
        hazard = normalized == "FALL_OR_HAZARD"
        decision = "SECONDARY_ONLY" if hazard else "INCLUDED"
        reason = (
            "Staged fall scenario is a separate hazard task, not ordinary activity."
            if hazard
            else "Local SisFall README label description supports this coarse activity mapping."
        )
        rows.append(_mapping_row(
            "sisfall", label, SISFALL_DESCRIPTIONS[label], normalized, "HIGH",
            reason, "ADXL345 ACC counts calibrated to m/s^2; magnitude-based features",
            decision,
            "F codes are staged falls; do not interpret as prospective real-world fall incidence."
            if hazard else "Original D-code and full activity description are retained."
        ))
    return rows


def _rejected_mapping_rows(
    dataset: str,
    labels: Iterable[str],
    mapper,
    reason: str,
    sensor_basis: str,
) -> list[dict[str, str]]:
    rows = []
    for source_label in labels:
        normalized, description = mapper(source_label)
        rows.append(_mapping_row(
            dataset, source_label, description, normalized, "MEDIUM",
            "Semantic mapping is retained for audit, but the dataset is rejected from model input: "
            + reason,
            sensor_basis, "REJECTED",
            "No conversion or training use is performed."
        ))
    return rows


def _all_mapping_rows(raw_root: Path) -> list[dict[str, str]]:
    rows = _aicos_mapping_rows(raw_root)
    rows.extend(_sisfall_mapping_rows())

    bits_labels = bits2_adapter.observed_labels(raw_root / "BITS-2" / "full_dataset.zip")

    def bits_mapper(label: str) -> tuple[str, str]:
        if label in bits2_adapter.ADL_LABELS.values():
            lowered = label.lower()
            if lowered.startswith(("walking_", "climbing_")):
                return "WALKING", label
            if lowered == "jogging":
                return "RUNNING", label
            if lowered in {"jumping", "swinging_hands"}:
                return "HIGH_MOTION", label
            if lowered in {"lying_on_bed"}:
                return "STATIC", label
            if lowered in {
                "slowly_sitting_on_chair", "rapidly_sitting_on_chair",
                "sitting_getting_up", "lying_back_getting_up_slowly",
                "lying_back_getting_up_normally", "sideways_to_back_lying",
            }:
                return "TRANSITION", label
        return "FALL_OR_HAZARD", label

    rows.extend(_rejected_mapping_rows(
        "bits2", bits_labels, bits_mapper,
        "acceleration/gyro physical units, sampling rate, and synchronized stream timing are not established locally.",
        "source ACC/GYRO tokens; physical conversion unavailable"
    ))

    forth_root = raw_root / "FORTH_TRACE" / "FORTH_TRACE_DATASET-master"
    forth_labels = forth_trace_adapter.observed_labels(forth_root)

    def forth_mapper(label: str) -> tuple[str, str]:
        description, normalized = forth_trace_adapter.ACTIVITY_LABELS[int(label)]
        return normalized, description

    rows.extend(_rejected_mapping_rows(
        "forth_trace", forth_labels, forth_mapper,
        "accelerometer/gyro units and timestamp unit are not established in the local README.",
        "ACC/GYRO available; physical conversion unavailable"
    ))

    pamap_path = raw_root / "PAMAP2" / "dataset2.csv"
    pamap_labels = pamap2_adapter.observed_labels(pamap_path)

    def pamap_mapper(label: str) -> tuple[str, str]:
        normalized = pamap2_adapter.ACTIVITY_MAP.get(label.casefold(), "UNKNOWN")
        return normalized, label

    rows.extend(_rejected_mapping_rows(
        "pamap2", pamap_labels, pamap_mapper,
        "the merged CSV has no timestamps and local evidence does not establish exact physical units for the stored hand ACC/GYRO values.",
        "hand ACC/GYRO columns; verified normalization unavailable"
    ))
    return rows


def _resampled_aicos(recording: Recording) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    deltas = np.diff(recording.timestamps_s)
    median_period = float(np.median(deltas[deltas > 0]))
    max_gap = max(0.1, median_period * 5.0)
    for times, acceleration in contiguous_segments(
        recording.timestamps_s, recording.accel_xyz_m_s2, max_gap_s=max_gap
    ):
        grid, values = resample_segment(times, acceleration, TARGET_HZ)
        if len(grid) >= int(TARGET_HZ * min(WINDOW_SECONDS)):
            yield grid, values


def _sisfall_at_50_hz(recording: Recording) -> tuple[np.ndarray, np.ndarray]:
    size = len(recording.timestamps_s) // SISFALL_BLOCK_SIZE
    if size < 2:
        return np.empty(0), np.empty((0, 3))
    values = recording.accel_xyz_m_s2[: size * SISFALL_BLOCK_SIZE]
    downsampled = values.reshape(size, SISFALL_BLOCK_SIZE, 3).mean(axis=1)
    timestamps = np.arange(size, dtype=np.float64) / TARGET_HZ
    return timestamps, downsampled


def _selected_starts(length: int, window_samples: int) -> np.ndarray:
    stride = max(1, window_samples // 2)
    candidates = np.arange(0, length - window_samples + 1, stride, dtype=np.int64)
    if len(candidates) > MAX_WINDOWS_PER_SESSION:
        selection = np.linspace(0, len(candidates) - 1, MAX_WINDOWS_PER_SESSION, dtype=int)
        candidates = candidates[selection]
    return candidates


def _emit_recording(
    writer: csv.DictWriter,
    recording: Recording,
    timestamps: np.ndarray,
    acceleration: np.ndarray,
    counters: Counter,
) -> int:
    if len(timestamps) < 2:
        return 0
    normalized, detail = safe_band_activity(recording.dataset_id, recording.source_label)
    mapping_status = (
        "SECONDARY_ONLY" if normalized == "FALL_OR_HAZARD"
        else "UNMAPPED" if normalized in {"UNMAPPED", "UNKNOWN"}
        else "INCLUDED"
    )
    hazard = (
        1 if normalized == "FALL_OR_HAZARD"
        else 0 if recording.dataset_id == "sisfall"
        else ""
    )
    magnitude = np.linalg.norm(acceleration, axis=1)
    rolling_mean, rolling_std, rolling_max = rolling_magnitude_statistics(
        timestamps, magnitude
    )
    count = 0
    for duration in WINDOW_SECONDS:
        window_samples = int(round(TARGET_HZ * duration))
        for start in _selected_starts(len(timestamps), window_samples):
            end = int(start + window_samples)
            t = timestamps[start:end]
            acc = acceleration[start:end]
            if len(t) != window_samples or np.any(np.diff(t) <= 0):
                raise ValueError(f"{recording.session_id}: invalid window at {start}")
            features = window_features(
                t,
                acc,
                rolling_mean=rolling_mean[start:end],
                rolling_std=rolling_std[start:end],
                rolling_max=rolling_max[start:end],
            )
            row = {
                "dataset_id": recording.dataset_id,
                "subject_id": recording.subject_id,
                "session_id": recording.session_id,
                "source_path": recording.source_path,
                "source_path_resolution": recording.source_path_resolution,
                "sample_index": int(start),
                "timestamp": float(t[0]),
                "window_end_timestamp": float(t[-1]),
                "window_seconds": duration,
                "activity_original": recording.source_label,
                "source_activity_detail": detail,
                "activity_safeband": normalized,
                "mapping_status": mapping_status,
                "hazard_label": hazard,
                "accel_original_unit": recording.source_unit,
                "accel_normalized_unit": "m/s^2",
                "accel_conversion_applied": (
                    "none; source already documented in m/s^2"
                    if recording.dataset_id == "aicos_har"
                    else "ADXL345 counts to g using +/-16g/13-bit scale, then g to m/s^2"
                ),
                "gyro_original_unit": (
                    "rad/s (documented; separate stream unused)"
                    if recording.dataset_id == "aicos_har"
                    else "not established; sensor stream unused"
                ),
                "gyro_normalized_unit": "rad/s; unavailable and masked",
                "gyro_conversion_applied": "none; no gyroscope measurements enter this dataset",
                "timestamp_original_unit": recording.timestamp_source_unit,
                "timestamp_normalized_unit": (
                    "seconds; source time origin retained"
                    if recording.dataset_id == "aicos_har"
                    else "seconds from recording start"
                ),
                "timestamp_conversion_applied": (
                    "nanoseconds divided by 1e9; source origin retained"
                    if recording.dataset_id == "aicos_har"
                    else "sample index divided by documented 200 Hz; recording origin set to 0"
                ),
                "accel_x_m_s2": float(np.mean(acc[:, 0])),
                "accel_y_m_s2": float(np.mean(acc[:, 1])),
                "accel_z_m_s2": float(np.mean(acc[:, 2])),
                "gyro_x_rad_s": 0.0,
                "gyro_y_rad_s": 0.0,
                "gyro_z_rad_s": 0.0,
                "accel_available": 1,
                "gyro_available": 0,
                "orientation_available": 0,
                "timing_interval_ms": float(np.mean(np.diff(t)) * 1000.0),
                **features,
            }
            if set(row) != set(OUTPUT_FIELDS):
                raise AssertionError("Processed activity row does not match the declared schema")
            writer.writerow(row)
            counters[(recording.dataset_id, duration, normalized)] += 1
            count += 1
    return count


def main() -> None:
    global MAX_WINDOWS_PER_SESSION
    parser = argparse.ArgumentParser(description="Build SafeBand-compatible external activity windows.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--max-windows-per-session",
        type=int,
        default=MAX_WINDOWS_PER_SESSION,
    )
    args = parser.parse_args()
    if args.max_windows_per_session <= 0:
        parser.error("--max-windows-per-session must be positive")
    MAX_WINDOWS_PER_SESSION = args.max_windows_per_session

    raw_root = args.repo_root / "datasets" / "raw"
    output_root = args.repo_root / "datasets" / "external_processed" / "activity"
    mapping_root = args.repo_root / "tools" / "external_datasets" / "activity_mapping" / "mappings"
    report_root = args.repo_root / "reports" / "external_evaluation"
    for directory in (output_root, mapping_root, report_root):
        directory.mkdir(parents=True, exist_ok=True)

    compatibility_rows = _compatibility_rows(raw_root)
    compat_path = report_root / "activity_dataset_compatibility.csv"
    with compat_path.open("w", encoding="utf-8", newline="") as target:
        fields = list(compatibility_rows[0])
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(compatibility_rows)

    label_rows = _all_mapping_rows(raw_root)
    mapping_path = mapping_root / "activity_label_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as target:
        fields = list(label_rows[0])
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(label_rows)

    dataset_path = output_root / "activity_windows.csv.gz"
    counters: Counter = Counter()
    sessions: Counter = Counter()
    aicos_path_resolutions: Counter = Counter()
    source_labels: dict[str, set[str]] = {"aicos_har": set(), "sisfall": set()}
    written = 0
    with gzip.open(dataset_path, "wt", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        aicos_root = raw_root / "AICOS-HAR" / "AICOS-HAR"
        for recording in aicos_har_adapter.iter_recordings(
            aicos_root, aicos_root / "metadata.csv"
        ):
            aicos_path_resolutions[recording.source_path_resolution] += 1
            source_labels["aicos_har"].add(recording.source_label)
            session_count = 0
            for timestamps, acceleration in _resampled_aicos(recording):
                session_count += _emit_recording(
                    writer, recording, timestamps, acceleration, counters
                )
            if session_count:
                sessions[recording.dataset_id] += 1
                written += session_count

        for recording in sisfall_adapter.iter_recordings(raw_root / "SisFall" / "SisFall_dataset.zip"):
            source_labels["sisfall"].add(recording.source_label)
            timestamps, acceleration = _sisfall_at_50_hz(recording)
            session_count = _emit_recording(
                writer, recording, timestamps, acceleration, counters
            )
            if session_count:
                sessions[recording.dataset_id] += 1
                written += session_count

    report = {
        "evaluation_scope": "DEVELOPMENT / COMPONENT-LEVEL EVALUATION",
        "output": str(dataset_path.relative_to(args.repo_root)),
        "rows": int(written),
        "window_seconds": list(WINDOW_SECONDS),
        "target_rate_hz": TARGET_HZ,
        "max_windows_per_session_per_size": MAX_WINDOWS_PER_SESSION,
        "window_selection": "Evenly spaced deterministic windows; 50% overlap before the per-session cap.",
        "processed_datasets": ["aicos_har", "sisfall"],
        "aicos_acquisition_path_resolution_note": (
            "A missing metadata path is resolved only when its AppleiPhone_ placement "
            "has exactly one accelerometer file in the same participant/trial directory "
            "and that file is under the matching AppleiPhoneSE2_ placement folder."
        ),
        "rejected_datasets": {
            row["dataset_id"]: row["compatibility_notes"]
            for row in compatibility_rows
            if row["decision"] == "REJECTED"
        },
        "sessions_with_windows": dict(sessions),
        "aicos_acquisition_path_resolutions": dict(aicos_path_resolutions),
        "source_labels_observed_in_processed_data": {
            key: sorted(values) for key, values in source_labels.items()
        },
        "windows_per_dataset_window_size_class": {
            f"{dataset}|{window:.1f}s|{activity}": int(count)
            for (dataset, window, activity), count in sorted(counters.items())
        },
        "model_features": list(MODEL_FEATURES),
        "feature_boundary": "Only rotation-invariant ACC-derived and timing/availability features that can be computed from the SafeBand runtime feature representation are used. Gyro is absent in the model input.",
        "unit_conversions": {
            "aicos_har": {
                "acceleration_original_unit": "m/s^2",
                "acceleration_normalized_unit": "m/s^2",
                "acceleration_conversion_applied": "none; source already documented in m/s^2",
                "timestamp_original_unit": "ns",
                "timestamp_normalized_unit": "seconds; source origin retained",
                "timestamp_conversion_applied": "divide by 1e9; source origin retained",
            },
            "sisfall": {
                "acceleration_original_unit": "ADXL345 counts (+/-16g, 13-bit)",
                "acceleration_normalized_unit": "m/s^2",
                "acceleration_conversion_applied": "counts * 32/8192 g/count * 9.80665 m/s^2/g",
                "timestamp_original_unit": "sample index at documented 200 Hz",
                "timestamp_normalized_unit": "seconds",
                "timestamp_conversion_applied": "sample index / 200; downsample by 4 to 50 Hz",
            },
            "bits2": "Rejected; no unit conversion performed.",
            "forth_trace": "Rejected; no unit conversion performed.",
            "pamap2": "Rejected; no unit conversion performed.",
        },
        "source_files_modified": False,
        "safeband_real_recordings_modified": False,
        "lineage_2_12_modified": False,
        "external_sources_added_to_safeband_session_count": 0,
    }
    report_path = report_root / "activity_dataset_build.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
