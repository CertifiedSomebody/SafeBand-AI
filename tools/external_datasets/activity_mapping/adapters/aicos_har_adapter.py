from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

import numpy as np

from .common import Recording, validate_recording


def _raw_acquisition_id(acquisition_id: str) -> str:
    parts = acquisition_id.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid AICOS acquisition ID: {acquisition_id}")
    activity_trial = parts[1]
    match = re.fullmatch(r"(.+)_([0-9]+)", activity_trial)
    if match and match.group(1) == "LiftDown":
        parts[1] = f"ElevatorDown_{match.group(2)}"
    elif match and match.group(1) == "LiftUp":
        parts[1] = f"ElevatorUp_{match.group(2)}"
    return "/".join(parts)


def _read_accelerometer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", errors="strict") as source:
        first_line = next((line.strip() for line in source if line.strip()), None)
    if first_line is None:
        return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    delimiter = "|" if "|" in first_line else ","
    try:
        values = np.loadtxt(
            path,
            delimiter=delimiter,
            dtype=np.float64,
            ndmin=2,
            encoding="utf-8-sig",
        )
    except ValueError as exc:
        raise ValueError(f"Unable to parse AICOS accelerometer file {path}: {exc}") from exc
    if values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected 4 finite numeric columns")

    timestamps = values[:, 0] / 1e9
    acceleration = values[:, 1:]
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    acceleration = acceleration[order]
    unique = np.r_[True, np.diff(timestamps) > 0]
    timestamps = timestamps[unique]
    acceleration = acceleration[unique]
    return timestamps, acceleration


def iter_recordings(
    raw_root: Path,
    metadata_path: Path,
) -> Iterator[Recording]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source:
        metadata = list(csv.DictReader(source))
    for row in metadata:
        acquisition_id = row["AcquisitionID"].strip()
        raw_id = _raw_acquisition_id(acquisition_id)
        path = raw_root.joinpath(*raw_id.split("/"), "Accelerometer.txt")
        path_resolution = "direct"
        if not path.is_file():
            device_folder = path.parent.name
            placement = device_folder.removeprefix("AppleiPhone_")
            alternate = path.parent.parent / f"AppleiPhoneSE2_{placement}" / "Accelerometer.txt"
            candidates = (
                [
                    folder / "Accelerometer.txt"
                    for folder in path.parent.parent.iterdir()
                    if folder.is_dir()
                    and folder.name.endswith(f"_{placement}")
                    and (folder / "Accelerometer.txt").is_file()
                ]
                if path.parent.parent.is_dir()
                else []
            )
            if (
                not device_folder.startswith("AppleiPhone_")
                or not placement
                or candidates != [alternate]
            ):
                raise FileNotFoundError(f"AICOS accelerometer is missing: {path}")
            path = alternate
            path_resolution = "unique_se2_same_placement_fallback"
        timestamps, acceleration = _read_accelerometer(path)
        if len(timestamps) < 2:
            continue
        parts = acquisition_id.replace("\\", "/").split("/")
        label_match = re.fullmatch(r"(.+)_([0-9]+)", parts[1])
        if not label_match:
            raise ValueError(f"Cannot resolve AICOS activity label: {acquisition_id}")
        label = label_match.group(1)
        delta = np.diff(timestamps)
        observed_rate = 1.0 / float(np.median(delta[delta > 0]))
        recording = Recording(
            dataset_id="aicos_har",
            subject_id=parts[0],
            session_id=acquisition_id,
            source_label=label,
            timestamps_s=timestamps,
            accel_xyz_m_s2=acceleration,
            sample_rate_hz=observed_rate,
            source_unit="m/s^2 (AICOS README)",
            timestamp_source_unit="ns (AICOS README)",
            source_path=path.as_posix(),
            source_path_resolution=path_resolution,
        )
        validate_recording(recording)
        yield recording


def observed_labels(metadata_path: Path) -> list[str]:
    labels: set[str] = set()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            parts = row["AcquisitionID"].strip().replace("\\", "/").split("/")
            match = re.fullmatch(r"(.+)_([0-9]+)", parts[1])
            if not match:
                raise ValueError(f"Cannot resolve AICOS activity label: {row['AcquisitionID']}")
            labels.add(match.group(1))
    return sorted(labels)
