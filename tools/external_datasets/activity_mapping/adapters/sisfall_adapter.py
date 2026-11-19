from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .common import Recording, validate_recording


FILE_PATTERN = re.compile(r"^(D|F)(\d{2})_(SA|SE)(\d{2})_R(\d{2})\.txt$", re.I)
SAMPLE_RATE_HZ = 200.0
G_PER_COUNT_ADXL345 = (2.0 * 16.0) / (2**13)
STANDARD_GRAVITY_M_S2 = 9.80665


def _parse_recording(payload: bytes, member: str) -> np.ndarray:
    normalized = payload.replace(b";\r\n", b",\n").replace(b";\n", b",\n")
    normalized = normalized.rstrip(b",\r\n")
    values = np.fromstring(normalized, sep=",", dtype=np.int32)
    remainder = values.size % 9
    if remainder not in (0, 1):
        raise ValueError(f"{member}: parsed value count has unexpected remainder {remainder}")
    if remainder:
        values = values[:-remainder]
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    rows = values.reshape(-1, 9)
    return rows[:, :3].astype(np.float64) * G_PER_COUNT_ADXL345 * STANDARD_GRAVITY_M_S2


def iter_recordings(archive_path: Path) -> Iterator[Recording]:
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if not name.startswith("__MACOSX/")
            and name.lower().endswith(".txt")
            and FILE_PATTERN.fullmatch(Path(name).name)
        )
        for member in members:
            match = FILE_PATTERN.fullmatch(Path(member).name)
            if match is None:
                continue
            kind, code, subject_kind, subject_number, trial = match.groups()
            acceleration = _parse_recording(archive.read(member), member)
            if len(acceleration) < 2:
                continue
            timestamps = np.arange(len(acceleration), dtype=np.float64) / SAMPLE_RATE_HZ
            recording = Recording(
                dataset_id="sisfall",
                subject_id=f"{subject_kind.upper()}{subject_number}",
                session_id=member,
                source_label=f"{kind.upper()}{code}",
                timestamps_s=timestamps,
                accel_xyz_m_s2=acceleration,
                sample_rate_hz=SAMPLE_RATE_HZ,
                source_unit=(
                    "ADXL345 counts -> g using documented +/-16g, 13-bit scale "
                    "-> m/s^2"
                ),
                timestamp_source_unit="sample index / documented 200 Hz",
                source_path=f"{archive_path.as_posix()}!{member}",
            )
            validate_recording(recording)
            yield recording


def observed_labels(archive_path: Path) -> list[str]:
    labels: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            match = FILE_PATTERN.fullmatch(Path(member).name)
            if match and not member.startswith("__MACOSX/"):
                kind, code, _, _, _ = match.groups()
                labels.add(f"{kind.upper()}{code}")
    return sorted(labels)
