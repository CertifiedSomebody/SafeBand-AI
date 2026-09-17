"""SafeBand AI repository/path helpers used by training tools."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def discover_airwise_dir(root: Path = ROOT) -> Path:
    """Find AIRWISE minute-average directory without assuming archive nesting."""
    base = root / "datasets" / "raw" / "AIRWISE"
    candidates = [
        base / "data" / "indoor" / "sensor_1min",
        base / "AIRWISE-v1.0.0" / "data" / "indoor" / "sensor_1min",
    ]
    candidates.extend(
        p.parent for p in base.rglob("*_minute_averages.csv")
        if p.is_file()
    )
    seen = set()
    valid = []
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        if p.is_dir() and list(p.glob("*_minute_averages.csv")):
            valid.append(p)
    if not valid:
        raise FileNotFoundError(
            f"No AIRWISE *_minute_averages.csv files found below {base}. "
            "Pass --data-dir if the dataset is stored elsewhere."
        )
    # Prefer the directory containing the most minute-average files.
    return max(valid, key=lambda p: len(list(p.glob("*_minute_averages.csv"))))

def discover_airwise_files(data_dir: Path | None = None) -> List[Path]:
    directory = Path(data_dir) if data_dir else discover_airwise_dir()
    paths = sorted(directory.glob("*_minute_averages.csv"))
    if not paths:
        raise FileNotFoundError(f"No AIRWISE minute files found under {directory}")
    return paths

def ensure_repo_root() -> Path:
    return ROOT
