"""
SAFEBAND AI - Sensor Window Buffer

Maintains a short, ordered sequence of sensor samples for ML
inference. The buffer is intentionally independent of Streamlit,
simulation and hardware code.
"""

from collections import deque
from typing import Any, Deque, Dict, List


class SensorWindowBuffer:
    """Fixed-size FIFO buffer for time-series sensor samples."""

    def __init__(self, max_samples: int) -> None:
        if int(max_samples) < 1:
            raise ValueError("max_samples must be >= 1")
        self.max_samples = int(max_samples)
        self._samples: Deque[Dict[str, Any]] = deque(
            maxlen=self.max_samples
        )

    def append(self, sample: Dict[str, Any]) -> None:
        if isinstance(sample, dict):
            self._samples.append(dict(sample))

    def extend(self, samples: List[Dict[str, Any]]) -> None:
        for sample in samples:
            self.append(sample)

    def get(self) -> List[Dict[str, Any]]:
        return [dict(sample) for sample in self._samples]

    def __len__(self) -> int:
        return len(self._samples)

    def is_ready(self, minimum_samples: int) -> bool:
        return len(self) >= int(minimum_samples)

    def clear(self) -> None:
        self._samples.clear()


__all__ = ["SensorWindowBuffer"]
