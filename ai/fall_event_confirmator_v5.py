"""
SafeBand AI — Fall event confirmator v5.

Temporal confirmation uses a cluster of candidate windows rather than a single
window. The rule is parameterized so the offline evaluator can select an
operating point on validation data before test evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class FallEventState:
    active: bool = False
    peak_probability: float = 0.0
    hit_count: int = 0
    last_hit_sample: Optional[int] = None
    hit_samples: List[int] = field(default_factory=list)


class FallEventConfirmatorV5:
    """Streaming implementation of a selected v5 temporal rule.

    A candidate hit is p >= hit_threshold. Hits separated by more than
    max_gap_samples start a new cluster. A cluster confirms when it contains
    at least min_hits and its peak probability is >= peak_threshold.
    """

    def __init__(self, hit_threshold: float = 0.55, min_hits: int = 3,
                 peak_threshold: float = 0.65, max_gap_samples: int = 30):
        self.hit_threshold = float(hit_threshold)
        self.min_hits = int(min_hits)
        self.peak_threshold = float(peak_threshold)
        self.max_gap_samples = int(max_gap_samples)
        self.state = FallEventState()

    def update(self, probability: float, sample_index: int) -> Dict[str, object]:
        p = float(probability)
        i = int(sample_index)

        if p >= self.hit_threshold:
            if (self.state.last_hit_sample is None or
                    i - self.state.last_hit_sample <= self.max_gap_samples):
                self.state.hit_count += 1
            else:
                self._reset_cluster()
                self.state.hit_count = 1

            self.state.peak_probability = max(self.state.peak_probability, p)
            self.state.last_hit_sample = i
            self.state.hit_samples.append(i)

        elif (self.state.last_hit_sample is not None and
              i - self.state.last_hit_sample > self.max_gap_samples):
            self._reset_cluster()

        self.state.active = (
            self.state.hit_count >= self.min_hits and
            self.state.peak_probability >= self.peak_threshold
        )
        return {
            "fall_event": self.state.active,
            "peak_probability": self.state.peak_probability,
            "hit_count": self.state.hit_count,
            "last_hit_sample": self.state.last_hit_sample,
        }

    def _reset_cluster(self) -> None:
        self.state = FallEventState()

    def reset(self) -> None:
        self._reset_cluster()
