"""
SafeBand AI — Event-confirmed fall runtime logic v4.

This layer is intentionally conservative: a single high-probability window
does not have to trigger a fall event. It confirms clusters of high-risk
windows. Integrate with SafeBand alert escalation only after offline and
hardware validation.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class FallEventState:
    active: bool=False
    peak_probability: float=0.0
    hit_count: int=0
    last_hit_sample: Optional[int]=None

class FallEventConfirmator:
    def __init__(self, threshold=.5, min_hits=2, max_gap_samples=30):
        self.threshold=threshold
        self.min_hits=min_hits
        self.max_gap_samples=max_gap_samples
        self.state=FallEventState()

    def update(self, probability:float, sample_index:int):
        hit=probability>=self.threshold
        if hit:
            if (self.state.last_hit_sample is None or
                sample_index-self.state.last_hit_sample<=self.max_gap_samples):
                self.state.hit_count+=1
            else:
                self.state.hit_count=1
            self.state.peak_probability=max(self.state.peak_probability,probability)
            self.state.last_hit_sample=sample_index
        elif (self.state.last_hit_sample is not None and
              sample_index-self.state.last_hit_sample>self.max_gap_samples):
            self.reset()
        if self.state.hit_count>=self.min_hits:
            self.state.active=True
        return {
            "fall_event":self.state.active,
            "peak_probability":self.state.peak_probability,
            "hit_count":self.state.hit_count,
        }

    def reset(self):
        self.state=FallEventState()
