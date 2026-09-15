"""SafeBand temporal fall-event confirmation engine.

The engine converts per-window fall probabilities into confirmed events.
It is deliberately stateful and conservative: a single high-probability
window is not enough unless min_hits is configured to 1.

This module contains no SMS/GPS/network side effects. Integrate it with the
Safety Engine only after offline and hardware validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class FallEventState:
    hit_count: int = 0
    last_hit_sample: Optional[int] = None
    peak_probability: float = 0.0
    active: bool = False
    cooldown_until_sample: Optional[int] = None


class FallEventConfirmator:
    """Confirm fall events from a stream of window probabilities.

    Parameters
    ----------
    threshold:
        Probability required for a window to count as a temporal hit.
    min_hits:
        Number of hits in one temporal cluster required for confirmation.
    max_gap_samples:
        Maximum distance between hit-window start indices within a cluster.
    cooldown_samples:
        After confirmation, suppress additional confirmations until this
        many samples have elapsed. This prevents repeated alerts for one
        physical fall.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_hits: int = 2,
        max_gap_samples: int = 30,
        cooldown_samples: int = 600,
    ) -> None:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if int(min_hits) < 1:
            raise ValueError("min_hits must be >= 1")
        if int(max_gap_samples) < 0:
            raise ValueError("max_gap_samples must be >= 0")
        if int(cooldown_samples) < 0:
            raise ValueError("cooldown_samples must be >= 0")

        self.threshold = float(threshold)
        self.min_hits = int(min_hits)
        self.max_gap_samples = int(max_gap_samples)
        self.cooldown_samples = int(cooldown_samples)
        self.state = FallEventState()

    def reset(self) -> None:
        """Clear the current temporal cluster and active state."""
        self.state = FallEventState()

    def update(self, probability: float, sample_index: int) -> Dict[str, Any]:
        """Consume one probability observation.

        Returns a dictionary containing the current state and an
        ``event_confirmed`` flag that is True only on the confirmation step.
        """
        p = float(probability)
        sample = int(sample_index)
        if not 0.0 <= p <= 1.0:
            raise ValueError("probability must be in [0, 1]")

        # A new stream observation should never move backwards in sample time.
        if self.state.last_hit_sample is not None and sample < self.state.last_hit_sample:
            raise ValueError("sample_index must be non-decreasing")

        if self.state.cooldown_until_sample is not None:
            if sample < self.state.cooldown_until_sample:
                return self._result(event_confirmed=False)
            self.state.cooldown_until_sample = None
            self.state.active = False
            self.state.hit_count = 0
            self.state.last_hit_sample = None
            self.state.peak_probability = 0.0

        hit = p >= self.threshold
        event_confirmed = False

        if hit:
            if (
                self.state.last_hit_sample is None
                or sample - self.state.last_hit_sample <= self.max_gap_samples
            ):
                self.state.hit_count += 1
            else:
                self.state.hit_count = 1
                self.state.peak_probability = 0.0

            self.state.last_hit_sample = sample
            self.state.peak_probability = max(self.state.peak_probability, p)

            if self.state.hit_count >= self.min_hits and not self.state.active:
                self.state.active = True
                event_confirmed = True
                if self.cooldown_samples > 0:
                    self.state.cooldown_until_sample = sample + self.cooldown_samples

        elif (
            self.state.last_hit_sample is not None
            and sample - self.state.last_hit_sample > self.max_gap_samples
        ):
            self.state.hit_count = 0
            self.state.last_hit_sample = None
            self.state.peak_probability = 0.0
            self.state.active = False

        return self._result(event_confirmed=event_confirmed)

    def _result(self, event_confirmed: bool) -> Dict[str, Any]:
        return {
            "fall_event": bool(self.state.active),
            "event_confirmed": bool(event_confirmed),
            "peak_probability": float(self.state.peak_probability),
            "hit_count": int(self.state.hit_count),
            "last_hit_sample": self.state.last_hit_sample,
            "cooldown_until_sample": self.state.cooldown_until_sample,
        }
