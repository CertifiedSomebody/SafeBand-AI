"""Feature contract, regression, and temporal physiological plausibility."""
from __future__ import annotations
from collections import deque
import numpy as np
from .hr_signal_pipeline import extract_candidates
from .motion_quality import extract_motion_quality


def make_features(bvp, acc, bvp_fs, acc_fs):
    c = extract_candidates(bvp, bvp_fs)
    m = extract_motion_quality(acc, acc_fs)
    filtered = c.pop("filtered")
    c.update(m)

    # Compact raw statistics remain useful, but normalized to avoid optical
    # amplitude/domain leakage.
    dx = np.diff(filtered)
    c.update({
        "ppg_rms": float(np.sqrt(np.mean(filtered**2))),
        "ppg_std": float(np.std(filtered)),
        "ppg_ptp": float(np.ptp(filtered)),
        "ppg_diff_rms": float(np.sqrt(np.mean(dx*dx))),
        "ppg_skew": float(np.mean(filtered**3)),
        "ppg_kurt": float(np.mean(filtered**4)),
    })
    return c


class PhysiologicalTracker:
    """Streaming HR stabilizer.

    It rejects implausible jumps and uses quality-weighted candidate consensus.
    It does not replace the learned estimator; it is a safety/continuity layer.
    """

    def __init__(self, min_hr=42.0, max_hr=180.0,
                 max_step_bpm=25.0, history=5):
        self.min_hr = float(min_hr)
        self.max_hr = float(max_hr)
        self.max_step_bpm = float(max_step_bpm)
        self.history = deque(maxlen=history)
        self.last = None

    def update(self, predicted_hr, candidate_hr=np.nan, quality=0.0):
        p = float(predicted_hr)
        if not np.isfinite(p):
            return self.last

        p = float(np.clip(p, self.min_hr, self.max_hr))
        if self.last is not None and abs(p-self.last) > self.max_step_bpm:
            # A large jump is allowed only when candidate evidence agrees.
            if not np.isfinite(candidate_hr) or abs(candidate_hr-self.last) > self.max_step_bpm:
                p = self.last

        if np.isfinite(candidate_hr) and 0 <= quality <= 1:
            # Small, quality-dependent correction toward the independent signal
            # estimator; this is intentionally conservative.
            w = 0.15 * float(quality)
            p = (1-w)*p + w*float(np.clip(candidate_hr, self.min_hr, self.max_hr))

        self.history.append(p)
        # Median is robust to one bad optical window.
        self.last = float(np.median(self.history))
        return self.last
