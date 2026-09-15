"""SafeBand AI — V7 fall event confirmation.

V7 converts ordered per-window fall probabilities into one recording-level
candidate event representation. It deliberately does not train or select on
the held-out test set.
"""
from __future__ import annotations

import math
from typing import Sequence

V7_FEATURES = [
    "p_max", "p_mean", "p_std", "p_median", "p_p90",
    "p_top3_mean", "p_top5_mean", "p_above_50", "p_above_60", "p_above_70",
    "max_run_50", "max_run_60", "max_run_70",
    "evidence_sum_50", "evidence_sum_60", "evidence_sum_70",
    "trajectory_range", "rise_to_peak", "fall_from_peak", "peak_index_ratio",
    "pre_peak_mean", "post_peak_mean", "pre_post_delta",
    "peak_width_50", "peak_width_70", "active_span_samples",
    "num_candidate_runs_50", "num_candidate_runs_70",
    "trajectory_slope", "trajectory_abs_change_mean",
]


def _mean(x: Sequence[float]) -> float:
    return float(sum(x) / len(x)) if x else 0.0


def _std(x: Sequence[float]) -> float:
    if not x:
        return 0.0
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / len(x))


def _percentile(x: Sequence[float], q: float) -> float:
    if not x:
        return 0.0
    a = sorted(float(v) for v in x)
    pos = (len(a) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (pos - lo)


def _max_run(x: Sequence[float], threshold: float) -> int:
    best = run = 0
    for v in x:
        if v >= threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _run_count(x: Sequence[float], threshold: float) -> int:
    count = 0
    active = False
    for v in x:
        now = v >= threshold
        if now and not active:
            count += 1
        active = now
    return count


def _width(x: Sequence[float], threshold: float) -> int:
    return int(sum(v >= threshold for v in x))


def extract_event_features(probabilities: Sequence[float], starts: Sequence[int] | None = None) -> dict[str, float]:
    """Extract compact trajectory/evidence features from ordered window probabilities."""
    p = [float(v) for v in probabilities]
    if not p:
        return {k: 0.0 for k in V7_FEATURES}

    n = len(p)
    peak = max(p)
    pi = p.index(peak)
    top = sorted(p, reverse=True)
    top3 = top[:3]
    top5 = top[:5]
    pre = p[:pi] or [p[0]]
    post = p[pi + 1:] or [p[-1]]

    diffs = [p[i] - p[i - 1] for i in range(1, n)]
    # A simple least-squares slope over the probability trajectory.
    if n > 1:
        xbar = (n - 1) / 2.0
        ybar = _mean(p)
        den = sum((i - xbar) ** 2 for i in range(n))
        slope = sum((i - xbar) * (p[i] - ybar) for i in range(n)) / den if den else 0.0
    else:
        slope = 0.0

    if starts is not None and len(starts) == n:
        s = [int(v) for v in starts]
        active = [s[i] for i, v in enumerate(p) if v >= 0.50]
        span = float(max(active) - min(active)) if len(active) >= 2 else 0.0
    else:
        span = float(_width(p, 0.50))

    return {
        "p_max": peak,
        "p_mean": _mean(p),
        "p_std": _std(p),
        "p_median": _percentile(p, 0.50),
        "p_p90": _percentile(p, 0.90),
        "p_top3_mean": _mean(top3),
        "p_top5_mean": _mean(top5),
        "p_above_50": _mean([1.0 if v >= 0.50 else 0.0 for v in p]),
        "p_above_60": _mean([1.0 if v >= 0.60 else 0.0 for v in p]),
        "p_above_70": _mean([1.0 if v >= 0.70 else 0.0 for v in p]),
        "max_run_50": float(_max_run(p, 0.50)),
        "max_run_60": float(_max_run(p, 0.60)),
        "max_run_70": float(_max_run(p, 0.70)),
        "evidence_sum_50": float(sum(max(0.0, v - 0.50) for v in p)),
        "evidence_sum_60": float(sum(max(0.0, v - 0.60) for v in p)),
        "evidence_sum_70": float(sum(max(0.0, v - 0.70) for v in p)),
        "trajectory_range": float(max(p) - min(p)),
        "rise_to_peak": float(peak - p[0]),
        "fall_from_peak": float(peak - p[-1]),
        "peak_index_ratio": float(pi / max(1, n - 1)),
        "pre_peak_mean": _mean(pre),
        "post_peak_mean": _mean(post),
        "pre_post_delta": float(abs(_mean(pre) - _mean(post))),
        "peak_width_50": float(_width(p, 0.50)),
        "peak_width_70": float(_width(p, 0.70)),
        "active_span_samples": span,
        "num_candidate_runs_50": float(_run_count(p, 0.50)),
        "num_candidate_runs_70": float(_run_count(p, 0.70)),
        "trajectory_slope": float(slope),
        "trajectory_abs_change_mean": _mean([abs(v) for v in diffs]),
    }
