# Multi-Dataset V4 protocol

V4 is the temporal-event stage after V3 augmentation benchmarking. V3 showed that raw-signal augmentation did not improve the frozen window benchmark, so V4 does not add another augmentation recipe.

## Primary metrics

Report separately for BITS-2 and SisFall:

- recording-level fall recall
- recording-level fall precision
- event F1
- false-positive rate
- false alarms
- missed falls
- confusion matrix

The pooled result is secondary. Dataset-macro metrics are preferred when comparing the two domains.

## Operating-point policy

The validation selector first looks for operating points with at least 90% recall on both datasets, then minimizes macro FPR and maximizes macro F1. If no such point exists, it maximizes the minimum dataset recall, then macro F1, then minimizes macro FPR.

This is deliberately a validation-only rule. The held-out test subjects are never used to choose threshold, hit count, or temporal gap.
