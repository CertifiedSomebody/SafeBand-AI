# PAMAP2 Stationary V1

This experiment is intentionally separate from the BITS2 activity benchmark.

Target classes:
- LYING
- SITTING
- STANDING

Input:
- hand/wrist accelerometer X/Y/Z
- hand/wrist gyroscope X/Y/Z

Evaluation:
- subject-independent 5-fold StratifiedGroupKFold
- normalization fitted on training subjects only
- inner grouped validation for early stopping
- untouched outer test subjects

The first step is the audit. Do not train until the audit completes and the detected schema looks correct.
