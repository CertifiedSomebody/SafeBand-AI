# AICOS-HAR Activity Recognition — Pipeline V3

## Status before V3

The original forensic audit established 5,231 ACC files and 18,587 total
sensor files. ACC median sampling rate was about 397 Hz, gyro about 397 Hz,
magnetometer about 100 Hz and barometer about 10 Hz. The dataset contains
106 participants and official folds.

A V2 path scanner falsely found zero sensor files because it assumed short
sensor names (`acc`, `gyro`, `mag`, `baro`). AICOS-HAR uses documented sensor
files such as accelerometer/gyroscope/magnetometer/barometer. V3 fixes this
parser and does not alter raw data.

## V3 protocol

### Stage 1 — path reconciliation
Run `audit_aicos_har_path_v3.py`.
Expected result should be consistent with the forensic audit:
- 5,231 ACC acquisitions
- 18,587 sensor files total
- no unexplained bulk path mismatch.

### Stage 2 — bounded sampling audit
Run `audit_aicos_har_sampling_v2.py --max-acquisitions 40`.
This checks real timestamps before choosing a resampling rate.

### Stage 3 — canonical windowing
The included builder defaults to:
- ACC only
- 50 Hz
- 2 s window
- 1 s step
- hard acquisition boundaries
- official Fold retained.

This is a baseline, not a claim that 50 Hz is optimal.

### Stage 4 — classical benchmark
`train_aicos_har_classical_v1.py`
- official `Fold=test` is untouched during training
- training uses folds 1–5
- 38-feature SafeBand ACC contract
- ExtraTrees, RandomForest, HGB, RBF-SVM
- metrics: accuracy, balanced accuracy, macro-F1, confusion matrix.

### Stage 5 — temporal DL
`train_aicos_har_tinycnn_v1.py`
- raw ACC windows
- training-only normalization
- class-balanced cross entropy
- compact 1D CNN
- official test fold evaluated once after training.

## Important methodological constraint

The first AICOS benchmark should be ACC-only because this directly tests whether
AICOS-HAR improves the existing SafeBand ACC activity branch. Gyro can be added
as a second controlled ablation after the ACC baseline.

The dataset contains 11 activity labels in the audit:
Running, Walking, Downstairs, LyingDown, Sitting, Standing, Upstairs,
LiftDown, LiftUp, RampDown, RampUp.

Do not merge these labels before the first benchmark. Keep the original labels
for the forensic benchmark; SafeBand semantic mapping can be a later stage.

## Reproducibility

- Raw data immutable.
- Derived windows under `datasets/processed/aicos_har`.
- Models under `models/aicos_har`.
- Reports under `reports/aicos_har_*`.
- Scripts run from repository root.
- Test fold is not used for tuning.
- Every benchmark receives a new numbered document; previous results are not overwritten.
