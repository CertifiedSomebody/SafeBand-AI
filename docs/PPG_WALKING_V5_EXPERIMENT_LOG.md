# PPG Walking V5 Experiment Log

## V4.3 baseline problem

The frozen V4.3 PTT/MAX30101-domain benchmark reported held-out walking MAE of 6.497 BPM, ±5 BPM coverage of 53.17%, ±10 BPM coverage of 80.92%, and bias of -6.42 BPM. Walking was the worst activity.

## V5 design decision

Use an independent walking-focused wrist-PPG dataset and preserve the established methodological guardrails:

1. 8 s windows.
2. 2 s shift.
3. Subject-independent split.
4. ECG-derived HR target.
5. No ECG waveform features.
6. Validation-only model selection.
7. Validation-only affine calibration.
8. Untouched test subjects.
9. Report MAE/RMSE/R², ±3/±5/±10 BPM, bias, median/P90/P95/max absolute error.

## External dataset

PhysioNet Wrist PPG During Exercise v1.0.0. The database provides wrist PPG, chest ECG, gyro, two accelerometer ranges and magnetometer at 256 Hz, with hand-identified ECG R peaks supplied as reference annotations. citeturn0search0

## Current status

- Raw acquisition: complete.
- Raw structure: verified.
- Walking records present: s1, s2, s3, s6, s8, s9.
- Preparation code: ready.
- Training/evaluation: pending execution in the user's SafeBand environment.

## Do not claim yet

No V5 improvement should be claimed until the generated report demonstrates improvement on an untouched walking test set.
