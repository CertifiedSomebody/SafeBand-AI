# SafeBand PPG V6 FINAL

## Purpose
Final Phase-1 HR-estimation reference pipeline for the MAX30101-family PTT dataset, with explicit preparation for later MAX30102 validation.

## What is locked
- 8 s windows, 2 s shift.
- Subject-independent numeric 60/20/20 split.
- ECG-derived HR is the target; ECG waveform is never an input.
- SpO2 is not trained as a continuous target.
- 0.5–5 Hz PPG bandpass and multi-channel signal-derived features.
- Signal-quality and motion-contamination features are retained.
- Validation selects the feature group/model; test subjects remain untouched.
- Validation-only calibration is allowed only when it improves validation MAE.
- Raw and temporal results are both reported.
- Bootstrap 95% CIs and Bland–Altman agreement statistics are reported.

## Current reference result
The PTT/MAX30101 benchmark previously produced by V6.1 selected a Random Forest over the candidate-quality feature group (97 features) and achieved 1.438 BPM held-out raw MAE, 2.960 BPM RMSE, R² 0.899, 94.795% within ±5 BPM and 97.702% within ±10 BPM on the locked test subjects.

Those values are a MAX30101-family reference result, not MAX30102 production validation.

## MAX30102 gate
The final promotion gate is:
1. Audit the available BITS2 MAX30102-domain evidence.
2. If raw MAX30102 RED/IR plus defensible reference HR are available, run a completely separate cross-device test without retraining on the test subjects.
3. If raw/reference pairing is unavailable, do not manufacture a transfer result. Proceed to controlled SafeBand MAX30102 acquisition.
4. Report device settings, placement, contact, environment and signal-quality exclusions.
5. Only after real MAX30102 recordings are evaluated should the artifact be described as MAX30102-validated.

## Research reporting
Do not call HR regression 'classification accuracy'. Report MAE, RMSE, R², tolerance percentages, bias, Bland–Altman limits of agreement, and subject/activity breakdowns.
