# PPG V4.2 Experiment Log

## Status

V4.2 is a diagnostic release. It intentionally does not claim an improvement
until the new run is completed.

## Historical V4.1

V4.1:
- 15,982 windows
- 22 subjects
- 66 records
- 159 features
- numeric 60/20/20 subject split
- HGB selected on validation MAE
- held-out test MAE 4.576 BPM
- held-out test RMSE 5.955 BPM
- held-out test R² 0.591
- ±5 BPM 69.57%
- ±10 BPM 92.47%
- bias −4.41 BPM

Walking was the main aggregate weakness and subjects s19/s20 contained
particularly difficult walking records.

## V4.2 research question

Before changing the model:

1. Does ML materially beat the constant train-mean baseline?
2. Which of Ridge/RF/Extra Trees/HGB generalizes best on validation?
3. Does calibration improve validation MAE?
4. Does the held-out test preserve the validation advantage?
5. Which activities and subjects dominate error?
6. Which features actually matter on validation data?

## Anti-leakage rules

- Subject-level split only.
- Validation used for model selection.
- Calibration uses validation only.
- Test labels are not used for model selection.
- Feature importance uses validation only.
- ECG waveform is not a feature.
- SpO2 start/end values are not copied into continuous windows.

## Interpretation principle

A lower test MAE is useful only if it comes from the same predefined
evaluation protocol. We will not cherry-pick subjects, activities, or metrics.
