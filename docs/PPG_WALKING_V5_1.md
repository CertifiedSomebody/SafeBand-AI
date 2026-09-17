# SafeBand PPG Walking V5.1

## Objective

V5.1 is a targeted experiment to address the dominant weakness identified in V4.3: held-out walking HR estimation.

V4.3 walking reference:
- MAE: 6.497 BPM
- within ±5 BPM: 53.17%
- within ±10 BPM: 80.92%
- bias: -6.424 BPM
- R²: -0.884

These values are the benchmark context only. V5.1 uses a different external walking dataset and therefore is not an apples-to-apples replacement for the V4.3 PTT result.

## Dataset

PhysioNet Wrist PPG During Exercise, v1.0.0.

Walking records available locally:
S1, S2, S3, S6, S8, S9.

Native signals:
- wrist PPG
- wrist gyroscope x/y/z
- wrist low-noise accelerometer x/y/z
- wrist wide-range accelerometer x/y/z
- wrist magnetometer x/y/z
- chest ECG
- timing channel

V5.1 deliberately retains the low-noise accelerometer + gyroscope as the common motion representation. Wide-range accelerometer and magnetometer are retained in raw data for later experiments rather than silently adding extra variables.

## Established Parameters

- window: 8 s
- shift: 2 s
- sampling rate: 256 Hz
- PPG band: 0.5–5 Hz
- ECG R peaks: HR target only
- ECG waveform: never a feature
- per-window robust normalization
- validation MAE for model selection
- validation-only affine calibration
- untouched subject-level test

## Controlled Feature Ablation

V5.1 compares:
1. PPG-only
2. PPG + gyroscope
3. PPG + low-noise accelerometer + gyroscope

Each representation is compared with:
- Ridge
- Random Forest
- Extra Trees
- HistGradientBoosting

The best representation/model pair is selected using validation MAE.

## Split

The established numeric 60/20/20 subject protocol is retained.

With six subjects this produces:
- 4 training subjects
- 1 validation subject
- 1 untouched test subject

This is intentionally documented because the small subject count makes the final result a stress-test experiment rather than a population-level performance estimate.

## Outputs

- processed feature CSV
- preparation JSON
- trained joblib artifact
- training report JSON
- held-out predictions CSV
- independent evaluation JSON

## Success Criterion

The experiment is considered useful only if it improves the walking-relevant metrics on an untouched subject without leakage.

Primary:
- MAE
- within ±5 BPM
- within ±10 BPM
- bias

Secondary:
- RMSE
- R²
- median/P90/P95/max absolute error
- subject and record breakdown

No improvement claim is made until the held-out test result is available.
