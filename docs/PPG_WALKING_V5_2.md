# SafeBand PPG Walking V5.2 — Robust Cross-Subject Experiment

## Why V5.2 exists

V5.1 showed that adding accelerometer + gyroscope features substantially improved its single validation subject, but its validation-derived unrestricted affine calibration collapsed on untouched S9.

V5.2 changes the experiment instead of hiding that failure.

## Parameters you should understand

| Parameter | V5.2 value | Why it matters |
|---|---:|---|
| Window | 8 s | Enough beats to estimate HR while keeping response reasonably fast |
| Shift | 2 s | New estimate every 2 s |
| Sampling | 256 Hz | Native dataset sampling rate |
| PPG band | 0.5–5 Hz | Keeps the main cardiac pulse band and suppresses slow drift/high-frequency noise |
| HR target | ECG R-peak RR intervals | Reference target; ECG waveform is NOT an input |
| Normalization | robust per window | Reduces subject/device amplitude differences |
| Model candidates | Ridge/RF/ExtraTrees/HGB | Same core family as previous SafeBand work |
| CV | 5-fold GroupKFold | Each development subject is held out as a complete fold |
| Selection | mean OOF MAE | Avoids trusting one validation subject |
| Calibration | optional bounded affine | Slope constrained to 0.80–1.20; no negative/inverted calibration |
| Final test | S9 | Never used during development |

## Development/test protocol

Development subjects:
`S1, S2, S3, S6, S8`

Untouched external test:
`S9`

The five development subjects are evaluated with 5-fold GroupKFold. In each fold, an entire subject is validation data.

S9 is not used for:
- model selection
- feature selection
- calibration
- threshold selection
- parameter tuning

Only after all decisions are frozen is the selected model trained on all five development subjects and evaluated on S9.

## Feature experiments

1. `ppg_only`
2. `ppg_plus_gyro`
3. `ppg_plus_acc_gyro`
4. `all_motion` = PPG + gyro + low-noise ACC + wide-range ACC + magnetometer

This lets us answer an engineering question: **does additional motion sensing improve walking HR generalization, rather than merely fitting one subject?**

## What counts as improvement

Primary:
- lower MAE
- higher percentage within ±5 BPM
- higher percentage within ±10 BPM
- bias closer to zero

Secondary:
- lower RMSE
- higher R²
- lower median/P90/P95/max absolute error
- consistency across subjects/records

For SafeBand, a model that performs very well on one person but fails on another is not considered a robust improvement.

## V5.1 comparison context

V5.1 selected PPG + ACC + gyro with RF. Its raw validation MAE was 7.33 BPM, but unrestricted calibration reduced validation MAE to 3.20 BPM and then produced 27.22 BPM MAE on untouched S9. That is why V5.2 uses grouped OOF selection and bounded calibration.

V5.2 must be judged on the untouched S9 result, not on its cross-validation score alone.
