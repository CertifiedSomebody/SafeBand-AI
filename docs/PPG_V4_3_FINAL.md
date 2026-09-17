# SafeBand AI — PPG V4.3 FINAL

## Purpose

V4.3 is the final controlled software release for the current PTT/MAX30101-domain
PPG heart-rate reference experiment. It consolidates the V4 → V4.1 → V4.2 work
without silently replacing earlier results.

### What is being solved?

Estimate heart rate (BPM) from wrist multi-wavelength PPG with motion context.

This is a **regression** problem. MAE/RMSE/R² and within-tolerance percentages
are therefore more meaningful than classification accuracy.

## Dataset and domain

The PTT dataset provides multi-wavelength PPG, IMU, ECG and reference pulse-oximeter
measurements. The model uses PPG + IMU-derived features. ECG supplies the HR target
through ECG peak annotations; the ECG waveform itself is not a feature.

The dataset is a **MAX30101-domain reference**, not a MAX30102 dataset. Final
SafeBand validation still requires actual MAX30102 recordings and hardware-domain
evaluation.

## V4.3 methodology

- 8-second windows, 2-second shift.
- 22 subjects, 66 records, run/sit/walk.
- 159 established features.
- Numeric subject-level 60/20/20 split.
- Validation selects the model by MAE.
- Optional affine calibration is fitted on validation only.
- Final model is refit on train + validation.
- Held-out test subjects remain untouched until final evaluation.
- Constant train-mean baseline is reported.
- A transparent PPG spectral/peak-HR baseline is reported.
- HGB PPG-only vs HGB PPG+motion ablation is reported.
- Activity, subject and record diagnostics are produced.

## Metrics

**MAE:** average absolute BPM error; lower is better.

**RMSE:** penalizes large errors more strongly; lower is better.

**R²:** variance explained relative to a mean baseline; higher is better.
Negative R² indicates the predictions do not beat that constant-reference
criterion for the evaluated subset.

**±3 / ±5 / ±10 BPM:** percentage of predictions within the stated tolerance;
higher is better.

**Bias:** mean(prediction − target). Near zero means less systematic
over/under-estimation.

**P90/P95 absolute error:** tail error; lower is better.

## Historical results

### PPG-DaLiA V3
E4 BVP reference benchmark. It is not MAX30102 validation.

### PTT V4
First MAX30101-domain feature/model experiment.

### V4.1
Corrected numeric subject ordering and preparation. Held-out test:
MAE 4.576 BPM, RMSE 5.955 BPM, R² 0.591, ±5 BPM 69.57%, ±10 BPM 92.47%,
bias −4.407 BPM.

### V4.2
Diagnostic release confirming that walking was the dominant weakness.
Held-out activity MAE:
- run: 2.688 BPM
- sit: 4.548 BPM
- walk: 6.497 BPM

The most difficult records were s19_walk and s20_walk.

V4.3 now makes the final controlled PPG-only vs PPG+motion comparison while
keeping the same dataset, features and subject split.

## Research interpretation

A good overall score must not hide activity- or subject-specific failure.
Therefore the final report contains overall, activity, subject and record metrics.

No claim of clinical accuracy is made. No continuous SpO2 target is fabricated.
No MAX30101 result is presented as MAX30102 hardware validation.

## Commands

### Prepare
```powershell
python tools/prepare_ptt_hr_v4.py `
  --ptt-root datasets/raw/PTT `
  --out datasets/processed/ptt/ppg_v4_features_v4_3.csv `
  --summary-out datasets/processed/ptt/ppg_v4_prepare_v4_3.json
```

### Train/final evaluation
```powershell
python tools/train_ppg_v4_3_final.py `
  --input datasets/processed/ptt/ppg_v4_features_v4_3.csv `
  --model-out models/ppg_v4/ppg_v4_3_final_max30101.joblib `
  --report-out models/ppg_v4/ppg_v4_3_final_report.json `
  --predictions-out models/ppg_v4/ppg_v4_3_final_test_predictions.csv
```

### Independent diagnostics
```powershell
python tools/evaluate_ppg_v4_3_final.py `
  --predictions models/ppg_v4/ppg_v4_3_final_test_predictions.csv `
  --report-out models/ppg_v4/ppg_v4_3_final_diagnostics.json
```

## V4.3 stopping rule

Do not call V4.3 an improvement merely because validation improved.
The final test set is the fixed comparison point.

If the controlled ablation shows motion helps and the held-out result remains
stable, retain PPG+motion. If PPG-only performs similarly or better, the model
can be simplified without sacrificing measured performance.

After this experiment, the next stage should be **real MAX30102 hardware
recording and validation**, not endless benchmark tuning.
