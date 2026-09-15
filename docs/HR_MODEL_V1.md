# SafeBand HR Model V1

## Purpose

Train and save a reproducible HR regression reference model from the verified
PPG-DaLiA dataset.

## Model

RandomForestRegressor:
- 500 trees
- min_samples_leaf=2
- random_state=42
- n_jobs=-1

The model uses compact BVP + wrist-ACC quality/spectral features.

## Evaluation

Subject-independent deterministic split:
- 60% subjects train
- 20% validation
- 20% held-out test

The RF is confirmed on validation, then refit on train+validation. The test set
is evaluated once after refitting.

Regression metrics:
- MAE (BPM)
- RMSE (BPM)
- R²
- percentage within 5 BPM
- percentage within 10 BPM

There is no meaningful binary "accuracy" metric for continuous HR regression.

## Run

```powershell
python tools	rain_ppgdalia_hr_model.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --model-out models\ppgdalia_hr_e4_reference.joblib `
  --report-out models\ppgdalia_hr_e4_reference_report.json
```

## Critical hardware-domain limitation

PPG-DaLiA is Empatica E4 wrist BVP. SafeBand's target sensor is MAX30102.

Therefore this `.joblib` is an **E4 reference model**, not a MAX30102 deployment model.
The final MAX30102 branch must be validated and, if necessary, retrained/fine-tuned
using real MAX30102 recordings.

## Expected use

This model answers:

> How well does our chosen HR-estimation approach work on an independent public
> wrist-PPG dataset?

It does not answer:

> How accurate is SafeBand's MAX30102 sensor?
