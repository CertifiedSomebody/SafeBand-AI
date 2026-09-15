# SafeBand HR Model V4 — leakage-safe temporal benchmark

V4 uses the exact V2 feature extractor and selected Extra Trees configuration.
The previous V4 implementation was incorrect because it imported a V3 feature
contract while loading a V2 model artifact. This package fixes that explicitly.

## Run

```powershell
python tools\evaluate_ppgdalia_temporal_v4.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --model models\ppgdalia_hr_e4_reference_v2.joblib `
  --report-out models\ppgdalia_hr_e4_temporal_v4_report.json
```

## Leakage controls

- subject-level 60/20/20 split
- exact V2 feature contract
- TRAIN-only Extra Trees learner for temporal parameter selection
- validation-only selection of temporal history, physiological rate limit,
  and candidate weight
- final V2 train+validation artifact used only for held-out TEST prediction
- state reset at every subject

The report includes MAE/RMSE/R² plus explicit accuracy-within-1/3/5/10 BPM.
Regression does not have a meaningful exact classification accuracy, so these
are the correct accuracy-style measures.

The E4 model remains a reference. Final SafeBand performance requires paired
MAX30102 + reference-HR recordings.
