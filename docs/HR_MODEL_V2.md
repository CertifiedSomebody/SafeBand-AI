# SafeBand HR Model V2 — Enhanced reference benchmark

V2 is a genuine feature/model upgrade, not threshold tuning.

## Improvements over V1

- PPG normalization to reduce dependence on optical amplitude scale.
- 0.6–4.0 Hz band-pass filtering.
- Zero-padded spectral peak localization.
- Spectral entropy and fundamental/harmonic energy.
- Autocorrelation-based HR candidate with parabolic lag interpolation.
- Peak/IBI regularity and prominence.
- Cross-method HR candidate median/spread.
- Additional wrist-ACC motion-quality features.
- Training-only median imputation.
- Validation comparison of Ridge, Random Forest, Extra Trees, and
  HistGradientBoosting.
- Final model refit on train+validation only after validation selection.
- Held-out subject test remains untouched.

## Run

```powershell
python tools	rain_ppgdalia_hr_model_v2.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --model-out models\ppgdalia_hr_e4_reference_v2.joblib `
  --report-out models\ppgdalia_hr_e4_reference_v2_report.json
```

## Important

V2 can improve the E4 benchmark, but no improvement is guaranteed.
If V2 does not improve held-out subject performance, we keep V1 as the
reference and stop optimizing the E4 dataset.

This remains an Empatica E4 reference model, not a MAX30102 model.
