# SafeBand HR Model V3

V3 is the consolidated HR pipeline improvement.

## Improvements
- robust PPG normalization
- adaptive validated bandpass range
- sub-bin spectral interpolation
- autocorrelation HR candidate
- peak/IBI candidate and regularity
- candidate-agreement quality score
- wrist-ACC motion-quality branch
- multiple nonlinear regressors
- validation-only affine calibration
- strict subject-independent split
- final train+validation refit
- deployment-oriented physiological continuity class for streaming use

## Train

```powershell
python tools\train_ppgdalia_hr_model_v3.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --model-out models\ppgdalia_hr_e4_reference_v3.joblib `
  --report-out models\ppgdalia_hr_e4_reference_v3_report.json
```

V3 must still be evaluated on the held-out subjects. It is not assumed to
beat V2. The MAX30102 branch must eventually be retrained/validated using
actual MAX30102 recordings.
