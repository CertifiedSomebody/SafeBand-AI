# V5.2 Requirements and Commands

## Environment

```powershell
pip install wfdb numpy pandas scipy scikit-learn joblib
```

Scripts are intentionally runnable from the repository root as `python tools\...`. Each tool inserts the repository root into `sys.path` before importing `ai`.

## 1. Prepare

```powershell
python tools\prepare_wrist_walking_v5_2.py `
  --root datasets/raw/WristPPGExercise `
  --out datasets/processed/wrist_ppg/walking_v5_2_features.csv `
  --summary-out datasets/processed/wrist_ppg/walking_v5_2_prepare.json
```

Expected baseline from V5.1 is approximately 1,097 valid windows. The exact V5.2 feature count can change because wide-range ACC and magnetometer features are now retained for controlled experiments.

## 2. Train

```powershell
python tools\train_walking_v5_2.py `
  --input datasets/processed/wrist_ppg/walking_v5_2_features.csv `
  --model-out models/ppg_walking_v5_2/ppg_walking_v5_2.joblib `
  --report-out models/ppg_walking_v5_2/ppg_walking_v5_2_report.json `
  --predictions-out models/ppg_walking_v5_2/ppg_walking_v5_2_test_predictions.csv
```

## 3. Independent evaluation

```powershell
python tools\evaluate_wrist_walking_v5_2.py `
  --input datasets/processed/wrist_ppg/walking_v5_2_features.csv `
  --predictions models/ppg_walking_v5_2/ppg_walking_v5_2_test_predictions.csv `
  --out models/ppg_walking_v5_2/ppg_walking_v5_2_diagnostics.json
```

## Interpreting the result

Do not choose a model from the largest validation percentage.

Use:
1. mean OOF MAE to select the development configuration;
2. untouched S9 MAE / ±5 / ±10 / bias to judge external generalization;
3. record-level diagnostics to identify walking failure modes.

A lower MAE is better. A higher ±5/±10 percentage is better. Bias closer to zero means less systematic over/under-estimation.
