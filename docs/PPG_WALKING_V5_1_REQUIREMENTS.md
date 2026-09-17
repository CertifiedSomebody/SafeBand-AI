# PPG Walking V5.1 Requirements

Install inside the SafeBand virtual environment:

```powershell
pip install wfdb numpy pandas scipy scikit-learn joblib
```

The scripts are designed to run from the repository root with:

```powershell
python tools\prepare_wrist_walking_v5_1.py ...
```

The tools explicitly add the repository root to `sys.path`, preventing the earlier `ModuleNotFoundError: No module named 'ai'` when scripts are launched from `tools/`.

## Preparation

```powershell
python tools\prepare_wrist_walking_v5_1.py `
  --root datasets/raw/WristPPGExercise `
  --out datasets/processed/wrist_ppg/walking_v5_1_features.csv `
  --summary-out datasets/processed/wrist_ppg/walking_v5_1_prepare.json
```

## Training

```powershell
python tools\train_walking_v5_1.py `
  --input datasets/processed/wrist_ppg/walking_v5_1_features.csv `
  --model-out models/ppg_walking_v5/ppg_walking_v5_1.joblib `
  --report-out models/ppg_walking_v5/ppg_walking_v5_1_report.json `
  --predictions-out models/ppg_walking_v5/ppg_walking_v5_1_test_predictions.csv
```

## Independent evaluation

```powershell
python tools\evaluate_wrist_walking_v5_1.py `
  --input datasets/processed/wrist_ppg/walking_v5_1_features.csv `
  --model models/ppg_walking_v5/ppg_walking_v5_1.joblib `
  --out models/ppg_walking_v5/ppg_walking_v5_1_diagnostics.json
```

## Reproducibility

Random state is fixed at 42 for tree models. The test subject is never used for model selection.
