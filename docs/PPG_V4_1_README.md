# PPG V4.1 Package

Copy into the SafeBand AI repository without overwriting V3.

```text
ai/ppg_v4_features.py

tools/prepare_ptt_hr_v4.py
tools/train_ppg_v4_ptt.py
tools/evaluate_ppg_v4.py
tools/predict_ppg_v4.py

docs/PPG_V4_1_MAX30101.md
docs/PPG_V4_1_EXPERIMENT_LOG.md
docs/PPG_V4_1_README.md
```

## Run order

### 1. Prepare

```powershell
python tools/prepare_ptt_hr_v4.py `
  --ptt-root datasets/raw/PTT `
  --out datasets/processed/ptt/ppg_v4_features_v4_1.csv `
  --summary-out datasets/processed/ptt/ppg_v4_prepare_v4_1.json
```

### 2. Train

```powershell
python tools/train_ppg_v4_ptt.py `
  --input datasets/processed/ptt/ppg_v4_features_v4_1.csv `
  --model-out models/ppg_v4/ppg_v4_1_max30101.joblib `
  --report-out models/ppg_v4/ppg_v4_1_max30101_report.json `
  --predictions-out models/ppg_v4/ppg_v4_1_test_predictions.csv
```

### 3. Diagnose

```powershell
python tools/evaluate_ppg_v4.py `
  --predictions models/ppg_v4/ppg_v4_1_test_predictions.csv `
  --report-out models/ppg_v4/ppg_v4_1_test_diagnostics.json
```

The diagnostics do not retrain the model.

### 4. Single-window inference

```powershell
python tools/predict_ppg_v4.py `
  --model models/ppg_v4/ppg_v4_1_max30101.joblib `
  --ppg-npy ppg.npy `
  --acc-npy acc.npy `
  --gyro-npy gyro.npy
```

Do not commit the complete PTT raw dataset if the repository's dataset policy
excludes large raw data.
