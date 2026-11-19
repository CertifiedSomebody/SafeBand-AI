# Development feature engineering (2.13)

`build_development_feature_matrix.py` uses the 2.11.9-accepted inventory and
the 2.12 source-session manifest to create a causal, per-sample feature matrix.
The output is explicitly **DEVELOPMENT ONLY**. Labels and source identifiers
are metadata columns and are not passed into feature extraction.

The fixed feature contract covers accelerometer/gyroscope magnitudes,
orientation encodings and angular rates, causal one-second motion summaries,
environment/audio/battery readings, PPG readings, timing, and per-sensor
availability flags. Invalid -999 readings are masked to zero only alongside
their explicit validity/availability mask. Valid PPG measurements that
contradict their validity flag are rejected. Time must be finite and strictly
increasing; rolling windows contain only current and past samples.

Run:

```powershell
.\venv\Scripts\python.exe tools\feature_engineering\build_development_feature_matrix.py
```

To rerun the full development workflow and generate the consolidated status
report, use `.\venv\Scripts\python.exe tools\run_development_readiness.py`.

Outputs:

- `datasets/features/development_feature_matrix_development_only.csv`
- `reports/dataset_training/development_feature_matrix_2.13.json`

No normalization is fit, because there is no independent training partition.
This matrix is not a train/validation/test set and does not establish model
performance. Model training and all held-out metrics remain blocked by 2.12.

Tests:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tools\feature_engineering\tests -v
```
