# SafeBand AI PPG V4.2 — Diagnostic/Analysis Release

## Goal

V4.2 does not blindly change the HR model. It establishes whether the V4.1
result is driven by useful PPG information, motion context, or specific
subject/activity failures.

### Existing V4.1 held-out result

V4.1 reported 4.576 BPM MAE, 5.955 BPM RMSE, R² 0.591, ±5 BPM 69.57% and
±10 BPM 92.47% on 3,612 held-out windows.

The diagnostics showed walking as the main weakness. V4.2 therefore focuses
on controlled diagnosis before adding new features.

## Important correction

The record parser uses named regex groups:

```python
(?P<subject>...)
(?P<activity>...)
```

and retrieves them by name. It does not depend on `group(1)`/`group(2)`, preventing
the specific capture-group error that occurred in the earlier preparation script.

## Workflow

1. Prepare the same PTT dataset.
2. Train four regressors and compare them against a constant train-mean baseline.
3. Select by validation MAE.
4. Fit optional affine calibration on validation predictions only.
5. Refit the selected model on train + validation.
6. Evaluate the untouched test subjects.
7. Run independent activity/subject/record diagnostics.
8. Run validation-only permutation feature importance.

## Commands

```powershell
python tools/prepare_ptt_hr_v4.py `
  --ptt-root datasets/raw/PTT `
  --out datasets/processed/ptt/ppg_v4_features_v4_2.csv `
  --summary-out datasets/processed/ptt/ppg_v4_prepare_v4_2.json
```

```powershell
python tools/train_ppg_v4_v2.py `
  --input datasets/processed/ptt/ppg_v4_features_v4_2.csv `
  --model-out models/ppg_v4/ppg_v4_2_max30101.joblib `
  --report-out models/ppg_v4/ppg_v4_2_max30101_report.json `
  --predictions-out models/ppg_v4/ppg_v4_2_test_predictions.csv
```

```powershell
python tools/evaluate_ppg_v4_v2.py `
  --predictions models/ppg_v4/ppg_v4_2_test_predictions.csv `
  --report-out models/ppg_v4/ppg_v4_2_test_diagnostics.json
```

```powershell
python tools/ppg_v4_feature_importance_v2.py `
  --input datasets/processed/ptt/ppg_v4_features_v4_2.csv `
  --model models/ppg_v4/ppg_v4_2_max30101.joblib `
  --out models/ppg_v4/ppg_v4_2_validation_feature_importance.csv
```

## Interpretation

For HR regression:

- MAE: lower = better.
- RMSE: lower = better; large mistakes affect it more.
- R²: higher = better; negative means worse than the constant-mean baseline.
- ±3/±5/±10 BPM: higher = better.
- Bias: closer to zero = less systematic over/under prediction.
- P90: lower = better tail error.
- Maximum error: lower is generally desirable but should not be used alone.

There is no meaningful single classification "accuracy" metric for this
regression task.

## Decision rules

Do not modify the feature set merely because a score looks imperfect.

A V4.2 follow-up should be justified if:

- a baseline materially explains the observed performance;
- one activity consistently fails;
- one or more subjects dominate error;
- feature importance shows a clear weakness;
- or a controlled ablation demonstrates that a change improves held-out
  generalization.

Any proposed improvement must be tested with the same subject-level split.
