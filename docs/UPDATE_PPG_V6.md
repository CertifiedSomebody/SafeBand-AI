# PPG V6.1 Update

This is a surgical update over the existing SafeBand Phase-1 repository. V4.1,
V4.2, V4.3 and Walking V5.3 are retained for rollback/reference.

## Updated

- `ai/ppg_v6_features.py`
- `tools/train_ppg_v6.py`
- `tools/prepare_ptt_hr_v6.py`
- `tools/audit_bits2_max30102_hr.py`
- `docs/PPG_V6_MAX30102_ROBUSTNESS.md`

## Main engineering improvements

- cached PPG filtering to avoid repeated work;
- FFT-based autocorrelation HR candidate;
- sub-bin spectral peak interpolation;
- additional SQI/motion contamination features;
- safer temporal post-processing implementation;
- stronger model artifact metadata;
- explicit raw-versus-temporal test reporting;
- defensive checks for empty splits, invalid values, and subject leakage.

## Preflight completed

The updated feature extractor was executed on a synthetic 8-second six-channel
PPG/IMU window and produced a finite feature vector. The training script was
also executed end-to-end on a synthetic subject-separated regression table.

The real PTT benchmark was **not** fabricated or substituted: the current
runtime copy of the repository does not contain `datasets/raw/PTT/csv` (or
`CSV`), so the real benchmark must be run on the machine containing the
acquired PTT dataset.

## Run on the real dataset

```powershell
python tools\prepare_ptt_hr_v6.py `
  --ptt-root datasets\raw\PTT `
  --out datasets\processed\ptt_hr_v6.csv `
  --summary-out datasets\processed\ptt_hr_v6_summary.json

python tools\train_ppg_v6.py `
  --input datasets\processed\ptt_hr_v6.csv `
  --model-out models\ppg_v6_max30101_reference.joblib `
  --report-out models\ppg_v6_max30101_reference_report.json `
  --predictions-out models\ppg_v6_max30101_reference_predictions.csv
```

**Do not replace V4.3 in project documentation until these commands have been
run against the real PTT data and the held-out test metrics have been checked.**


## V6.1.1 compatibility patch
Fixed NumPy compatibility in `ai/ppg_v6_features.py`: avoid eagerly evaluating the removed `np.trapz` fallback when `np.trapezoid` is available. This supports newer NumPy releases where `np.trapz` is absent.
