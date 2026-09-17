# SafeBand PPG V6.1 — MAX30102 Robustness Stage

## Status

**Implementation updated and preflight-tested; benchmark execution is pending on the local PTT dataset.** V4.3 remains the frozen reference until V6.1 produces an untouched held-out result.

## What changed from V6

1. **Cached PPG filtering:** each PPG channel is band-pass filtered once per window instead of being repeatedly filtered during correlation extraction.
2. **FFT autocorrelation:** autocorrelation HR estimation now uses an FFT implementation rather than an O(N²) direct correlation.
3. **Interpolated spectral HR:** the dominant periodogram peak is locally interpolated to reduce dependence on FFT-bin resolution.
4. **Expanded quality diagnostics:** spectral peak SNR, harmonic ratio, absolute PPG-motion correlation, and RR consistency are included.
5. **Explicit raw-vs-temporal reporting:** the final report always exposes both the direct model output and the post-processed temporal output.
6. **Artifact contract:** the saved model records exact feature order, preprocessing assumptions, calibration, and temporal settings.
7. **More defensive training script:** empty partitions, NaN/Inf values, and subject leakage are explicitly rejected.

## Scientific guardrails

- HR is a **regression** target; classification accuracy is not the primary metric.
- ECG waveform is never an input feature. ECG R-peaks provide the HR target in the reference dataset.
- SpO2 is not treated as a continuous target in this stage.
- Subject-independent train/validation/test separation is retained.
- Calibration is fitted on validation data only and can be disabled for an ablation.
- Temporal stabilization resets at every recording and cannot cross record boundaries.
- Raw and temporally stabilized test metrics are both reported, so smoothing cannot hide a degraded raw model.
- BITS-2 remains MAX30102-domain context evidence only because its public canonical release does not provide raw MAX30102 RED/IR waveforms for this pipeline.
- V6.1 is still a **MAX30101-family reference result**, not MAX30102 production validation. Real MAX30102 raw RED/IR recordings are required for the final hardware-domain claim.

## Intended engineering targets

These are acceptance targets, not claimed results:

- MAE approximately 3–4 BPM or lower on an untouched subject-independent test;
- ≥90% of windows within ±5 BPM;
- ≥95% within ±10 BPM;
- low absolute bias;
- materially better walking robustness;
- no hidden degradation from temporal smoothing.

## Execution

From the repository root:

```powershell
python tools\prepare_ptt_hr_v6.py `
  --ptt-root datasets\raw\PTT `
  --out datasets\processed\ptt_hr_v6.csv `
  --summary-out datasets\processed\ptt_hr_v6_summary.json
```

Then:

```powershell
python tools\train_ppg_v6.py `
  --input datasets\processed\ptt_hr_v6.csv `
  --model-out models\ppg_v6_max30101_reference.joblib `
  --report-out models\ppg_v6_max30101_reference_report.json `
  --predictions-out models\ppg_v6_max30101_reference_predictions.csv
```

Optional no-calibration ablation:

```powershell
python tools\train_ppg_v6.py `
  --input datasets\processed\ptt_hr_v6.csv `
  --model-out models\ppg_v6_max30101_reference_nocal.joblib `
  --report-out models\ppg_v6_max30101_reference_nocal_report.json `
  --predictions-out models\ppg_v6_max30101_reference_nocal_predictions.csv `
  --disable-calibration
```

## Interpretation rule

Do **not** overwrite the frozen V4.3 result merely because V6.1 is newer. Promote V6.1 only if its untouched test evidence improves the agreed HR metrics without sacrificing the walking regime or introducing leakage. Otherwise keep V4.3 as the benchmark reference and proceed to real MAX30102 acquisition/domain validation.
