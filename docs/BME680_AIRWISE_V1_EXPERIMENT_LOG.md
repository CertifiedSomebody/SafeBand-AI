# BME680 AIRWISE V1 — Experiment Log

## Starting point
AIRWISE v1.0.0 was selected as the primary BME680 dataset. The acquisition log reports 146,538 minute-level indoor readings across office, kitchen and hallway, with BME680 temperature, humidity, pressure and gas resistance.

## Experiment order
1. Audit the actual extracted files.
2. Verify schema, missingness and class distribution.
3. Prepare SafeBand-specific processed data without modifying raw AIRWISE.
4. Build leakage-safe chronological splits.
5. Compare four baseline classifiers.
6. Select using validation balanced accuracy.
7. Refit selected model on train+validation.
8. Evaluate untouched test.
9. Inspect confusion matrix and per-location performance.
10. Only after this benchmark decide whether additional modeling is justified.

## Leakage controls
Do not use:
- IAQ_proxy
- IAQ_proxy_zscore
- source sensor z-scores
- Z-score
- target labels
as model inputs.

Do not randomly split adjacent time-series rows.

## Interpretation rule
A high test score would show that AIRWISE's environmental state can be learned from BME680 measurements. It would not automatically mean that a physical SafeBand can detect dangerous environments in the real world.

## Next stage after V1
If the benchmark is useful:
- evaluate per-location robustness;
- test anomaly detection separately;
- collect actual ESP32 + BME680 data;
- compare hardware distributions against AIRWISE;
- calibrate/adapt only if domain shift is demonstrated.
