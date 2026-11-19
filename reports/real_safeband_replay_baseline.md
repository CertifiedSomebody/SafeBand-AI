# Real SafeBand Replay Baseline

DEVELOPMENT/REPLAY ONLY. These rows do not establish final independent SafeBand generalization.

The baseline model is the BITS-2 ExtraTrees activity model using 38 ACC features. Window sizes are 10/20/40/60 consecutive CSV rows (nominally 0.5/1/2/3 seconds at 20 Hz), but real recordings have a median interval of roughly 394 ms and substantial stalls.

## 40-row baseline

| Recording | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Windows |
|---|---|---|---|---|---|
| stationary.csv | 0.0 | 0.0 | 0.0 | 0.0 | 17 |
| stationary_2.csv | 0.0 | 0.0 | 0.0 | 0.0 | 17 |
| walking.csv | 0.23529411764705882 | 0.23529411764705882 | 0.19047619047619047 | 0.38095238095238093 | 17 |