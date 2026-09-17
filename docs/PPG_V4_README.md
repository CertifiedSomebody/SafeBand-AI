# PPG V4 files

Copy these files into the SafeBand AI repository while preserving the existing V3 files:

```text
ai/ppg_v4_features.py
tools/prepare_ptt_hr_v4.py
tools/train_ppg_v4_ptt.py
tools/predict_ppg_v4.py
docs/PPG_V4_MAX30101.md
docs/PPG_V4_EXPERIMENT_LOG.md
```

Do not commit the full PTT raw dataset if repository policy excludes large raw datasets.

Run preparation before training. Training produces the V4 MAX30101-domain model and JSON report.
