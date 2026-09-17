# SafeBand Activity Strong V4

Controlled strengthening of the BITS2 ordinary-activity model after rejection of the V3 hierarchical specialist.

## Run

From SafeBand-AI repository root:

```powershell
python tools\train_activity_strong_v4.py
```

The script expects:

`datasets/processed/bits2/activity_windows.csv`

and writes:

`models/activity_strong_v4/activity_model.joblib`

`models/activity_strong_v4/activity_strong_v4_report.json`

## Design

- RESTING / SITTING / WALKING / RUNNING only
- FALL remains a separate detector
- 38-feature BITS2 ACC contract unchanged
- 5-fold subject-grouped outer evaluation
- 3-fold grouped inner model selection
- ExtraTrees / Random Forest / RBF-SVM candidates
- macro-F1 first, balanced accuracy second
- no outer-fold leakage

See `docs/ACTIVITY_STRONG_V4.md` for the research record and acceptance criteria.
