# Model Baseline Audit

- **Baseline:** SafeBand BITS-2 Motion V2 ExtraTrees
- **Runtime artifact:** `models/activity_bits2_v2/activity_model.joblib`
- **Feature set:** 38 deterministic ACC-only window features in `ai/feature_extraction.py`
- **Predicted labels:** RESTING, SITTING, WALKING, RUNNING; replay maps RESTING/SITTING to STATIC.
- **Training split:** Subject-disjoint BITS-2 development split.
- **External benchmark:** AICOS-HAR subject-disjoint is development/component-level evidence only.
- **Real replay:** CSVs are replay/sanity checks, not final independent validation.
- **Current 2.12 gate:** Must remain BLOCKED: confirmed independent source-session count remains 1.
