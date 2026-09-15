# SafeBand AI — ML Phase Preparation

## Changed files

1. `config/settings.py`
   - Added ML model path/name/version/type settings.
   - Added activity confidence thresholds.
   - Added ML window configuration.
   - Added explicit initial ML activity labels.

2. `ai/activity_recognition.py`
   - Added an ML inference path that is opt-in through `AI_MODEL_ENABLED`.
   - Preserved the existing rule-based classifier as the fallback.
   - Added `sensor_window` support.
   - Added result metadata: source, model name/version, window size.
   - The simulation scenario is still never used for classification.
   - ML FALL only becomes an automatic emergency when its confidence reaches the configured fall threshold.

3. `ai/window_buffer.py` (new)
   - Fixed-size FIFO sensor window for time-series inference.
   - Independent of Streamlit and hardware implementation.

4. `ai/feature_extraction.py` (new)
   - Deterministic window-level features for the first activity model.
   - Feature contract is intentionally provisional and can be revised after dataset research.

5. `ai/ml_activity_model.py` (new)
   - Runtime adapter for a future `models/activity_model.joblib`.
   - Supports a joblib payload containing `model`, `feature_columns`, `model_name`, and `model_version`.
   - No trained model is included yet.

6. `data/dataset_schema.py` (new)
   - Canonical SafeBand row-level data schema.
   - Separates raw sensor fields, metadata and ML labels.
   - Activity datasets only require timestamp/subject/session/accelerometer/label fields; other SafeBand sensors are optional.

7. `app.py`
   - Added a per-user sensor window in Streamlit session state.
   - Clears the window whenever the demo scenario changes.
   - Passes the window to the activity-recognition layer.
   - Activity logs now include ML/rule source and window size.

8. `requirements.txt`
   - Added `numpy`, `scikit-learn`, and `joblib` for the upcoming training/inference pipeline.

9. `.gitignore`
   - Added trained-model artifact patterns.

## Important behavior

`AI_MODEL_ENABLED` remains `False`.

Therefore the current application continues to behave exactly as a rule-based prototype until a trained and validated model is produced.

Do NOT train a model from the current deterministic demo profiles and call it the final AI model. The demo profiles remain for application testing. The real training dataset will be selected and constructed in the next phase.
