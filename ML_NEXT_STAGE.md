# SafeBand AI — BITS-2 ML Next Stage

The first naive 5-class experiment was NOT accepted as the final model:
subject-independent test accuracy was about 75% and macro-F1 about 0.63.

This stage separates the problem into:
- activity_model.joblib: RESTING / SITTING / WALKING / RUNNING
- fall_detector.joblib: FALL / NON_FALL

The first model is accelerometer-first because BITS-2's released timestamps
do not retain reliable sub-second synchronization across its sensor blocks.

## Commands

From the SafeBand-AI project root:

python tools/prepare_bits2.py --zip datasets/raw/BITS-2/full_dataset.zip --out datasets/processed/bits2

python tools/create_activity_windows.py --input datasets/processed/bits2/bits2_canonical_long.csv --out datasets/processed/bits2/activity_windows.csv

python tools/train_activity_models.py --input datasets/processed/bits2/activity_windows.csv --model-dir models --report models/bits2_training_report.json

## Important

Do not enable AI_MODEL_ENABLED and present the BITS-2 model as the final
SafeBand model until its report has been reviewed and it has been tested
against SafeBand hardware recordings.

BITS-2 does not provide a clean STANDING class, so STANDING remains outside
the first learned activity model. The existing deterministic recognizer
remains the fallback.

Synthetic demo scenarios are not training data.
