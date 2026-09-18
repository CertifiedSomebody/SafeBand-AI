# SafeBand PAMAP2 Stationary V2

PAMAP2 stationary/postural benchmark using a compact, classical RBF-SVM on robust window-level IMU features.

## Why this model
The current PAMAP2 stationary set is small (8 subjects) and has large subject-to-subject variation. A compact RBF-SVM is a better next experiment than another large neural sequence model: it works well on small tabular feature spaces and can model nonlinear boundaries without adding many trainable parameters.

The feature extractor uses only the hand/wrist accelerometer + gyroscope window. No chest/ankle signals or heart rate are used.

## Expected repository layout

SafeBand-AI-main/
  datasets/raw/PAMAP2/dataset2.csv
  datasets/processed/pamap2/stationary_accgyro_windows.npz
  tools/train_pamap2_stationary_v2.py

Run from repository root:

    python tools\train_pamap2_stationary_v2.py

Optional:

    python tools\train_pamap2_stationary_v2.py --data datasets\processed\pamap2\stationary_accgyro_windows.npz --out models\pamap2_stationary_v2

## Evaluation
- 5-fold StratifiedGroupKFold by subject.
- Inner 4-fold grouped CV selects C and gamma using macro-F1.
- Outer test subjects are never used for model selection.
- Standardization is fitted only on the outer training fold.
- Features are computed per window before scaling.
- Reports pooled and per-fold accuracy, balanced accuracy, macro-F1, class F1 and confusion matrix.

This is a benchmark experiment, not a claim of deployment performance.
