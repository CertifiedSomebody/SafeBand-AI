# SafeBand Activity Recognition V2 — FORTH-TRACE

## Dataset / sensor match
FORTH-TRACE contains 3-axis accelerometer, gyroscope and magnetometer signals at 51.2 Hz from 15 participants using five Shimmer nodes. Device 1 is the left wrist. SafeBand's eventual BNO055 provides the same nine motion axes (ACC/GYRO/MAG), so this experiment uses the common 9-DoF sensor space.

## Preparation
`tools/prepare_forth_trace.py` reads only `part*/part*dev1.csv` and never modifies raw files.

Two targets are produced:

- `core_activity_windows.csv`: STANDING, SITTING, WALKING, STAIRS. Talk variants are mapped to the parent activity. Transition labels are excluded.
- `transition_aware_windows.csv`: the same four classes plus TRANSITION, where all nine transition labels are grouped together.

Default segmentation is 3.0 s, 50% overlap, 90% dominant-label purity. Participant IDs are retained only for group splitting and are never model features.

## Training
`tools/train_forth_trace_activity.py` extracts deterministic ACC/GYRO/MAG time-domain and spectral features and benchmarks ExtraTrees, RandomForest, HistGradientBoosting, RBF-SVM and XGBoost when available. The test subjects are never used for model selection.

The selected model is chosen on validation macro-F1, then balanced accuracy, then accuracy. Final test metrics are reported only after model selection.

## Why this is not merged into BITS2 yet
FORTH-TRACE has no running class and has different sensor hardware, coordinate frames and sampling rate from BITS2. It is therefore an external wrist-domain validation dataset first. Cross-dataset training/fusion should be a separate experiment after independent baselines are established.

## Reproducibility result from the current implementation
Using the default 3.0 s / 50% overlap / 90% purity preparation and a fixed subject-independent split (60/20/20 by participant), the core four-class benchmark selected an RBF-SVM on validation and achieved on untouched test subjects:

- Accuracy: **98.20%**
- Balanced accuracy: **98.31%**
- Macro-F1: **98.19%**

Per-class test F1: SITTING 0.99, STAIRS 0.98, STANDING 0.98, WALKING 0.98.

The transition-aware five-class experiment selected Random Forest and achieved 97.15% accuracy, 90.82% balanced accuracy and 93.21% macro-F1. The lower macro score is driven mainly by the small TRANSITION class; its test recall was 62% in this fixed split. This is retained as a secondary experiment rather than being presented as the core activity model.

These are results on FORTH-TRACE only. They must not be presented as expected performance on the eventual BNO055 hardware until cross-dataset and real-device validation are completed.
