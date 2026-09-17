# PAMAP2 Stationary V2 — RBF-SVM

## Objective
Test a compact nonlinear classifier suited to a small subject-grouped stationary/postural dataset.

Classes:
- LYING
- SITTING
- STANDING

Input:
- hand/wrist ACC: ax, ay, az
- hand/wrist GYRO: gx, gy, gz
- 2 s windows at the PAMAP2 100 Hz source rate
- 50% overlap

## Feature design
The feature vector contains:
1. per-axis ACC statistics
2. per-axis GYRO statistics
3. ACC and GYRO magnitude statistics
4. mean-acceleration direction cosines
5. cross-axis ACC covariance terms
6. low-frequency spectral energy ratios

The aim is to retain posture/gravity information while reducing sensitivity to raw sample-level noise.

## Validation protocol
Five outer StratifiedGroupKFold splits are used with PeopleId as the group. A separate four-fold grouped split inside each outer training set selects the SVM hyperparameters. The outer test subjects are not used for selection.

Scaling is also fitted only on the outer training data.

## Interpretation
Compare this result directly against PAMAP2 Stationary V1 (TinyCNN). Do not select a winner from one metric alone. Look especially at:
- pooled macro-F1
- balanced accuracy
- SITTING F1/recall
- SITTING↔STANDING confusion
- fold-to-fold variance

If this model remains unstable across subjects, the next step should be sensor-orientation/domain analysis rather than more hyperparameter hunting.
