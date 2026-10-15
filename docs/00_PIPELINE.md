# AICOS-HAR V4 pipeline contract

## Purpose
Establish a reproducible, leakage-safe ACC activity benchmark on AICOS-HAR before adding sensor fusion.

## Data contract
- Source hierarchy: Participant / Activity_TrialID / Device_Position / Sensor.txt
- Canonical sensors: ACC, GYRO, MAG, BARO
- Metadata activity labels are authoritative.
- `LiftDown`/`LiftUp` are metadata names; raw path aliases are `ElevatorDown`/`ElevatorUp`.
- Acquisition ID is Participant/Activity_TrialID/Device_Position.

## Split contract
AICOS official `Fold=test` is the final holdout. All non-test folds are training material. CNN validation is an acquisition-grouped split inside the non-test portion.

## Window contract
ACC only, 50 Hz, 2 seconds, 1 second step, no crossing acquisition boundaries. Resampling uses timestamp-based linear interpolation.

## Metrics
Report accuracy, balanced accuracy, macro-F1 and confusion matrix. Macro-F1 and balanced accuracy are important because the 11-class activity inventory is imbalanced.

## Stop conditions
Stop and fix the pipeline if path reconciliation leaves unresolved IDs, if no official test samples are present, or if the window builder produces zero samples.
