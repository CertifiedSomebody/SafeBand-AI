# PPG V4 Experiment Log

## Decision

Create a new PPG V4 branch instead of modifying V3 in place. V3 remains the reproducible E4 reference benchmark; V4 adds a MAX30101-domain branch using the Pulse Transit Time PPG Dataset.

## Dataset basis

The PTT dataset contains 22 healthy subjects and 66 recordings across sitting, walking and running. Its README describes two Maxim MAX30101 PPG sensors with red/infrared/green wavelengths, two finger sites, 500 Hz distributed PPG channels, MPU-9250 inertial data and ECG R-peak annotations.

## Target

Primary target: window-level HR in BPM derived from ECG R peaks.

## Windowing

- Window: 8 seconds
- Shift: 2 seconds
- PPG: 500 Hz
- Motion: 500 Hz

## Features added relative to the E4-oriented representation

1. Six optical channels instead of one BVP channel.
2. Red / IR / green relationships.
3. Distal / proximal optical relationships.
4. Per-channel AC/DC representation.
5. Multi-channel signal morphology and spectral HR candidates.
6. PPG-to-motion coupling for quality/context information.

## Explicit non-goals

- No continuous SpO2 model from `spo2_start` / `spo2_end`.
- No claim that MAX30101 training equals MAX30102 validation.
- No ECG waveform input to the model.
- No random window split.

## Calibration plan

Real MAX30102 recordings will be collected after the V4 reference model is trained. Calibration will be evaluated on held-out MAX30102 recordings rather than assumed to be beneficial. If calibration or fine-tuning is used, its data and selection procedure will be documented separately.

## Reproducibility

All preparation, feature extraction, training and prediction code is versioned under `tools/` and `ai/`. Model artifacts and benchmark reports belong under `models/`.
