# BNO055 Synthetic Dataset V2.1

## Purpose

V2.1 is a controlled synthetic benchmark for SafeBand AI development while
real BNO055 hardware recordings are unavailable.

It validates the software path:

`generation -> audit -> windowing -> visualization -> ML`

It does **not** validate BNO055 hardware performance, sensor calibration,
human activity recognition in the field, or real-world fall detection.

## Why V2.1 is the final synthetic revision

V1 exposed fixed-orientation and waveform shortcuts.

V2 removed the largest shortcuts but still used orientation trajectories whose
derivatives produced overly large and overly regular gyro patterns.

V2.1 changes the motion source itself:

`bounded body angular rate -> quaternion integration -> orientation -> gravity/MAG`

This makes gyro, attitude, gravity and magnetic field mutually consistent at
the synthetic-model level.

## Signal model

### Stationary

SITTING, STANDING and LYING contain:
- subject-specific baseline orientation
- session-specific pose jitter
- weak periodic wrist micro-motion
- low-frequency drift
- accelerometer and gyro noise
- sensor bias

The absolute orientation is therefore not a fixed label code.

### Dynamic

WALKING, RUNNING and STAIRS contain:
- randomized gait frequency
- randomized amplitude
- phase variation
- harmonics
- cross-axis coupling
- subject/session variation
- bounded body angular rates
- linear acceleration with activity-specific but non-unique structure

### Transitions

SIT_TO_STAND and STAND_TO_SIT contain:
- an explicit central event interval
- smooth posture change
- angular motion
- linear acceleration
- settling behavior

Event timing is stored in every row so event-centered visualization and later
event-aware window analysis are reproducible.

### Fall

FALL contains:
- pre-event destabilization
- rapid multi-axis rotation
- short impact acceleration
- impact rotation
- post-impact settling

The event is intentionally temporal rather than being represented as random
high gyro values throughout the recording.

### Magnetometer

A world-frame magnetic vector is generated per subject/session and transformed
into the sensor frame using the generated attitude. Small environmental drift,
scale variation, bias and measurement noise are added.

The magnetic field is therefore not directly selected from the activity label.

## Window artifact

Default:
- 100 Hz
- 2 s window
- 1 s step
- 9 channels
- shape `(N, 9, 200)`

Saved metadata:
- `y`
- `groups` = subject IDs
- `sessions`
- `starts`
- `event_start_s`
- `event_end_s`
- `is_event_window`
- channel names
- class names
- sampling rate
- window/step duration

## Evaluation rule

Any ML benchmark must use subject-independent grouping. A random row split is
not acceptable because overlapping windows from the same synthetic subject can
otherwise leak subject-specific characteristics into the test set.

## Interpretation

A strong synthetic-domain score means the implementation can learn the
patterns produced by this generator. It does not establish performance on a
real BNO055, real users, or real-world activity/fall scenarios.

The project should move to real hardware-domain validation when the BNO055
prototype is available.
