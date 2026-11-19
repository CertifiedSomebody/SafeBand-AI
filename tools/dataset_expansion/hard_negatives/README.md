# SafeBand AI — Milestone 2.10.6

## Hard-Negative Generation

Hard negatives are intentionally difficult non-hazard samples.

They preserve:

- source activity
- source label semantics
- validity flags
- sample index
- CSV schema
- sentinel semantics
- physical session ordering

They modify only selected sensor/timing values.

## Profiles

### ABRUPT_ACCELERATION
Short acceleration bursts without a fall transition.

### ABRUPT_DECELERATION
Rapid deceleration while remaining a normal activity sample.

### SHARP_TURN
Gyroscope-dominant rotational event.

### MOTION_SPIKE
Combined accelerometer and gyroscope transient.

### STOP_START
Temporary motion suppression followed by recovery.

### PPG_ARTIFACT
Optical waveform disturbance while retaining the original contact/validity state.

### TIMING_STALL_NO_HAZARD
Acquisition stall without changing sensor values or activity.

### NEAR_FALL_RECOVERY
Strong transient motion followed by recovery, without converting the sample into a hazard label.

## Important

These generated samples are:

SYNTHETIC_UNVALIDATED

They are augmentation fixtures and hard-negative training material, not evidence of real-world hazard performance.

Real validation remains mandatory.
