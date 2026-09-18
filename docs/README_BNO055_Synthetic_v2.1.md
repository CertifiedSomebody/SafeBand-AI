# SafeBand AI — BNO055 Synthetic V2.1

V2.1 is the **final planned synthetic generator revision** for the current
software/ML development phase.

It is designed to support:
- software pipeline integration
- reproducible window generation
- controlled subject-independent ML experiments
- development before real BNO055 hardware data is available

It must not be presented as real-hardware validation.

## Quick start

From the SafeBand-AI repository root:

```powershell
python tools\generate_bno055_synthetic_v2_1.py --subjects 20 --sessions 2 --duration 15 --seed 42
python tools\audit_bno055_synthetic_v2_1.py
python tools\build_bno055_windows_v2_1.py
python tools\visualize_bno055_synthetic_v2_1.py
```

The default raw dataset is:

`datasets/synthetic/bno055/bno055_synthetic_v2_1.csv`

The default window artifact is:

`datasets/synthetic/bno055/bno055_windows_v2_1.npz`

## Raw sensor contract

9 raw channels:

1. accel_x_mps2
2. accel_y_mps2
3. accel_z_mps2
4. gyro_x_dps
5. gyro_y_dps
6. gyro_z_dps
7. mag_x_uT
8. mag_y_uT
9. mag_z_uT

Sampling: 100 Hz.

Default window: 2 seconds / 200 samples, with a 1-second step.

## Activity classes

- FALL
- LYING
- RUNNING
- SITTING
- SIT_TO_STAND
- STAIRS
- STANDING
- STAND_TO_SIT
- WALKING

## V2.1 changes

The generator uses body-frame angular velocity as the source signal and
integrates it into quaternion attitude. Gravity and magnetometer measurements
are then derived from that attitude. This removes the previous shortcut of
driving gyro from large Euler-angle derivatives.

Motion amplitudes and angular rates are randomized within conservative
software-domain bounds. Gait contains frequency, amplitude, phase, harmonics,
cross-axis coupling, and noise variation.

Fall and posture-transition recordings carry explicit event metadata:
`event_type`, `event_start_s`, and `event_end_s`.

Magnetometer conditions are sampled from subject/session environmental
parameters and are not assigned by activity label.

## Finality rule

After V2.1 passes the waveform checkpoint, do not create V2.2/V3 merely to
increase synthetic ML accuracy. The next meaningful validation source is real
BNO055 hardware data.

Synthetic ML results must be reported explicitly as synthetic-domain results.
