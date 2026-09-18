# BNO055 Synthetic V2 — Implementation Notes

## Dataset contract

- Sampling rate: 100 Hz
- Raw ML channels: 9
  - accelerometer X/Y/Z
  - gyroscope X/Y/Z
  - magnetometer X/Y/Z
- Window: 2 seconds = 200 samples
- Default overlap: 50% (100-sample step)
- Group split key: `subject_id`
- Label column: `activity_label`
- Classes: FALL, LYING, RUNNING, SITTING, SIT_TO_STAND, STAIRS,
  STANDING, STAND_TO_SIT, WALKING

## V2 anti-shortcut design

1. Subject-specific wrist orientation and sensor biases.
2. Session-specific magnetic environment.
3. Magnetometer is produced from a world field transformed by orientation,
   rather than assigning a fixed magnetic vector to each activity.
4. Stationary states contain small wrist micro-motion.
5. Dynamic states vary frequency, amplitude and phase.
6. Gait uses harmonics and cross-axis coupling.
7. Transitions contain an explicit temporal posture change.
8. Fall contains a pre-event, rapid rotation/impact and post-event settling.
9. Quaternion output is normalized after generation.

## Validation sequence

Run these in order from the repository root:

```powershell
python tools\generate_bno055_synthetic_v2.py --subjects 20 --sessions 2 --duration 15 --seed 42
python tools\audit_bno055_synthetic_v2.py
python tools\build_bno055_windows_v2.py
python tools\visualize_bno055_synthetic_v2.py
```

Do not use a V2 ML result as evidence of real hardware performance.
Real BNO055 measurements remain the hardware-domain validation target.
