# SafeBand AI — BNO055 Synthetic V2

V2 is a deliberately harder synthetic dataset than V1. It is intended for:
- software pipeline integration
- deterministic window generation
- preliminary ML experimentation
- testing the future BNO055 feature/model interface

It is **not** evidence of real BNO055 hardware performance.

## Main changes from V1

- subject-specific wrist orientation and sensor bias
- session-level magnetic environment variation
- magnetometer generated from a world magnetic vector and sensor orientation
- low-frequency environmental magnetic drift
- nonzero stationary micro-motion
- variable gait frequency/amplitude/phase
- multiple motion harmonics and cross-axis coupling
- explicit sit-to-stand / stand-to-sit temporal transitions
- explicit fall sequence: pre-event → rapid rotation/impact → post-event settling
- normalized quaternions derived from the generated orientation

## Generate

From repository root:

```powershell
python tools\generate_bno055_synthetic_v2.py --subjects 20 --sessions 2 --duration 15 --seed 42
```

## Audit

```powershell
python tools\audit_bno055_synthetic_v2.py
```

## Build 2-second raw9 windows

```powershell
python tools\build_bno055_windows_v2.py
```

Expected default window shape is `(N, 9, 200)`.

## Interpretation rule

A high score on V2 is still a synthetic benchmark result. Real-device validation must use synchronized BNO055 measurements from the SafeBand hardware.
