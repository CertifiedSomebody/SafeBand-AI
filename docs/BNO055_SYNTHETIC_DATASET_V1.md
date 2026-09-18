# SafeBand AI — BNO055 Synthetic Sensor Dataset V1

## Purpose

This package creates a **software-integration/test dataset** that follows the
measurement semantics and units documented for the Bosch BNO055.

It is intentionally not presented as real sensor data and must not be used as
hardware-validation evidence.

The BNO055 datasheet describes the device as integrating a 14-bit accelerometer,
16-bit gyroscope, geomagnetic sensor and an internal Cortex-M0+ running sensor
fusion software. It supports I2C/UART and exposes raw sensor data plus fused
outputs such as Euler angles, quaternions, linear acceleration and gravity.

## Datasheet-grounded configuration

For this synthetic trial we use a 100 Hz stream. The datasheet's fusion output
table specifies 100 Hz output for accelerometer and gyroscope data and 100 Hz
fusion data in the NDOF mode; the IMU mode also provides 100 Hz accelerometer,
gyroscope and fusion output. The exact hardware configuration should be fixed
when the real prototype is assembled.

Engineering units used here:

| Signal | Dataset unit |
|---|---|
| Accelerometer X/Y/Z | m/s² |
| Gyroscope X/Y/Z | °/s |
| Magnetometer X/Y/Z | µT |
| Euler heading/roll/pitch | degrees |
| Quaternion W/X/Y/Z | unitless |
| Linear acceleration X/Y/Z | m/s² |
| Gravity X/Y/Z | m/s² |
| Temperature | °C |

The datasheet states that acceleration can be represented in m/s² or mg,
angular rate in degrees/s or rad/s, Euler angles in degrees or radians, and
quaternion values use a 2^14 LSB representation at the register level.
This project stores engineering units rather than raw register LSB values.

## Dataset schema

Each row represents one 100 Hz timestamp.

Required metadata:

- `subject_id`
- `session_id`
- `timestamp_s`
- `activity_label`

Sensor/fusion fields:

- `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`
- `gyro_x_dps`, `gyro_y_dps`, `gyro_z_dps`
- `mag_x_uT`, `mag_y_uT`, `mag_z_uT`
- `euler_heading_deg`, `euler_roll_deg`, `euler_pitch_deg`
- `quat_w`, `quat_x`, `quat_y`, `quat_z`
- `linear_accel_x_mps2`, `linear_accel_y_mps2`, `linear_accel_z_mps2`
- `gravity_x_mps2`, `gravity_y_mps2`, `gravity_z_mps2`
- `temperature_c`

## Activity scenarios

V1 supports:

- SITTING
- STANDING
- LYING
- WALKING
- RUNNING
- STAIRS
- SIT_TO_STAND
- STAND_TO_SIT
- FALL

Fall remains a separate emergency-detection target in the SafeBand
architecture. The label is included here so the complete sensor stream can be
used for software integration testing.

## How the synthetic signal is constructed

The generator creates scenario-specific motion and orientation patterns,
then produces:

1. gravity in the sensor frame;
2. linear acceleration;
3. total accelerometer output = gravity + linear acceleration + noise/bias;
4. angular velocity;
5. magnetic-field measurements from a fixed nominal Earth field;
6. Euler angles and quaternion orientation;
7. linear acceleration and gravity-vector outputs;
8. temperature variation.

Subject-specific bias and small session variations are included so every
subject is not an exact copy.

The generator deliberately avoids claiming that these distributions reproduce
the true BNO055 noise spectrum, calibration behavior, or human biomechanics.

## Important scientific limitation

This dataset is **synthetic**.

It can test:

- CSV/schema handling;
- timestamp handling;
- windowing;
- feature extraction;
- model input pipelines;
- sensor-fusion software;
- risk-engine logic;
- end-to-end software interfaces.

It cannot establish:

- real BNO055 accuracy;
- real sensor noise characteristics in the assembled device;
- real user activity-recognition performance;
- real fall-detection performance;
- deployment/generalization performance.

Real hardware recordings remain necessary for those claims.

## Generate

From the SafeBand repository root:

```powershell
python tools\generate_bno055_synthetic.py
```

Default output:

```text
datasets\synthetic\bno055\bno055_synthetic_v1.csv
```

Default dataset:

- 8 subjects
- 2 sessions/subject
- 9 activities
- 15 seconds/activity
- 100 Hz
- 216,000 rows

For a smaller quick test:

```powershell
python tools\generate_bno055_synthetic.py --subjects 2 --sessions 1 --duration 5
```

To generate only stationary/dynamic activity data:

```powershell
python tools\generate_bno055_synthetic.py --activities SITTING,STANDING,LYING,WALKING,RUNNING,STAIRS
```

## Audit

```powershell
python tools\audit_bno055_synthetic.py
```

The audit checks the required schema, numeric/finite values, activity counts,
subject/session coverage and approximate 100 Hz timing.

## Next use

Do not immediately train a final classifier on this dataset.

The first intended use is to connect this telemetry to the SafeBand software
pipeline and verify:

```text
BNO055-like stream
      ↓
windowing
      ↓
activity / fall inference interfaces
      ↓
sensor-fusion inputs
      ↓
risk engine
      ↓
SAFE / WARNING / EMERGENCY
```

After that, synthetic scenarios can be expanded to test edge cases and fusion
logic. When the actual BNO055 hardware is available, the same schema should be
used as closely as practical so real recordings can replace synthetic streams
without redesigning the software interfaces.

## Source

Primary source supplied for this project:

**Bosch Sensortec — BNO055 Data Sheet, revision 1.8, October 2021,
BST-BNO055-DS000-18.**

Relevant sections:

- pp. 2–4: integrated sensors and interfaces
- p. 32: unit selection and fusion output format
- p. 33: fusion output data rates
- pp. 35–38: sensor and fused output data formats
