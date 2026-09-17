# SafeBand Activity ACC+GYRO V1

This is the next activity-recognition experiment after the ACC-only DL work.

## Why this batch exists

The ACC-only experiments consistently struggled with RESTING/SITTING. The BITS-2
raw ingestion record documents accelerometer, linear acceleration, gyroscope,
heart-rate and magnetometer sensor blocks, while preserving timestamps and
recording/subject identifiers. This batch therefore introduces the gyroscope
without changing the subject-independent evaluation protocol.

## Pipeline

1. `audit_bits2_imu_activity.py`
   - validates the actual canonical CSV schema
   - checks source-file coverage
   - checks ACC/GYRO availability

2. `build_activity_accgyro_windows.py`
   - uses the existing `activity_windows.csv` window metadata
   - extracts the corresponding ACC samples
   - matches/interpolates gyro onto the ACC timestamps
   - outputs `(N,6,40)` = ax, ay, az, gx, gy, gz
   - records coverage and skipped-window reasons

3. `train_activity_accgyro_v1.py`
   - lightweight 6-channel CNN
   - 5-fold StratifiedGroupKFold by subject
   - training-only channel normalization
   - grouped inner validation for early stopping
   - untouched outer test folds

## IMPORTANT

Do not run training before the audit and builder.

Run from repo root:

    python tools/audit_bits2_imu_activity.py

Then:

    python tools/build_activity_accgyro_windows.py

Then:

    python tools/train_activity_accgyro_v1.py

## Data integrity

The builder does not invent a resampling clock. It uses the timestamp values
present in `bits2_canonical_long.csv` and interpolates gyro values only onto
the timestamps of the already-defined ACC window.

The existing activity labels and window boundaries are reused rather than
reconstructed from guessed label mappings.

## Failure handling

The scripts fail loudly when:
- expected columns are absent
- labels are unknown
- source files are missing
- tensor dimensions are wrong
- X/y/group lengths differ
- required ACC/GYRO coverage is unavailable

This is deliberate: a missing schema must not be silently "fixed" by guessing.
