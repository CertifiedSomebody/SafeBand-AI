# AICOS-HAR V4 runbook

Run one stage at a time from the repository root.

1. `audit_aicos_har_path_v4.py`
2. `audit_aicos_har_sampling_v4.py`
3. `build_aicos_har_windows_v4.py`
4. `train_aicos_har_classical_v4.py`
5. `train_aicos_har_tinycnn_v4.py`

The first command is the mandatory gate. If it reports anything other than zero unresolved metadata/raw acquisitions after aliasing, do not build windows or train.

Reference values from the forensic audit are 5,231 unique acquisitions and 18,587 raw sensor files, with 5,231 ACC, 4,899 gyro, 4,744 magnetometer and 3,713 barometer files. The V4 path gate should reconcile the Lift/Elevator naming convention instead of treating those acquisitions as missing.
