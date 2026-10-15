# AICOS-HAR — V3 Path and Training Package

This document records the V3 correction.

The V2 parser produced zero raw sensor files. This was a parser bug, not a
dataset failure. The documented AICOS-HAR hierarchy uses `Sensor.txt`, and
the original forensic audit found 18,587 files with:
- ACC 5,231
- gyro 4,899
- mag 4,744
- baro 3,713.

V3 recognizes both canonical long sensor filenames and short aliases.

## Commands

### 1. Reconcile paths

```powershell
python tools\audit_aicos_har_path_v3.py --data-dir datasets\raw\AICOS-HAR\AICOS-HAR --metadata datasets\raw\AICOS-HAR\AICOS-HAR\metadata.csv --inventory datasets\processed\aicos_har\aicos_har_acquisition_inventory.csv --out reports\aicos_har_path_reconciliation_v3
```

### 2. Bounded sampling

Only after Stage 1 succeeds:

```powershell
python tools\audit_aicos_har_sampling_v2.py --data-dir datasets\raw\AICOS-HAR\AICOS-HAR --max-acquisitions 40 --out reports\aicos_har_sampling_v2
```

### 3. Build ACC windows

Default baseline:

```powershell
python tools\build_aicos_har_windows_v1.py --data-dir datasets\raw\AICOS-HAR\AICOS-HAR --metadata datasets\raw\AICOS-HAR\AICOS-HAR\metadata.csv --out datasets\processed\aicos_har\aicos_acc_50hz_2s.npz --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv
```

### 4. Classical benchmark

```powershell
python tools\train_aicos_har_classical_v1.py --npz datasets\processed\aicos_har\aicos_acc_50hz_2s.npz --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv --out models\aicos_har\classical_v1
```

### 5. Tiny CNN benchmark

```powershell
python tools\train_aicos_har_tinycnn_v1.py --npz datasets\processed\aicos_har\aicos_acc_50hz_2s.npz --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv --out models\aicos_har\tinycnn_v1
```

Do not execute stages 3–5 until the preceding audit has been reviewed.
