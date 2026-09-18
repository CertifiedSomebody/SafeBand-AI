# SafeBand BNO055 ML Benchmark V1

## Purpose

Benchmark the frozen BNO055 Synthetic V2.1 window artifact using subject-independent
evaluation and controlled sensor-channel ablations.

This is a **software/ML benchmark only**. Synthetic results must not be presented
as real BNO055 hardware performance.

## Input

```text
datasets/synthetic/bno055/bno055_windows_v2_1.npz
```

Expected:

```text
X = (N, 9, 200)
```

Channels:

1. ACC X
2. ACC Y
3. ACC Z
4. GYRO X
5. GYRO Y
6. GYRO Z
7. MAG X
8. MAG Y
9. MAG Z

## Models

- RF: handcrafted temporal/statistical features + Random Forest
- CNN: small 1-D CNN
- TCN: residual dilated temporal CNN

## Ablations

- `acc`
- `accgyro`
- `accgyromag`

## Evaluation

`StratifiedGroupKFold` is grouped by `subject_id`.

Normalization for deep learning is computed on each outer training fold only.
The test fold remains untouched until final prediction.

Primary metrics:

- accuracy
- balanced accuracy
- macro-F1
- per-class F1
- per-class recall
- confusion matrix

Event-overlap windows are additionally reported separately.

## Commands

Run from SafeBand repository root.

### 1. Preflight

```powershell
python tools\bno055_ml_preflight.py
```

### 2. Classical baseline first

```powershell
python tools\train_bno055_ml_v1.py --model rf --channels acc
python tools\train_bno055_ml_v1.py --model rf --channels accgyro
python tools\train_bno055_ml_v1.py --model rf --channels accgyromag
```

### 3. CNN

```powershell
python tools\train_bno055_ml_v1.py --model cnn --channels acc
python tools\train_bno055_ml_v1.py --model cnn --channels accgyro
python tools\train_bno055_ml_v1.py --model cnn --channels accgyromag
```

### 4. TCN

```powershell
python tools\train_bno055_ml_v1.py --model tcn --channels acc
python tools\train_bno055_ml_v1.py --model tcn --channels accgyro
python tools\train_bno055_ml_v1.py --model tcn --channels accgyromag
```

Run these as separate checkpoints rather than all at once.

## Output

```text
reports/bno055_ml_v1/
├── rf_acc.json
├── rf_accgyro.json
├── rf_accgyromag.json
├── cnn_acc.json
├── cnn_accgyro.json
├── cnn_accgyromag.json
├── tcn_acc.json
├── tcn_accgyro.json
└── tcn_accgyromag.json
```


### 5. Summarize completed runs

```powershell
python tools\summarize_bno055_ml_v1.py --output reports\bno055_ml_v1_summary.csv
```

The summary is descriptive only; it does not rank or select a model.
