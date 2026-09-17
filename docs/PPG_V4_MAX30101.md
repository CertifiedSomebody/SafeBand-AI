# SafeBand AI — PPG V4: MAX30101-Domain HR Estimation

## Purpose

PPG V4 extends the PPG-DaLiA/E4 reference pipeline toward the optical and motion domain of the project's planned MAX30102 hardware by using the PhysioNet Pulse Transit Time PPG Dataset.

This version is a **MAX30101-domain reference model**, not MAX30102 validation. Real MAX30102 recordings will be used later for hardware-domain validation and calibration.

## Why this version exists

PPG-DaLiA remains an important benchmark, but it uses Empatica E4 wrist BVP. The PTT dataset provides multi-wavelength PPG recorded with Maxim MAX30101 sensors, which is a closer optical hardware family to MAX30102.

The PTT README documents two MAX30101 sensors with red, infrared and green wavelengths, two finger measurement sites, 500 Hz distributed PPG channels, MPU-9250 motion sensing and ECG R-peak annotations. It also documents SpO2 only at the start and end of each activity. Therefore PTT is suitable for an HR model and optical-domain feature development, but its sparse SpO2 numerics must not be broadcast across every waveform window as continuous ground truth.

## Channel mapping

| Feature name | PTT column | Meaning |
|---|---|---|
| distal_red | pleth_1 | distal red |
| distal_ir | pleth_2 | distal infrared |
| distal_green | pleth_3 | distal green |
| proximal_red | pleth_4 | proximal red |
| proximal_ir | pleth_5 | proximal infrared |
| proximal_green | pleth_6 | proximal green |

The PTT README identifies these six channels and their optical locations/wavelengths.

## Pipeline

```text
PTT raw CSV
   ↓
6-channel MAX30101 PPG + MPU-9250 ACC/GYRO
   ↓
8 s windows / 2 s shift
   ↓
ECG R peaks → HR target only
   ↓
MAX30101-aware feature extraction
   ├─ waveform statistics
   ├─ 0.5–5 Hz morphology/spectral features
   ├─ PPG peak HR candidates
   ├─ AC/DC representation features
   ├─ red/IR/green correlations
   ├─ distal/proximal correlations
   └─ motion + PPG/motion coupling
   ↓
subject-level 60/20/20 split
   ↓
Ridge / RF / Extra Trees / HGB
   ↓
validation MAE model selection
   ↓
optional validation-only affine calibration
   ↓
held-out test
```

## Important leakage rule

ECG is **not** a model input. The ECG `peaks` column is used only to derive the HR regression target for each window. This prevents the model from receiving a direct cardiac timing signal during inference.

## SpO2 handling

PTT provides `spo2_start` and `spo2_end` participant/activity numerics. V4 does **not** treat these as continuous window-level labels. Future SpO2 work requires a dataset with appropriately timed reference measurements or a separate calibration protocol.

The `ac_dc` features in V4 are optical representation features only. They are **not** a clinical SpO2 equation.

## Commands

From the SafeBand repository root:

```powershell
python tools/prepare_ptt_hr_v4.py `
  --ptt-root datasets/raw/PTT `
  --out datasets/processed/ptt/ppg_v4_features.csv
```

Then:

```powershell
python tools/train_ppg_v4_ptt.py `
  --input datasets/processed/ptt/ppg_v4_features.csv `
  --model-out models/ppg_v4/ppg_v4_max30101.joblib `
  --report-out models/ppg_v4/ppg_v4_max30101_report.json
```

Prediction requires three NumPy arrays representing one window:

```powershell
python tools/predict_ppg_v4.py `
  --model models/ppg_v4/ppg_v4_max30101.joblib `
  --ppg-npy ppg.npy `
  --acc-npy acc.npy `
  --gyro-npy gyro.npy
```

## Research split

The split is deterministic and subject-level. Subjects never cross train, validation or test partitions. The validation set selects the model and whether affine calibration is useful. The final held-out subjects remain untouched until final evaluation.

## Expected interpretation

The key question for V4 is not whether PTT performance is automatically a MAX30102 result. The question is whether a model trained on a MAX30101-family multi-wavelength domain gives a stronger, more hardware-relevant starting point for later MAX30102 validation than the E4-only reference.

Final progression:

```text
PPG-DaLiA / E4
      ↓
PPG V3 reference
      ↓
PTT / MAX30101
      ↓
PPG V4 MAX30101-domain reference
      ↓
real MAX30102 recordings
      ↓
hardware-domain validation
      ↓
validation-based calibration / fine-tuning
```

## Status

- V3 E4 reference: completed.
- PTT/MAX30101 acquisition: completed.
- PTT audit: completed.
- V4 implementation: this document and associated scripts.
- V4 training: pending local execution on the full downloaded PTT dataset.
- MAX30102 calibration/validation: pending real sensor recordings.
