# SafeBand AI — Project Progress & Continuity Record

> **Project:** SafeBand AI — *An Intelligent Safety Band Using Sensor Fusion and Artificial Intelligence*  
> **Purpose:** Preserve the verified engineering/research history so future development sessions can continue without restarting or guessing previous work.

## 1. Project Vision

SafeBand AI is a wearable safety system combining multiple sensors with AI to recognize potentially unsafe user situations and trigger an appropriate response.

```text
Multiple Sensors
      │
      ├── IMU / Motion
      ├── PPG / MAX30102
      ├── Environmental sensing
      ├── Audio (optional/experimental)
      └── GPS
             │
             ▼
   Sensor-specific processing
             │
             ▼
       Sensor Fusion
             │
             ▼
         Risk Engine
             │
       ┌─────┼─────┐
       ▼     ▼     ▼
     SAFE  WARNING EMERGENCY
                     │
                     ▼
                 GPS + SMS
```

The project is best described as a **multimodal / multi-sensor AI system**, rather than one monolithic model.

## 2. Current AI Development Status

| Modality / Sensor | Software | AI training / benchmark | Status |
|---|---:|---:|---|
| IMU / accelerometer | Yes | **Yes — V5** | Mature AI branch |
| PPG / wrist BVP | Yes | **Yes — V3** | Mature reference branch |
| MAX30102 | Yes | No real hardware-domain training yet | Validation/domain-transfer stage |
| BNO055 / richer IMU | Yes | Not independently benchmarked | Future extension |
| BME680 | Yes | No | Sensor integration only |
| INMP441 | Yes | No | Future AI modality |
| GPS | Yes | Not an ML target | Position + emergency communication |
| Multimodal sensor fusion | Yes, architecturally | Final fused model not trained | Major remaining stage |
| Final risk engine | Yes | Not finalized as learned multimodal model | Major remaining stage |

**Important:** A sensor driver being present does not mean an AI model has been trained for that sensor.

# 3. IMU / Fall Detection Branch

## Objective

Build fall detection from motion data and evaluate generalization across datasets.

## Datasets

### BITS2

Processed fall/non-fall window data is stored locally as:

```text
datasets/processed/bits2/fall_windows_v6.csv
```

### SisFall

Preparation command:

```powershell
python tools/prepare_sisfall.py `
  --zip datasets/raw/SisFall/SisFall_dataset.zip `
  --out datasets/processed/sisfall_windows.csv `
  --sensor adxl
```

Approximately **47,683 windows** and **4,505 recordings** were processed.

Raw datasets are not intended for GitHub.

# 4. IMU Model Evolution

## V4 — Event / Temporal Pipeline

The fall branch evolved toward event-oriented detection instead of triggering from one isolated window.

Key ideas:

- temporal confirmation,
- hit counting,
- tolerance for gaps,
- recording/event-level evaluation,
- cross-dataset evaluation.

## V5 — Balanced Multidataset Model

Main files:

```text
tools/train_multidataset_v5_balanced.py
ai/ml_safety_v5.py
docs/MULTIDATASET_V5.md
```

V5 introduced:

- equal dataset weighting,
- equal class weighting,
- multiple model comparison,
- validation-selected operating point,
- optional hard-negative mining,
- temporal event confirmation.

## V5 baseline

Command:

```powershell
python tools/train_multidataset_v5_balanced.py `
  --bits2 datasets/processed/bits2/fall_windows_v6.csv `
  --sisfall datasets/processed/sisfall_windows.csv `
  --out models/multidataset_v5_event
```

### BITS2

- Recordings: 144
- Falls: 64
- Non-falls: 80
- Accuracy: 0.875
- Balanced accuracy: 0.875
- Precision: 0.8485
- Recall: 0.875
- F1: 0.8615
- FPR: 0.125
- False alarms: 10
- Missed falls: 8

### SisFall

- Recordings: 931
- Falls: 375
- Non-falls: 556
- Accuracy: 0.9828
- Balanced accuracy: 0.9817
- Precision: 0.9812
- Recall: 0.9760
- F1: 0.9786
- FPR: 0.01259
- False alarms: 7
- Missed falls: 9

### Macro

- Accuracy: 0.9289
- Balanced accuracy: 0.92835
- Precision: 0.91486
- Recall: 0.9255
- F1: 0.92007
- FPR: 0.06879

### Pooled event result

- Threshold: 0.75
- Minimum hits: 3
- Maximum gap: 120 samples
- Accuracy: 0.96837
- Balanced accuracy: 0.96727
- Precision: 0.96128
- Recall: 0.96128
- F1: 0.96128
- FPR: 0.02673
- False alarms: 17
- Missed falls: 17

# 5. IMU V5 Hard-Negative Experiment

Command:

```powershell
python tools/train_multidataset_v5_balanced.py `
  --bits2 datasets/processed/bits2/fall_windows_v6.csv `
  --sisfall datasets/processed/sisfall_windows.csv `
  --out models/multidataset_v5_hardneg `
  --hard-negative
```

### SisFall after hard-negative mining

- Recall: 0.97867
- F1: 0.98260
- FPR: 0.00899
- False alarms: 5
- Missed falls: 8

### Macro

- Recall: 0.92683
- F1: 0.92207
- FPR: 0.066996

### Pooled event result

- Threshold: 0.8
- Minimum hits: 2
- Maximum gap: 30 samples
- Accuracy: 0.97116
- Balanced accuracy: 0.96998
- Precision: 0.96575
- Recall: 0.96355
- F1: 0.96465
- FPR: 0.02358
- False alarms: 15
- Missed falls: 16

The hard-negative experiment improved SisFall false-alarm behavior. BITS2 remains the more difficult domain and should remain visible in the research record.

# 6. PPG / Heart-Rate Branch

## Objective

Estimate heart rate from wrist optical PPG/BVP while using wrist motion as contextual signal-quality information.

PPG-DaLiA is used as a research/reference benchmark with ECG-derived HR as the target.

## Critical sensor-domain distinction

PPG-DaLiA uses **Empatica E4 wrist BVP**, not MAX30102.

Therefore:

```text
PPG-DaLiA / E4 BVP ≠ MAX30102 PPG
```

The project does **not** claim E4-trained performance as MAX30102 validation.

Intended progression:

```text
PPG-DaLiA / E4
      ↓
Research benchmark
      ↓
Real MAX30102 recordings
      ↓
Hardware-domain validation
      ↓
Optional calibration / fine-tuning
```

# 7. PPG Development History

## Stage 1

Established the PPG-DaLiA benchmark methodology:

- ECG-derived HR target,
- subject-independent split,
- no use of E4 HR.csv/IBI.csv as ground truth,
- documented resampling/normalization,
- classical baselines before deep learning,
- motion/context quality analysis.

Baseline approaches:

- BVP-only FFT HR,
- BVP-only peak/IBI HR,
- classical ML using PPG + motion-quality features.

## V1

First classical ML HR estimator:

- deterministic subject-level 60/20/20 split,
- no subject leakage,
- Random Forest regression,
- signal-derived HR candidates,
- wrist-motion quality features,
- held-out test evaluation.

## V2

Expanded feature representation:

- normalized PPG morphology,
- spectral localization,
- autocorrelation,
- peak/IBI quality,
- wrist-accelerometer motion features.

Models compared:

- Ridge,
- Random Forest,
- Extra Trees,
- HistGradientBoosting.

## V3

Current PPG reference model.

Training script:

```text
train_ppgdalia_hr_model_v3.py
```

Feature entry point:

```text
ai.hr_estimator.make_features
```

Models compared:

- Ridge,
- Random Forest,
- Extra Trees,
- HistGradientBoosting.

Selected model:

**Extra Trees**

Validation-only affine calibration was tested and disabled because it worsened validation MAE. The final estimator was refit on train + validation; held-out test subjects were not used for selection.

# 8. PPG-DaLiA V3 Dataset / Split

- Total windows: **64,697**
- Subjects: **15**

Subject-level split:

```text
Train:      S1, S2, S3, S10, S11, S12, S13, S14, S15
Validation: S4, S5, S6
Test:       S7, S8, S9
```

Window counts:

- Train: 39,872
- Validation: 11,843
- Test: 12,982

Subject overlap: **false**

# 9. PPG V3 Validation Results

| Model | MAE (BPM) | RMSE | R² | ±3 BPM | ±5 BPM | ±10 BPM | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 14.73 | 20.39 | 0.342 | 13.58% | 24.45% | 47.94% | -12.26 |
| Random Forest | 11.86 | 19.44 | 0.403 | 37.79% | 49.65% | 65.35% | -9.73 |
| Extra Trees | **11.77** | **19.35** | **0.408** | 37.66% | 49.65% | 65.06% | -9.77 |
| HGB | 12.33 | 19.80 | 0.380 | 34.25% | 46.38% | 63.35% | -10.11 |

Extra Trees was selected by validation MAE.

# 10. PPG V3 Held-Out Test

Final selected Extra Trees model:

- Test windows: **12,982**
- MAE: **7.707 BPM**
- RMSE: **12.469 BPM**
- R²: **0.5334**
- Within ±3 BPM: **43.76%**
- Within ±5 BPM: **56.75%**
- Within ±10 BPM: **74.44%**
- Bias: **+3.98 BPM**

Final artifact:

```text
models/ppgdalia_hr_e4_reference_v3.joblib
```

## Important interpretation

This is a valid subject-independent **E4 BVP reference benchmark**.

It is **not yet a MAX30102-validated production result**.

# 11. Existing Sensor Software

Current sensor modules:

```text
sensors/
├── bme680.py
├── bno055.py
├── gps.py
├── inmp441.py
└── max30102.py
```

Sensor interface presence and AI training are documented separately.

# 12. Final Intended AI Architecture

The project does not need to force every sensor into one giant model.

A hierarchical multimodal architecture is more appropriate:

```text
                 SAFE BAND
                     │
       ┌─────────────┼─────────────┐
       │             │             │
      IMU           PPG        Environment
       │             │             │
       ▼             ▼             ▼
  Fall/Activity    HR/Quality   Environment
     Model           Model        Features
       │             │             │
       └─────────────┼─────────────┘
                     │
              SENSOR FUSION
                     │
                     ▼
                RISK ENGINE
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        SAFE      WARNING    EMERGENCY
                                  │
                         ┌────────┴────────┐
                         ▼                 ▼
                        GPS               SMS
```

Each modality can use a model appropriate to its signal. The fusion layer then combines the outputs to estimate overall safety/risk.

# 13. Completed vs Remaining

## Completed / strong

1. IMU fall-detection AI branch
2. BITS2 + SisFall cross-dataset evaluation
3. Temporal fall-event logic
4. Balanced V5 training
5. Hard-negative V5 experiment
6. PPG-DaLiA preparation and methodology
7. PPG Stage 1 → V1 → V2 → V3
8. Subject-independent PPG evaluation
9. PPG V3 Extra Trees reference model
10. Sensor software interfaces
11. Streamlit/software demonstration infrastructure

## In progress

1. Real MAX30102 recording
2. MAX30102-domain PPG validation
3. Multi-sensor integration
4. Sensor-fusion logic
5. Final risk-state architecture

## Not yet trained as dedicated AI branches

1. BME680-specific AI
2. INMP441-specific AI
3. Dedicated BNO055 model
4. Final learned multimodal fusion model

# 14. Repository Documentation Strategy

Raw datasets should remain outside GitHub when large, restricted, or unnecessary for repository distribution.

Recommended documentation structure:

```text
docs/
├── PROJECT_PROGRESS.md
├── DATASETS.md
├── EXPERIMENT_LOG.md
├── IMU_FALL_DETECTION.md
├── MULTIDATASET_V5.md
├── PPGDALIA_MAX30102_STAGE1.md
├── PPGDALIA_V1.md
├── PPGDALIA_V2.md
├── PPGDALIA_V3.md
├── MAX30102_VALIDATION.md
├── SENSOR_FUSION.md
└── ARCHITECTURE.md
```

For each dataset, document:

- source,
- citation/link,
- expected directory structure,
- download instructions,
- preprocessing command,
- generated format,
- dataset statistics,
- split methodology,
- known limitations.

Preserve:

```text
models/   → model artifacts and benchmark JSON
tools/    → reproducible preprocessing/training/evaluation
ai/       → reusable model/inference logic
docs/     → experiment history and decisions
```

# 15. Research Rules

1. Never call E4 data MAX30102 data.
2. Keep subject separation strict for physiological ML.
3. Never present smoke-test metrics as final research results.
4. Preserve failed experiments and explain why later versions changed.
5. Keep model selection separate from the final held-out test.
6. Do not call sensor integration “AI training.”
7. Do not add a model solely to increase the number of AI models; every model should have a measurable safety/research purpose.

# 16. Future-Session Continuity Checkpoint

When continuing SafeBand AI in a new development session, start from this checkpoint.

```text
IMU:
    V5 hard-negative experiment completed
    Cross-dataset evaluation completed
    Strong pooled event-level performance
    BITS2 remains the harder domain

PPG:
    Stage 1 → V1 → V2 → V3 completed
    Extra Trees selected
    Held-out MAE = 7.707 BPM
    E4 reference sensor only
    MAX30102 validation still required

Other sensors:
    Drivers/modules exist
    Dedicated AI training not completed

Architecture:
    Sensor-specific AI branches exist
    Final multimodal sensor-fusion/risk layer remains
```

**Do not restart the IMU or PPG pipelines from scratch. Continue from this checkpoint.**

## Project Status in One Sentence

> **SafeBand AI currently has two experimentally developed AI branches — IMU-based fall detection and PPG-based heart-rate estimation — and is transitioning from individual sensor intelligence toward real MAX30102 hardware validation and multimodal sensor fusion.**
