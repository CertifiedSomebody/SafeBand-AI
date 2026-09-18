# SAFEBAND AI 🛡️

## Intelligent AI-Based Safety Monitoring Wearable

SAFEBAND AI is a final-year engineering project for a wearable safety-monitoring system that combines **motion, physiological, environmental and acoustic sensing**, AI-based interpretation, sensor fusion and emergency communication.

The project proposal defines the final system around an ESP32-S3, BNO055, MAX30102, BME680, INMP441, GPS/GNSS and EC200U cellular communication. The software is deliberately modular so Phase 1 AI work can be completed before the physical hardware is ready.

---

## 1. The Actual System Goal

The project is **not** a collection of independent sensor classifiers.

The intended pipeline is:

```text
REAL SENSORS
    │
    ├── BNO055 ──→ Activity / Fall Intelligence
    ├── MAX30102 → Physiological Intelligence
    ├── BME680 ──→ Environmental Intelligence
    ├── INMP441 ─→ Acoustic Intelligence
    └── GPS ─────→ Location
                     │
                     ↓
              SENSOR FUSION
                     ↓
                RISK ENGINE
                     ↓
          SAFE / WARNING / EMERGENCY
                     │
                     ↓
              GPS + EC200U
                     │
                     ↓
             Caregiver / Cloud
```

### Phase 1 — Current software/AI responsibility

Build and validate the individual intelligence modules and make their interfaces hardware-ready.

### Phase 2 — Hardware/domain integration

When the hardware stack is available:

- collect synchronized SafeBand data
- validate each model on the actual sensors
- calibrate sensor-specific preprocessing
- develop and validate multimodal fusion
- deploy suitable models on the ESP32-S3
- validate the complete emergency/communication pipeline

---

## 2. Phase 1 Status

| Branch | Status |
|---|---|
| BITS-2/SisFall fall detection | ✅ Multidataset benchmark + hard-negative/event pipeline |
| PPG-DaLiA E4 HR reference | ✅ Frozen reference |
| PTT/MAX30101 HR reference | ✅ V4.3 frozen |
| Walking PPG stress experiment | ✅ V5.3 frozen |
| BME680/AIRWISE IAQ | ✅ V1 frozen; V2.1 not retained as improvement |
| Activity recognition | ✅ BITS2/FORTH/PAMAP2 + BNO055 synthetic benchmarks established |
| BNO055 synthetic dataset | ✅ V2.1 frozen and audited |
| BNO055 ML benchmark | ✅ RF + ACC/GYRO baseline; classical ML/DL comparison complete |
| INMP441 audio intelligence | ✅ Nonspeech7k V1/V2 benchmark + CNN benchmark + best-fit HGB model frozen |
| Common sensor contracts | 🟡 Being hardened for hardware integration |
| Sensor fusion | 🟡 Software prototype; multimodal validation requires synchronized SafeBand data |
| Risk engine | 🟡 Prototype; final validation requires real hardware |
| ESP32-S3 deployment | ⏳ Phase 2 |
| Real BNO055 validation | ⏳ Phase 2 |
| Real MAX30102 validation | ⏳ Phase 2 |
| Real BME680 validation | ⏳ Phase 2 |
| Real INMP441 validation | ⏳ Phase 2 |
| GPS + EC200U integration | ⏳ Phase 2 |
| MAX30205 body-temperature sensor | ❌ Removed from final PCB due to reliability/measurement concerns |

---

## 3. Important Research Positioning

### BME680

AIRWISE is a **BME680-native indoor environmental benchmark**. V1 uses 24 engineered features and a chronological 60/20/20 split within location.

The frozen held-out test result is:

- Accuracy: **91.40%**
- Balanced accuracy: **79.47%**
- Macro precision: **77.37%**
- Macro recall: **79.47%**
- Macro F1: **78.24%**

This is an IAQ classification benchmark, not a clinical or personal emergency classifier.

### PPG

The PPG branches deliberately distinguish sensor domains:

- PPG-DaLiA = Empatica E4 BVP reference
- PTT = MAX30101-domain reference
- SafeBand target = MAX30102 hardware

Neither E4 nor MAX30101 results are presented as MAX30102 validation.

### Walking PPG

The Wrist PPG During Exercise experiment is retained as a cross-subject stress test. Its poor generalization is treated as evidence of domain/subject shift rather than hidden through repeated benchmark tuning.

---

## 4. Activity Recognition

Activity recognition has now been investigated across multiple public and synthetic datasets. It is treated as a **sensor-specific evidence source**, not the final SafeBand safety decision.

### BITS-2

Reference activity set:

```text
RESTING
SITTING
WALKING
RUNNING
```

Strongest tested raw-window Tiny CNN result:

- Accuracy: **83.02%**
- Balanced accuracy: **81.09%**
- Macro-F1: **80.97%**

The recurring weakness is stationary-state ambiguity. `FALL` remains a separate binary emergency detector.

### FORTH-TRACE

Locked four-class benchmark:

```text
SITTING
STAIRS
STANDING
WALKING
```

RBF-SVM result:

- Accuracy: **98.20%**
- Balanced accuracy: **98.31%**
- Macro-F1: **98.19%**

Cross-domain BITS-2 ↔ FORTH-TRACE testing demonstrated substantial domain shift, so within-dataset accuracy is not treated as universal deployment performance.

### PAMAP2 stationary benchmark

Hand/wrist ACC+GYRO benchmark for:

```text
LYING
SITTING
STANDING
```

Best completed pooled result:

- Accuracy: **76.05%**
- Balanced accuracy: **76.07%**
- Macro-F1: **76.02%**

Large subject variability reinforced the need for subject-independent evaluation and real SafeBand data.

### BNO055 synthetic benchmark

Frozen V2.1 benchmark:

```text
20 subjects
40 sessions
9 classes
100 Hz
5040 windows
9 × 200 representation
```

Classes:

```text
FALL
LYING
RUNNING
SITTING
SIT_TO_STAND
STAIRS
STANDING
STAND_TO_SIT
WALKING
```

Random Forest sensor ablation:

| Input | Accuracy | Macro-F1 |
|---|---:|---:|
| ACC | 69.38% | 69.27% |
| **ACC + GYRO** | **72.54%** | **72.51%** |
| ACC + GYRO + MAG | 72.10% | 71.87% |

Classical ML on ACC+GYRO:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Random Forest | **72.54%** | 72.51% |
| HistGradientBoosting | 72.50% | **72.54%** |
| Extra Trees | 71.53% | 71.40% |
| Logistic Regression | 71.13% | 70.85% |
| RBF SVM | 71.07% | 71.02% |
| Linear SVM | 70.08% | 69.54% |
| Decision Tree | 67.02% | 67.11% |
| TCN | 63.27% | 62.99% |
| CNN | 61.03% | 60.76% |
| k-NN | 59.74% | 60.01% |

Current synthetic BNO055 baseline: **ACC + GYRO → engineered features → Random Forest**. HistGradientBoosting is retained as a near-tied reference.

The result is not a claim of 72.54% real-hardware accuracy. It establishes that GYRO adds useful information in the synthetic benchmark, while MAG did not improve this tested activity classifier.

### Current direction

The intended SafeBand vocabulary is hierarchical:

```text
ACTIVITY STATE
├── STATIONARY
│   ├── SITTING
│   ├── STANDING
│   └── LYING / SLEEPING
├── DYNAMIC
│   ├── WALKING
│   ├── RUNNING
│   └── STAIRS
└── OTHER / UNKNOWN
```

`FALL` remains a separate emergency branch.

The next meaningful activity validation is **real BNO055 data**, not further synthetic classifier tuning.

## 5. Audio / INMP441

The INMP441 is treated as a genuine sensing modality rather than a simple loudness threshold.

### Phase-1 reference dataset

**Nonspeech7k** is the current human non-speech/acoustic reference dataset.

It provides seven event classes:

```text
BREATH
COUGH
CRYING
LAUGH
SCREAMING
SNEEZE
YAWN
```

The approved clean training set contains:

- **6,283 recordings**
- **1,899 unique File-ID groups**
- official **725-recording test set kept untouched**
- train/test File-ID overlap quarantined before benchmarking

The dataset is treated as a **reference acoustic-event benchmark**, not as a complete SafeBand emergency-audio dataset. Environmental events such as glass breaking, alarms, sirens and impacts still require additional data/domain validation.

### Audio benchmark progression

The audio branch was evaluated progressively rather than selecting a model from a single run.

**V1 handcrafted features**

Initial signal/statistical features established the baseline. The best completed V1 HGB result was approximately **67.29% accuracy**.

**V2 time-frequency representation**

The representation was expanded to:

```text
16 kHz mono
    ↓
64-bin log-mel
    ↓
20 MFCC
    ↓
MFCC delta
    ↓
MFCC delta²
    ↓
statistical summaries + RMS/ZCR
```

Leakage-safe evaluation used **5-fold StratifiedGroupKFold grouped by File ID**.

The V2 HistGradientBoosting reference reached:

- Accuracy: **83.40% pooled**
- Balanced accuracy: **75.33% pooled**
- Macro-F1: **77.19% pooled**

### CNN benchmark

A small log-mel CNN was also evaluated using the same grouped protocol.

The corrected V3.1 result was:

- Mean accuracy: **71.57% ± 5.22 pp**
- Mean balanced accuracy: **70.47% ± 1.75 pp**
- Mean Macro-F1: **64.97% ± 2.09 pp**

The CNN was therefore retained as a documented deep-learning benchmark, but not selected as the primary reference model.

### Best-fit model selection

A final model-selection experiment compared multiple classical learners on the frozen V2 representation:

```text
V2 time-frequency features
        ↓
5-fold StratifiedGroupKFold
grouped by File ID
        ↓
HGB / RBF-SVM / RF / Logistic Regression / Extra Trees
        ↓
select highest mean Macro-F1
tie-break: balanced accuracy
tie-break: accuracy
```

The selected model was **HistGradientBoosting**.

Best-fit grouped-CV result:

- Mean accuracy: **82.87% ± 3.35 pp**
- Mean balanced accuracy: **74.56% ± 3.68 pp**
- Mean Macro-F1: **76.62% ± 3.26 pp**
- Pooled accuracy: **83.30%**
- Pooled balanced accuracy: **74.97%**
- Pooled Macro-F1: **77.17%**

The selected HGB model was then refit on all **6,283 approved training recordings** and saved as:

```text
models/nonspeech7k_audio_bestfit_v1/
└── nonspeech7k_audio_bestfit_model.joblib
```

The official 725-recording test set remains untouched.

### Research positioning

The final audio model is an **acoustic evidence source** for sensor fusion. It is not itself the final emergency decision-maker.

Runtime concept:

```text
INMP441
   ↓
PCM audio
   ↓
Frozen V2 feature extraction
   ↓
Best-fit HGB
   ↓
audio event + confidence
   ↓
sensor fusion
```

The runtime interface does **not** equate loudness with a scream, distress event or emergency.

The next meaningful audio validation is **real INMP441 hardware/domain validation and expansion to environmental safety sounds**, rather than repeated optimization on Nonspeech7k.

## 6. BME680 AIRWISE

Audit:

```powershell
python tools\audit_airwise.py
```

Preparation:

```powershell
python tools\prepare_airwise_bme680.py
```

Training:

```powershell
python tools\train_airwise_bme680.py
```

Prediction:

```powershell
python tools\predict_airwise_bme680.py --csv <input.csv>
```

AIRWISE extraction depth is auto-discovered, so the scripts do not assume:

```text
datasets/raw/AIRWISE/data/...
```

They can also accept:

```powershell
python tools\audit_airwise.py --data-dir <actual_sensor_1min_directory>
```

---

## 7. PPG Reference Experiments

### PTT / MAX30101 V4.3

Preparation:

```powershell
python tools\prepare_ptt_hr_v4.py --ptt-root datasets/raw/PTT --out datasets/processed/ptt_hr_v4.csv --summary-out datasets/processed/ptt_hr_v4_summary.json
```

Training/evaluation:

```powershell
python tools\train_ppg_v4_3_final.py --input datasets/processed/ptt_hr_v4.csv --model-out models/ppg_v4_3/ppg_v4_3_max30101.joblib --report-out models/ppg_v4_3/report.json --predictions-out models/ppg_v4_3/test_predictions.csv
```

The experiment is a MAX30101-domain reference and must not be described as MAX30102 validation.

### Walking stress experiment

```powershell
python tools\prepare_walking_v5_3_final.py --root datasets/raw/WristPPGExercise --out datasets/processed/wrist_walking_v5_3.csv --summary-out datasets/processed/wrist_walking_v5_3_summary.json
```

The V5.3 walking experiment is frozen as a stress/generalization result.

---

## 8. Sensor Fusion

Sensor fusion is the **central research objective** of SafeBand AI.

Individual models provide evidence; the fusion layer combines heterogeneous evidence over time.

```text
BNO055
├── activity evidence
├── motion/posture evidence
└── fall/emergency evidence

MAX30102
├── heart rate
├── SpO₂
├── PPG-derived features
└── physiological evidence

BME680
└── environmental evidence

INMP441
└── acoustic event evidence

GPS
└── location context
        ↓
  SENSOR EVIDENCE VECTOR
        ↓
   TEMPORAL FUSION
        ↓
     RISK ENGINE
        ↓
 SAFE / WARNING / EMERGENCY
```

A sensor does **not** need to independently solve the complete safety problem. For example, BNO055 can contribute posture and motion evidence while MAX30102 contributes physiological context.

HR and SpO₂ are therefore supporting physiological evidence, not direct activity labels.

The final fusion model should be trained/validated only after synchronized SafeBand recordings exist.

### Removed sensor

The **MAX30205 body-temperature sensor is removed from the final PCB design** because of the identified measurement/reliability concerns. Body temperature is therefore not a final SafeBand input.

## 9. Risk Engine

The risk engine produces:

```text
0–29   LOW       → SAFE
30–59  MODERATE  → WARNING
60–79  HIGH      → WARNING
80–100 CRITICAL  → EMERGENCY
```

Manual SOS remains an explicit user action and is not an ML class.

Automatic fall/emergency decisions must be validated using real hardware recordings before being treated as production safety behavior.

---

## 10. Hardware-Ready Interfaces

Current Python sensor modules expose stable application-level interfaces for:

- `sensors/bno055.py`
- `sensors/max30102.py`
- `sensors/bme680.py`
- `sensors/inmp441.py`
- `sensors/gps.py`

The Python prototype may simulate hardware, but it does not pretend that desktop Python is connected to the physical ESP32-S3 peripherals.

The intended hardware target is:

```text
ESP32-S3
├── I²C
│   ├── BNO055
│   ├── MAX30102
│   └── BME680
├── I²S
│   └── INMP441
├── UART
│   ├── GPS/GNSS
│   └── EC200U
└── Display / Power
```

---

## 11. Hardware Stack

Current final hardware direction:

```text
ESP32-S3
├── I²C
│   ├── BNO055
│   ├── MAX30102
│   └── BME680
├── I²S
│   └── INMP441
├── UART
│   ├── GPS/GNSS
│   └── EC200U
└── Display / Power
```

**MAX30205 is not part of the final PCB stack.**

---

## 13. Repository Structure

```text
SafeBand-AI/
├── ai/
│   ├── activity_recognition.py
│   ├── feature_extraction.py
│   ├── ml_activity_model.py
│   ├── audio_features.py
│   ├── ml_audio_model.py
│   ├── audio_event_recognition.py
│   ├── nonspeech7k_audio_bestfit.py
│   ├── bme680_features.py
│   ├── ml_bme680_model.py
│   ├── ppg_v4_features.py
│   ├── walking_features_v5_3.py
│   ├── sensor_fusion.py
│   └── risk_engine.py
│
├── tools/
│   ├── repo_utils.py
│   ├── preflight_phase1.py
│   ├── audit_airwise.py
│   ├── prepare_airwise_bme680.py
│   ├── train_airwise_bme680.py
│   ├── prepare_ptt_hr_v4.py
│   ├── train_ppg_v4_3_final.py
│   ├── evaluate_ppg_v4_3_final.py
│   ├── prepare_walking_v5_3_final.py
│   ├── train_walking_v5_3_final.py
│   ├── evaluate_wrist_walking_v5_3_final.py
│   ├── train_activity_models.py
│   ├── train_audio_model.py
│   └── evaluate_audio_model.py
│
├── sensors/
├── communication/
├── dashboard/
├── config/
├── data/
├── models/
├── datasets/
└── docs/
```

---

## 13. Engineering Rules

1. Never use simulation scenario names as ML inputs.
2. Never train on deterministic demo profiles.
3. Never report a regression problem using classification accuracy as the primary metric.
4. Never call E4 or MAX30101 results MAX30102 validation.
5. Never use ECG waveform as a PPG feature when ECG is only the reference target.
6. Never turn AIRWISE IAQ classes into medical/emergency labels.
7. Keep test subjects/records untouched during model selection.
8. Keep raw datasets immutable; derived data belongs under `datasets/processed/`.
9. Every tool must run from the repository root.
10. Every new `tools/*.py` script must be tested by actual invocation, not only syntax compilation.
11. Dataset archive extraction depth must not be assumed when robust discovery is possible.
12. Sensor acquisition, feature extraction, ML inference, fusion and risk assessment remain separate layers.
13. Final emergency behavior requires real SafeBand hardware validation.
14. The frozen Nonspeech7k best-fit HGB artifact is a reference acoustic model; it is not a complete emergency classifier.

---

## 14. Phase-1 Preflight

Run:

```powershell
python tools\preflight_phase1.py
```

This verifies:

- required AI modules import
- sensor modules import
- reference feature modules import
- configured model artifacts are visible
- known raw dataset roots are visible

A missing dataset/model is reported as an informational `--` state rather than being mistaken for an import failure.

---

## 15. Running the Software Prototype

Install:

```powershell
pip install -r requirements.txt
```

Run:

```powershell
streamlit run app.py
```

The current dashboard remains a prototype/demo layer using simulated sensor data unless real hardware integration is explicitly implemented.

---

## 16. Phase 2 Direction

When the physical SafeBand stack is ready, the next major dataset becomes our own synchronized multimodal recording:

```text
timestamp
├── BNO055
├── MAX30102
├── BME680
├── INMP441
└── GPS
```

Those recordings will allow us to study:

```text
Activity
   +
Motion event
   +
Physiological state
   +
Environmental state
   +
Acoustic event
   +
Temporal context
        ↓
   SENSOR FUSION
        ↓
    RISK ENGINE
        ↓
 SAFE / WARNING / EMERGENCY
```

That is the point at which SafeBand AI moves from independent sensor benchmarks to its actual research objective: **multimodal safety-event detection with false-alarm reduction.**
