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
| BITS-2/SisFall fall detection | ✅ Benchmark established |
| PPG-DaLiA E4 HR reference | ✅ Frozen reference |
| PTT/MAX30101 HR reference | ✅ V4.3 frozen |
| Walking PPG stress experiment | ✅ V5.3 frozen |
| BME680/AIRWISE IAQ | ✅ V1 frozen |
| Activity recognition | 🟡 Benchmark pipeline ready; model training pending/optional |
| INMP441 audio intelligence | 🟡 Feature/training/runtime pipeline ready |
| Common sensor contracts | 🟡 Being hardened for hardware integration |
| Sensor fusion | 🟡 Software prototype exists; final fusion requires synchronized SafeBand data |
| Risk engine | 🟡 Prototype; hardware validation required |
| ESP32-S3 TinyML deployment | ⏳ Phase 2 |
| Real MAX30102 validation | ⏳ Phase 2 |
| Real BME680 validation | ⏳ Phase 2 |
| Real INMP441 validation | ⏳ Phase 2 |

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

The first learned activity benchmark is motion-first because the BITS-2 dataset provides suitable wrist accelerometer data without requiring artificial cross-sensor synchronization.

The current learned activity set is:

```text
RESTING
SITTING
WALKING
RUNNING
```

`FALL` is maintained as a separate binary detector.

`STANDING` is part of the runtime activity vocabulary, but BITS-2 does not provide a clean standing class. It must therefore be learned/validated from an appropriate dataset or SafeBand hardware recordings rather than fabricated by relabeling unrelated data.

Run:

```powershell
python tools\prepare_bits2.py --zip datasets/raw/BITS-2/full_dataset.zip --out datasets/processed/bits2
python tools\create_activity_windows.py --input datasets/processed/bits2/bits2_canonical_long.csv --out datasets/processed/bits2/activity_windows.csv
python tools\train_activity_models.py --input datasets/processed/bits2/activity_windows.csv
```

The resulting model is a **reference model**, not the final SafeBand hardware model.

---

## 5. Audio / INMP441

The INMP441 is treated as a genuine sensing modality rather than a simple loudness threshold.

The Phase-1 audio pipeline provides:

```text
PCM audio
   ↓
Windowing
   ↓
Signal + spectral features
   ↓
Audio event classifier
   ↓
event + confidence
   ↓
sensor fusion
```

The runtime interface does **not** equate loudness with a scream, distress event or emergency.

Audio training is intentionally dataset-driven. No synthetic scenario is accepted as training data.

Expected training layout:

```text
datasets/raw/audio/
├── CLASS_A/
│   ├── subject01/
│   │   └── *.wav
│   └── subject02/
│       └── *.wav
├── CLASS_B/
│   └── ...
└── ...
```

Run:

```powershell
python tools\train_audio_model.py --input datasets/raw/audio
```

The training script enforces subject-disjoint train/validation/test evaluation.

---

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

The current fusion layer is intentionally a **software prototype**.

It combines evidence from:

- motion
- physiological measurements
- body temperature
- environmental measurements
- acoustic information
- activity context

GPS is retained as location information and is not itself treated as a safety evidence source.

The final fusion model should be trained/validated only after synchronized SafeBand data exists.

---

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

## 11. Repository Structure

```text
SafeBand-AI/
├── ai/
│   ├── activity_recognition.py
│   ├── feature_extraction.py
│   ├── ml_activity_model.py
│   ├── audio_features.py
│   ├── ml_audio_model.py
│   ├── audio_event_recognition.py
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

## 12. Engineering Rules

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

---

## 13. Phase-1 Preflight

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

## 14. Running the Software Prototype

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

## 15. Phase 2 Direction

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
