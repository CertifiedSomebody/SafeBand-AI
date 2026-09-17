# SafeBand AI — Model Results, Baselines & Phase-1 Continuity Note

**Purpose:** This document preserves the authoritative model results, explains why some newly generated Phase-1 models show lower accuracy, and prevents future chats/work from accidentally treating an auxiliary model as a replacement for a validated model.

**Project:** SafeBand AI — An Intelligent Safety Band Using Sensor Fusion and Artificial Intelligence  
**Repository:** https://github.com/CertifiedSomebody/SafeBand-AI  
**Status:** Phase 1 software/AI integration

---

## 1. IMPORTANT: DO NOT CONFUSE THE CURRENT AUXILIARY MODELS WITH OUR BEST VALIDATED RESULTS

The recent command:

```text
python tools\train_activity_models.py --input datasets\processed\bits2\activity_windows.csv
```

produced:

### Activity Recognition — current BITS-2 model

- Model: ExtraTrees
- Accuracy: **82.824%**
- Balanced accuracy: **80.737%**
- Macro F1: **80.876%**

Confusion matrix:

```text
[[182,   0,  66,   0],
 [  0, 529,  11,  35],
 [ 55,   0, 288,  56],
 [  1,  13,  61, 438]]
```

### Fall Detector — current simple BITS-2 model

- Model: ExtraTrees
- Accuracy: **78.421%**
- Balanced accuracy: **77.588%**
- Fall recall: **82.495%**
- Macro F1: **77.691%**

Confusion matrix:

```text
[[2017, 428],
 [ 474, 1261]]
```

These models are **not the authoritative final cross-dataset fall system**, and the activity model is a BITS-2-only baseline.

They must NOT be used in documentation as evidence that the whole SafeBand AI system only achieves ~78–83% accuracy.

---

# 2. AUTHORITATIVE HIGH-PERFORMANCE FALL RESULTS

The project previously developed and validated a **multidataset fall-detection pipeline (V5)** using BITS-2 + SisFall.

The authoritative pipeline is:

```text
tools/train_multidataset_v5_balanced.py
ai/ml_safety_v5.py
```

It uses balanced multidataset training, validation-based operating-point selection, and temporal event confirmation.

## V5 baseline — per-dataset results

### BITS-2

- Recordings: 144
- Falls: 64
- Non-falls: 80
- Accuracy: **87.50%**
- Balanced accuracy: **87.50%**
- Precision: **84.85%**
- Recall: **87.50%**
- F1: **86.15%**
- False-positive rate: **12.50%**
- False alarms: 10
- Missed falls: 8

### SisFall

- Recordings: 931
- Falls: 375
- Non-falls: 556
- Accuracy: **98.28%**
- Balanced accuracy: **98.17%**
- Precision: **98.12%**
- Recall: **97.60%**
- F1: **97.86%**
- False-positive rate: **1.259%**
- False alarms: 7
- Missed falls: 9

### Macro summary

- Accuracy: **92.89%**
- Balanced accuracy: **92.835%**
- Precision: **91.486%**
- Recall: **92.550%**
- F1: **92.007%**
- False-positive rate: **6.879%**

## V5 pooled event-level result

Using temporal event confirmation:

- Threshold: **0.75**
- Minimum hits: **3**
- Maximum gap: **120 samples**
- Accuracy: **96.837%**
- Balanced accuracy: **96.727%**
- Precision: **96.128%**
- Recall: **96.128%**
- F1: **96.128%**
- False-positive rate: **2.673%**
- False alarms: 17
- Missed falls: 17

This is one of the project's key high-performance results.

---

# 3. V5 HARD-NEGATIVE EXPERIMENT

A hard-negative experiment was also performed:

```text
python tools/train_multidataset_v5_balanced.py \
  --bits2 datasets/processed/bits2/fall_windows_v6.csv \
  --sisfall datasets/processed/sisfall_windows.csv \
  --out models/multidataset_v5_hardneg \
  --hard-negative
```

Results:

### BITS-2

The BITS-2 result remained unchanged.

### SisFall

- Recall: **97.867%**
- F1: **98.260%**
- False-positive rate: **0.899%**
- False alarms: 5
- Missed falls: 8

### Macro

- Recall: **92.683%**
- F1: **92.207%**
- False-positive rate: **6.700%**

### Pooled event result

- Selected threshold: **0.80**
- Minimum hits: **2**
- Maximum gap: **30 samples**
- Accuracy: **97.116%**
- Balanced accuracy: **96.998%**
- Precision: **96.575%**
- Recall: **96.355%**
- F1: **96.465%**
- False-positive rate: **2.358%**
- False alarms: 15
- Missed falls: 16

This hard-negative result is important because it shows that the project has already reached the **~97% event-level range** on the validated multidataset benchmark.

---

# 4. WHY THE NEW NUMBERS LOOK LOWER

There is no contradiction.

The recent activity/fall trainer is a **different experiment**.

The new command trains:

```text
BITS-2 only
        ↓
simple classifier
        ↓
activity model + auxiliary fall model
```

The validated V5 system trains:

```text
BITS-2 + SisFall
        ↓
balanced multidataset learning
        ↓
validation-selected operating point
        ↓
temporal event confirmation
        ↓
fall event decision
```

Therefore:

**82.8% activity accuracy is not a replacement for the 96–98% fall-event results.**

Likewise:

**78.4% simple BITS-2 fall accuracy is not a replacement for the validated V5 multidataset fall detector.**

Different task, datasets, features, training protocol, and evaluation protocol mean the percentages cannot be compared as if they were the same experiment.

---

# 5. DATASET-SPECIFIC VS CROSS-DATASET RESULTS

This distinction is critical.

The high numbers are not a claim that SafeBand will achieve 97–98% accuracy on arbitrary real-world users.

They are benchmark results under the documented evaluation protocols.

In particular:

- BITS-2 and SisFall are different datasets.
- Sensor hardware and signal characteristics can differ.
- Public-dataset performance does not automatically equal MAX30102/BNO055/BME680/INMP441 hardware performance.
- Real-world deployment requires hardware-domain validation.
- Subject-independent evaluation is important when claiming generalization.

The 97% range should therefore be described as:

> **Validated benchmark/event-level performance of the multidataset fall-detection pipeline under its documented evaluation protocol.**

It should NOT be described as:

> “The SafeBand AI device has 97% real-world accuracy.”

---

# 6. ACTIVITY RECOGNITION IS A SEPARATE TASK

Activity recognition currently has:

- ExtraTrees
- BITS-2 dataset
- Accuracy: **82.824%**
- Balanced accuracy: **80.737%**
- Macro F1: **80.876%**

This is not inherently inconsistent with a stronger fall detector.

Activity recognition is a multiclass classification problem with several ordinary activities, whereas fall detection is a binary event-detection problem and uses a different validated pipeline.

Future activity work may improve this, but the current result should remain clearly labeled as the **BITS-2 activity baseline**.

---

# 7. PPG RESULTS — KEEP THESE SEPARATE TOO

## PPG-DaLiA / E4 reference

Final V3 reference model:

- 64,697 windows
- 15 subjects
- ExtraTrees selected
- Held-out test MAE: **7.707 BPM**
- RMSE: **12.469 BPM**
- R²: **0.533**
- ±3 BPM: **43.76%**
- ±5 BPM: **56.75%**
- ±10 BPM: **74.44%**

This is an **E4 benchmark/reference**, not MAX30102 production performance.

## PTT / MAX30101 V4.3

Final V4.3:

- 15,982 windows
- 22 subjects
- HGB selected
- Held-out test MAE: **4.576 BPM**
- RMSE: **5.955 BPM**
- R²: **0.591**
- ±5 BPM: **69.57%**
- ±10 BPM: **92.47%**

Walking subset was much harder:

- MAE: **6.497 BPM**
- R²: **-0.884**
- ±5 BPM: **53.17%**
- ±10 BPM: **80.92%**

This demonstrates why PPG performance must be reported with the dataset and evaluation condition.

## Wrist PPG exercise stress test V5.3

Locked final S9 test:

- MAE: **11.276 BPM**
- RMSE: **12.162 BPM**
- R²: **-36.829**
- ±10 BPM: **45.794%**

This cross-subject exercise result was intentionally frozen and not endlessly tuned.

### PPG interpretation

These results do NOT provide a 97–98% HR accuracy claim.

They are different physiological regression benchmarks and must be reported using MAE/RMSE/R²/tolerance percentages rather than classification accuracy.

Actual MAX30102 hardware-domain validation is still required.

---

# 8. BME680 RESULTS

AIRWISE is the primary BME680-native benchmark.

Dataset:

- 3 indoor locations
- 146,538 total rows
- BME680-derived environmental variables
- minute-averaged data

V1 selected model:

- Histogram Gradient Boosting

Held-out test:

- Accuracy: **91.399%**
- Balanced accuracy: **79.465%**
- Macro precision: **77.368%**
- Macro recall: **79.465%**
- Macro F1: **78.244%**

The large difference between accuracy and balanced accuracy matters because the environmental classes are imbalanced.

This is a multiclass environmental classification benchmark, not a safety-event accuracy claim.

---

# 9. CURRENT PHASE-1 MODEL STATUS

As of the current integration checkpoint:

```text
ACTIVITY
  BITS-2 baseline                 ✅
  ~82.8% accuracy

FALL
  Multidataset V5                 ✅ AUTHORITATIVE
  ~96–97% pooled event metrics
  ~98% SisFall benchmark metrics

PPG
  E4 reference V3                 ✅
  MAX30101 PTT V4.3               ✅
  Exercise stress V5.3            ✅ frozen
  Actual MAX30102 validation      ⏳ later hardware stage

BME680
  AIRWISE V1                      ✅
  ~91.4% raw accuracy
  ~79.5% balanced accuracy

AUDIO / INMP441
  Dataset/model                   ⏳ not trained yet

SENSOR FUSION
  Architecture                    ✅
  Real synchronized hardware      ⏳ later

RISK ENGINE
  Architecture                    ✅
  Hardware validation             ⏳ later
```

---

# 10. MODEL ARTIFACT RULE — VERY IMPORTANT

Never silently overwrite a validated model with a lower-performing exploratory model.

Use explicit directories/names:

```text
models/
├── multidataset_v5_hardneg/
│   └── AUTHORITATIVE FALL PIPELINE
│
├── activity_bits2_v2/
│   └── BITS-2 ACTIVITY BASELINE
│
├── bme680_airwise_v1/
│   └── BME680 V1
│
├── ppg_v4/
│   └── PPG V4.3
│
└── audio_event_v1/
    └── future audio model
```

If a future experiment produces a lower score, retain it as an experiment/baseline and do not call it the new final model.

---

# 11. HOW TO REPORT RESULTS IN FUTURE CHATS

When discussing SafeBand AI results, always include:

1. Model/version
2. Dataset
3. Task
4. Split/evaluation protocol
5. Metric
6. Whether the result is validation or held-out test
7. Whether it is benchmark/reference or hardware-domain
8. Whether it is authoritative/frozen or exploratory

Example:

> “The authoritative fall pipeline is multidataset V5/hard-negative. Its pooled event-level held-out benchmark achieved 97.116% accuracy, 96.998% balanced accuracy, 96.575% precision, 96.355% recall and 96.465% F1 under the documented event-confirmation protocol.”

Do not shorten that to:

> “Our model has 97% accuracy.”

The latter loses critical experimental context.

---

# 12. WHAT SHOULD HAPPEN NEXT

The project should NOT spend more time trying to force every individual model above 95%.

The next meaningful gaps are:

### A. Audio / INMP441

Select a defensible audio dataset and build the actual acoustic-event pipeline.

Important:

```text
LOUD ≠ SCREAM ≠ EMERGENCY
```

Audio should initially provide evidence to the sensor-fusion layer rather than independently declaring an emergency.

### B. Hardware-domain validation

Once the hardware is available:

```text
MAX30102
BNO055
BME680
INMP441
GPS
EC200U
        ↓
synchronized recordings
        ↓
real sensor-domain validation
```

### C. Sensor fusion

Combine:

```text
activity
fall/event
PPG
environment
audio
GPS
```

into a unified risk state:

```text
SAFE
WARNING
EMERGENCY
```

### D. Embedded/TinyML deployment

Only after the models and feature contracts are stable should the final embedded inference path be optimized for ESP32-S3.

---

# 13. PHASE-1 INTEGRATION CHECKPOINT

Latest successful verification:

```text
python tools\train_activity_models.py --input datasets\processed\bits2\activity_windows.csv
```

Result:

```text
activity: selected=extra_trees
fall_detector: selected=extra_trees
Saved report:
models\activity_bits2_v2\report.json
```

Then:

```text
python tools\preflight_phase1.py
```

Result:

```text
PHASE-1 PREFLIGHT: PASS
```

Detected artifacts:

```text
[OK] activity model
[OK] fall model
[OK] BME680 V1 model
[OK] PPG V4.3 model
[--] audio model: no artifact found
```

The missing audio artifact is currently expected.

Finally:

```text
python -m compileall ai tools
```

completed successfully.

---

# 14. CONTINUITY INSTRUCTION FOR FUTURE CHATS

If this document is encountered in a new SafeBand AI chat:

**Treat the multidataset V5/hard-negative fall pipeline as the authoritative high-performance fall result.**

Do not replace it merely because:

```text
activity_bits2_v2
```

reports ~82.8% accuracy or its auxiliary fall detector reports ~78.4%.

Those are separate BITS-2-only models.

The project already has documented benchmark results in the **96–98% range for the validated fall pipeline**, depending on whether the metric is pooled event-level performance or the SisFall benchmark.

Future experiments must be compared against the correct baseline and must preserve the existing validated artifacts/reports.

The ultimate goal is not to maximize one public-dataset percentage. The goal is a reproducible, hardware-compatible SafeBand pipeline whose individual sensor outputs can be synchronized and fused into a reliable safety decision.

---

## Quick reference

| Branch | Authoritative result/status |
|---|---|
| Fall — V5 hard-negative pooled event | **97.116% accuracy / 96.465% F1** |
| Fall — V5 baseline pooled event | **96.837% accuracy / 96.128% F1** |
| Fall — SisFall V5 hard-negative | **98.260% F1 / 97.867% recall** |
| Fall — BITS-2 V5 | **87.50% accuracy / 87.50% recall** |
| Activity — BITS-2 v2 | **82.824% accuracy / 80.737% balanced accuracy** |
| BME680 — AIRWISE V1 | **91.399% accuracy / 79.465% balanced accuracy** |
| PPG — PTT V4.3 | **4.576 BPM MAE / 92.47% within ±10 BPM** |
| PPG — E4 V3 | **7.707 BPM MAE** |
| PPG — Exercise V5.3 | **11.276 BPM MAE** |
| Audio — INMP441 | **Pending dataset/model** |

**Last updated:** September 2026
