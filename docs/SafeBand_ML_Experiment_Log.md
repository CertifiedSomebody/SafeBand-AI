# SafeBand-AI — ML Development & Experimental Documentation

## 1. Project purpose

SafeBand-AI is an intelligent wearable safety system intended to recognize user activity/fall-related risk from sensor data and eventually combine this with physiological, environmental, audio, GPS and communication subsystems.

This document records the ML experiments chronologically so that the methodology, decisions, limitations and results can be reused directly during final-year project reporting and research-paper preparation.

---

## 2. Dataset and feature foundation

### BITS-2 v2
- 41 subjects.
- Released ZIP physically contains 984 CSV recordings:
  - 656 ADL recordings.
  - 328 fall recordings.
- Sensor blocks: accelerometer (ACC), linear acceleration (ACG), gyroscope, heart rate (HRT), magnetometer.
- The fall-model benchmark uses the accelerometer-derived 84-feature V6 contract.
- Window: 60 samples at 20 Hz = 3 seconds.
- Step: 30 samples = 1.5 seconds.
- Subject-independent train/validation/test splitting is mandatory.

### SisFall
- 38 subjects.
- 4,505 recordings.
- 1,798 fall recordings and 2,707 non-fall recordings.
- Raw sampling rate: 200 Hz.
- The compatibility adapter uses the ADXL345 accelerometer, converts counts to g using the published sensor specification, then downsamples 200 Hz → 20 Hz by 10-sample block averaging.
- Same 60-sample / 30-step / 84-feature contract is used for cross-dataset comparison.

### V6 feature contract
The final engineered fall-window representation contains 84 features:
- Base statistical/time-domain/spectral features.
- Original fall-event morphology features.
- V6 morphology features including peak prominence, peak-to-baseline, pre/post-impact statistics, energy ratios, post-impact quietness, jerk morphology, impulse area, axis dominance, tilt change and spectral entropy.

---

# 3. Chronological ML model history

> Important: The exact wall-clock timestamps of the older experiments were not preserved in the available project record. Where an exact timestamp is unavailable, this document deliberately records the experiment by development stage/date rather than inventing a time.

## Stage 0 — Initial combined activity/fall baseline
**Development stage:** Initial SafeBand ML baseline

A combined 5-class model was first evaluated using:
- FALL
- RESTING
- RUNNING
- SITTING
- WALKING

Result:
- Approximately 75% test accuracy.
- Approximately 0.63 macro-F1.

Decision:
- Rejected as the final model.
- The project moved to a hierarchical design separating activity recognition from fall detection.

Reason:
- Fall detection requires different operating characteristics from ordinary activity classification.
- Accuracy alone was not considered sufficient for a safety system.

---

## Stage 1 — BITS-2 hierarchical baseline

**Dataset:** BITS-2 v2  
**Window experiments:** 40, 60 and 80 samples at 20 Hz.

Activity recognition:
- RESTING
- SITTING
- WALKING
- RUNNING

Fall detection:
- Separate binary FALL / NON_FALL model.

Subject-independent evaluation was used.

### Window comparison
- 40 samples / 2 s:
  - Activity macro-F1 ≈ 75.8%
  - Balanced accuracy ≈ 76.7%
- 60 samples / 3 s:
  - Activity macro-F1 ≈ 77.1%
  - Balanced accuracy ≈ 78.3%
  - Fall recall ≈ 91.1%
- 80 samples / 4 s:
  - Activity macro-F1 ≈ 78.8%
  - Balanced accuracy ≈ 79.6%
  - Fall recall ≈ 87.4%

Decision:
- 80 samples was favorable for activity recognition.
- 60 samples was favorable for fall detection.
- The fall pipeline therefore standardized on 60 samples at 20 Hz.

---

# Stage 2 — BITS-2 V2 benchmark

**Development stage:** BITS-2 V2  
**Purpose:** Establish a defensible subject-independent fall/activity baseline.

### Activity model
Random Forest selected.

Test:
- Accuracy: 80.41%
- Balanced accuracy: 81.10%
- Macro-F1: 79.82%

Per-class F1:
- RESTING: 0.803
- SITTING: 0.633
- WALKING: 0.785
- RUNNING: 0.972

### Fall model
Random Forest with threshold 0.40.

Test:
- Accuracy: 75.34%
- Balanced accuracy: 77.14%
- Precision: 65.02%
- Recall: 87.66%
- F1: 75.32%
- FPR: 33.08%
- ROC-AUC: 0.872

Conclusion:
- Strong fall recall, but false-positive rate was too high.
- Single-window classification was not sufficient for a safety alert.

---

# Stage 3 — BITS-2 V3

**Development stage:** BITS-2 V3  
**Purpose:** Improve the fall detector and investigate threshold behavior.

Activity Random Forest:
- Accuracy: 81.39%
- Balanced accuracy: 81.82%
- Macro-F1: 80.77%

Fall Random Forest:
- Threshold 0.41:
  - Recall: 88.46%
  - Precision: 65.43%
  - FPR: 33.08%
  - ROC-AUC: 0.872

Threshold trade-off:
- 0.50 → recall 83.73%, precision 70.57%, FPR 30.36%
- 0.60 → recall 73.06%, precision 76.70%, FPR 19.30%
- 0.70 → recall 57.42%, precision 85.93%, FPR 8.17%

Conclusion:
- Threshold adjustment alone cannot simultaneously provide very high recall and very low false-positive rate.
- This motivated temporal confirmation.

---

# Stage 4 — BITS-2 V4 temporal event confirmation

**Development stage:** BITS-2 V4  
**Purpose:** Convert noisy window predictions into recording-level fall events.

Temporal rule:
- Threshold: 0.65
- Minimum hits: 3
- Maximum gap: 30 samples

Historical result:
- Event recall: 93.54%
- Precision: 93.54%
- F1: 93.54%
- FPR: 5.20%
- False alarms: 21
- Missed falls: 21

Conclusion:
- Temporal confirmation produced a major improvement over independent window decisions.
- This became the conceptual basis for the later leakage-safe temporal event engine.

---

# Stage 5 — BITS-2 V5

**Development stage:** BITS-2 V5  
**Purpose:** Correct validation/test leakage in temporal threshold selection.

Validation-selected rule:
- Hit threshold: 0.55
- Peak threshold: 0.70
- Minimum hits: 3
- Maximum gap: 60 samples

Validation:
- Recall: 95.31%
- Precision: 67.78%
- F1: 79.22%
- FPR: 36.25%

Held-out test:
- Recall: 85.94%
- Precision: 62.50%
- F1: 72.37%
- FPR: 41.25%

Conclusion:
- Strong validation performance did not generalize.
- Threshold-only tuning was therefore stopped.
- This was an important methodological finding: temporal parameters must be selected strictly on validation subjects and evaluated once on unseen test subjects.

---

# Stage 6 — BITS-2 V6 engineered fall morphology

**Development stage:** BITS-2 V6  
**Purpose:** Improve the underlying window classifier using richer fall morphology.

Added 36 event-morphology features, giving an 84-feature representation.

Compared:
- Extra Trees
- Random Forest
- HistGradientBoosting

Validation:
- Extra Trees:
  - F1: 0.8618
  - Recall: 0.8281
  - FPR: 0.075
- Random Forest:
  - F1: 0.8615
  - Recall: 0.8750
  - FPR: 0.125
- HistGradientBoosting:
  - F1: 0.8992
  - Recall: 0.9063
  - FPR: 0.0875

HGB was selected for the V6 window-model benchmark.

A temporal operating rule was also tested:
- Hit threshold: 0.50
- Peak threshold: 0.75
- Minimum hits: 2
- Maximum gap: 60

Held-out test:
- Recall: 90.63%
- Precision: 63.04%
- F1: 74.36%
- FPR: 42.50%
- ROC-AUC: 0.92285

Conclusion:
- The 84-feature representation substantially improved model ranking/ROC-AUC.
- Event-level false alarms remained too high.

---

# Stage 7 — BITS-2 V7 event-trajectory model

**Development stage:** BITS-2 V7  
**Purpose:** Learn event-level probability trajectories instead of relying only on hand-selected temporal rules.

Created 30 event-trajectory features, including:
- Maximum probability.
- Mean/std/median/p90.
- Top-k probability statistics.
- Proportion above probability thresholds.
- Maximum runs.
- Evidence sums.
- Rise/fall behavior.
- Peak location.
- Pre/post probability statistics.
- Candidate-run and active-span characteristics.

Leakage correction:
- V6 clone trained on TRAIN subjects only for validation/test probability generation.
- Event-model training used subject-grouped 5-fold OOF probabilities on training subjects.

Models:
- Logistic Regression
- Extra Trees
- Random Forest
- HistGradientBoosting

Selected:
- Logistic Regression.
- Threshold: 0.26.

Validation:
- Recall: 98.44%
- Precision: 72.41%
- F1: 83.44%
- FPR: 30.00%
- ROC-AUC: 0.9551

Held-out test:
- Recall: 92.19%
- Precision: 71.95%
- F1: 80.82%
- FPR: 28.75%
- False alarms: 23
- Missed falls: 5
- ROC-AUC: 0.9291

Conclusion:
- Event trajectory modeling improved the benchmark but still did not meet the target FPR.

---

# Stage 8 — BITS-2 V8 final single-dataset benchmark

**Development stage:** BITS-2 V8  
**Purpose:** Final BITS-2-only benchmark before adding another dataset.

Compared:
- Logistic Regression
- Extra Trees
- Random Forest
- HistGradientBoosting

Tested false-negative weighting and probability-floor strategies.

Selected:
- HistGradientBoosting
- FN weight: 1
- Threshold: 0.12
- pmax floor: 0.675

Validation:
- Recall: 95.31%
- Precision: 76.25%
- F1: 84.72%
- FPR: 23.75%

Held-out test:
- Recall: 90.63%
- Precision: 77.33%
- F1: 83.45%
- FPR: 21.25%
- False alarms: 17
- Missed falls: 6
- ROC-AUC: 0.93574

Conclusion:
- V8 was frozen as the final BITS-2-only benchmark.
- Further BITS-2-specific threshold tuning was stopped.
- The project moved to cross-dataset robustness.

---

# Stage 9 — BITS-2 → SisFall cross-dataset compatibility test

**Purpose:** Test whether a BITS-2-trained model transfers to a different fall dataset.

Important correction:
- The original evaluator accidentally used `predict_proba(X)[:, 1]` even though model classes were `['FALL', 'NON_FALL']`.
- Therefore column 0 was the FALL probability.
- The evaluator was corrected before interpreting the final compatibility result.

SisFall preparation was also corrected:
- ADXL345 counts converted to g, not m/s².
- 200 Hz → 20 Hz block averaging.
- Exact 84-feature contract preserved.

Corrected cross-dataset result using BITS-2 V6 model:
- Window accuracy: 66.26%
- Window balanced accuracy: 49.96%
- Window precision: 14.29%
- Window recall: 0.037%
- Window F1: 0.075%
- Window ROC-AUC: 0.480

Recording:
- Accuracy: 59.87%
- Balanced accuracy: 49.86%
- Precision: 25.00%
- Recall: 0.278%
- F1: 0.550%
- ROC-AUC: 0.575

Conclusion:
- Direct BITS-2 → SisFall transfer was poor.
- This demonstrated significant sensor/domain shift.
- This was a key justification for multi-dataset training.

---

# Stage 10 — Multi-Dataset V1

**Development stage:** Multi-Dataset V1  
**Purpose:** Train on BITS-2 + SisFall and evaluate subject-independent robustness.

Design:
- 84 V6 features.
- 60 samples @ 20 Hz.
- Step 30.
- Subject-independent split.
- Explicit feature schema.
- No dataset/subject identity feature.
- Train-only preprocessing.
- Compared Logistic/RF/ET/HGB.

Selected:
- Random Forest.

Held-out results:

### BITS-2
- Accuracy: 75.28%
- Balanced accuracy: 75.15%
- Precision: 73.20%
- Recall: 81.37%
- F1: 77.07%
- ROC-AUC: 0.833

### SisFall
- Accuracy: 89.03%
- Balanced accuracy: 86.61%
- Precision: 87.46%
- Recall: 79.06%
- F1: 83.05%
- ROC-AUC: 0.944

Conclusion:
- Combined training improved cross-domain robustness relative to direct single-domain transfer.
- Dataset-specific performance was still unequal.

---

# Stage 11 — Multi-Dataset V2

**Development stage:** Multi-Dataset V2  
**Purpose:** Correct dataset imbalance and make dataset-macro performance a primary selection criterion.

Training strategies:
1. Unweighted.
2. Equal dataset weight (50% BITS-2 / 50% SisFall).
3. Balanced subsampling.

Selection:
- Mean of BITS-2 validation F1 and SisFall validation F1.

Selected:
- Equal-dataset weighting.
- Random Forest.

Held-out results:

### BITS-2
- Accuracy: 75.56%
- Balanced accuracy: 75.52%
- Precision: 75.51%
- Recall: 77.14%
- F1: 76.31%
- ROC-AUC: 0.834
- FPR: 26.09%

### SisFall
- Accuracy: 88.95%
- Balanced accuracy: 86.22%
- Precision: 88.42%
- Recall: 77.68%
- F1: 82.71%
- ROC-AUC: 0.943
- FPR: 5.24%

Dataset macro:
- Accuracy: 82.26%
- Balanced accuracy: 80.87%
- Precision: 81.97%
- Recall: 77.41%
- F1: 79.51%
- ROC-AUC: 0.888

Conclusion:
- Balanced training modestly improved dataset-macro F1.
- The BITS-2 false-positive rate remained the major bottleneck.
- Target of ≥95% recall and ≤5% FPR per dataset was not achieved.

---

# Stage 12 — Multi-Dataset V3 raw-signal augmentation benchmark

**Development stage:** Multi-Dataset V3  
**Purpose:** Test whether realistic raw-signal augmentation can improve cross-dataset generalization.

Augmentation was performed on raw 3-axis accelerometer windows and the 84 features were recomputed.

Validation/test data were untouched.

Recipes:
1. Baseline.
2. Random 3-D orientation rotation, maximum ±12°.
3. Magnitude scaling + Gaussian noise.
4. Orientation + scaling/noise.
5. Temporal jitter, ±3 samples.
6. Combined augmentation with three synthetic copies.

All augmentation was restricted to training subjects, and dataset contribution was reweighted to maintain 50/50 total training weight.

Result:
- **Baseline was selected.**
- None of the augmentation recipes improved the validation selection metric sufficiently to replace the baseline.

Final held-out benchmark therefore remained:

### BITS-2
- Recall: 77.14%
- F1: 76.31%
- ROC-AUC: 0.834
- FPR: 26.09%

### SisFall
- Recall: 77.68%
- F1: 82.71%
- ROC-AUC: 0.943
- FPR: 5.24%

Conclusion:
- Raw-signal augmentation did not solve the domain-shift problem.
- The baseline RF was frozen.
- The project moved to temporal event-level confirmation.

---

# Stage 13 — Multi-Dataset V4 Temporal Event Engine

**Development stage:** Multi-Dataset V4  
**Current completed benchmark**

Purpose:
- Convert window-level RF probabilities into recording-level fall events.
- Avoid test leakage.
- Select event parameters on validation subjects only.
- Refit the final window model on TRAIN+VALIDATION.
- Evaluate the selected event operating point once on held-out TEST subjects.

The model uses:
- Random Forest.
- 84 features.
- 60 samples @ 20 Hz.
- Step 30.
- Subject-level splitting.

Final split:

### BITS-2
- Train: 9,613 windows / 25 subjects.
- Validation: 2,162 windows / 8 subjects.
- Test: 2,913 windows / 8 subjects.

### SisFall
- Train: 27,674 windows / 23 subjects.
- Validation: 10,028 windows / 8 subjects.
- Test: 9,981 windows / 8 subjects.

The BITS-2 subject split uses a subject-level fallback because every BITS-2 subject contains fall data; it does NOT perform window-level splitting.

### Validation-selected event rule
- Threshold: 0.75.
- Minimum hits: 2.
- Maximum gap: 90 samples.
- Validation pooled recall: 96.81%.
- Validation pooled FPR: 4.88%.
- Validation macro F1: 90.21%.

### Held-out TEST — BITS-2
- Accuracy: 82.64%
- Balanced accuracy: 83.28%
- Precision: 76.00%
- Recall: **89.06%**
- F1: **82.01%**
- FPR: **22.50%**
- False alarms: 18
- Missed falls: 7

### Held-out TEST — SisFall
- Accuracy: 98.82%
- Balanced accuracy: 98.79%
- Precision: 98.40%
- Recall: **98.67%**
- F1: **98.54%**
- FPR: **1.08%**
- False alarms: 6
- Missed falls: 5

### Dataset-macro TEST
- Accuracy: 90.73%
- Balanced accuracy: 91.04%
- Precision: 87.20%
- Recall: **93.86%**
- F1: **90.27%**
- FPR: **11.79%**

### Pooled TEST
- Accuracy: 96.65%
- Balanced accuracy: 96.75%
- Precision: 94.68%
- Recall: **97.27%**
- F1: **95.96%**
- FPR: **3.77%**
- False alarms: 24
- Missed falls: 12

Methodological integrity:
- Validation probabilities came from a model trained only on TRAIN subjects.
- Event parameters were selected only on validation subjects.
- The final window model was refit using TRAIN+VALIDATION.
- Held-out TEST was evaluated once.
- V4 uses recording-level event metrics as its primary metric rather than pooled window metrics.

Conclusion:
- Temporal event confirmation is a substantial improvement over the earlier window-only benchmarks.
- The pooled benchmark reaches 97.27% recall and 3.77% FPR.
- However, the dataset-macro target of ≥95% recall and ≤5% FPR is **not yet achieved**, primarily because BITS-2 remains difficult.
- V4 should therefore be treated as the current research benchmark, not as deployment-ready clinical/safety performance.

---

# 4. Current achievement

As of the latest V4 experiment, SafeBand-AI has achieved:

1. A reproducible 84-feature fall-detection representation.
2. Subject-independent evaluation across two heterogeneous datasets.
3. Corrected BITS-2 and SisFall preprocessing pipelines.
4. Demonstrated and quantified BITS-2 → SisFall domain shift.
5. Demonstrated that multi-dataset training substantially improves robustness.
6. Tested dataset balancing.
7. Tested raw-signal augmentation without changing validation/test data.
8. Developed a leakage-safe temporal fall-event engine.
9. Achieved 97.27% pooled held-out event recall and 3.77% pooled FPR.
10. Established that BITS-2 remains the limiting domain.
11. Established a rigorous benchmark that can be extended using real SafeBand hardware data.

---

# 5. What has NOT yet been achieved

The following claims must NOT be made in the paper:

- Do not claim ≥95% recall and ≤5% FPR on every dataset.
- Do not call the current model deployment-ready.
- Do not claim clinical validation.
- Do not claim real-world SafeBand performance from BITS-2/SisFall alone.
- Do not use synthetic augmentation data as evidence of independent test performance.
- Do not claim that pooled metrics represent equal performance across datasets.

The correct statement is that the current V4 benchmark demonstrates strong pooled event-level performance but substantial dataset-specific variation.

---

# 6. Recommended next phase

The ML work should now move from repeated public-dataset threshold tuning toward **SafeBand-specific data and multimodal fusion**.

Recommended research sequence:

```text
V4 temporal fall engine
        ↓
SafeBand hardware acquisition
        ↓
BNO055 / accelerometer + gyroscope
MAX30102 physiological signals
BME680 environmental signals
INMP441 audio/context
        ↓
SafeBand-specific synchronized recordings
        ↓
Domain adaptation / calibration
        ↓
Sensor fusion
        ↓
Fall + activity + physiological anomaly
        ↓
Safety state
NORMAL / WARNING / CRITICAL
        ↓
GPS + caregiver notification
```

HARMES should be considered later for multimodal ADL/context research because its modalities are closely related to the planned SafeBand sensor stack, but its large raw/preprocessed data volume makes it unsuitable as the immediate next training step.

---

# 7. Paper-ready methodological narrative

The experiments establish a progression from simple window classification toward robust event-level safety inference:

**Baseline → hierarchical activity/fall modeling → richer fall morphology → temporal modeling → cross-dataset validation → balanced multi-dataset training → augmentation study → leakage-safe temporal event confirmation.**

The central research finding is:

> Increasing model complexity or tuning probability thresholds alone did not eliminate false alarms under dataset shift. Combining heterogeneous datasets and introducing temporal event confirmation substantially improved held-out event-level robustness, but performance remained dataset-dependent, motivating evaluation on SafeBand-specific multimodal hardware data.

This should be the backbone of the ML methodology/results section of the paper.

---

## 8. Experiment artifact naming convention

Use the following artifact naming pattern going forward:

```text
models/
├── bits2_v2/
├── bits2_v3/
├── bits2_v4/
├── bits2_v5/
├── bits2_v6/
├── bits2_v7/
├── bits2_v8/
├── multidataset_v1/
├── multidataset_v2/
├── multidataset_v3/
└── multidataset_v4_event/
```

For every future experiment record:
- timestamp
- Git commit/hash
- dataset version
- preprocessing version
- feature version
- window size
- step
- train/validation/test subject lists
- model/hyperparameters
- random seed
- selection metric
- validation result
- held-out test result
- artifact filename
- conclusion/decision

This makes the experimental history auditable and paper-ready.
