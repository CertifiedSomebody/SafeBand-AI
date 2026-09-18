# SafeBand AI — Activity Dataset Discovery & Stationary-State Findings

**Date:** 18 September 2026  
**Phase:** Phase 1 — Software / AI  
**Topic:** Activity Recognition and Stationary-State Dataset Strategy

## 1. Discovery Summary

During Phase 1 activity-recognition experiments, multiple public datasets and model families were evaluated under subject-independent validation.

The recurring finding is **strong subject-to-subject variation**, particularly for stationary activities such as sitting, standing, lying/resting.

The project conclusion is:

> Public activity datasets should be retained as external benchmarks, but the final SafeBand activity-recognition system should eventually be validated using data collected specifically for the SafeBand sensor configuration, activity taxonomy, and deployment conditions.

This does **not** mean PAMAP2 or BITS2 is a bad dataset. It means their collection conditions and labels are not guaranteed to match SafeBand.

## 2. BITS2 Findings

The strongest ACC-only deep-learning experiment was Activity DL V1 using raw temporal acceleration windows and a Tiny 1D CNN:

- Accuracy: **83.023%**
- Balanced accuracy: **81.089%**
- Macro-F1: **80.973%**

Despite this improvement, stationary states remained the main confusion region, especially RESTING ↔ SITTING.

An ACC+GYRO Tiny CNN was subsequently tested:

- Accuracy: **74.654%**
- Balanced accuracy: **71.229%**
- Macro-F1: **71.252%**

Sequence models on the same ACC+GYRO representation were also weaker:

**Transformer**
- Accuracy: **70.974%**
- Balanced accuracy: **67.266%**
- Macro-F1: **67.453%**

**LSTM**
- Accuracy: **69.456%**
- Balanced accuracy: **65.326%**
- Macro-F1: **65.420%**

This suggested that simply adding model complexity or another sensor stream was not resolving the underlying stationary-state problem.

## 3. PAMAP2 Stationary Benchmark

PAMAP2 was introduced because it contains explicit stationary activities:

- LYING
- SITTING
- STANDING

The selected PAMAP2-derived CSV contains hand/wrist ACC, GYRO and MAG plus chest/ankle streams and PeopleId.

For the SafeBand-oriented benchmark, only **hand/wrist ACC + GYRO** were used. Chest and ankle channels were excluded because they are not representative of a single wrist-worn SafeBand.

The processed benchmark contains:

- **5,640 windows**
- **8 subjects**
- Input shape: **(N, 6, 200)**
- 2-second windows
- 50% overlap
- subject-independent 5-fold evaluation

### PAMAP2 Tiny CNN V1

- Accuracy: **75.266%**
- Balanced accuracy: **75.178%**
- Macro-F1: **74.971%**

Class F1:

- LYING: **0.759**
- SITTING: **0.679**
- STANDING: **0.811**

### PAMAP2 RBF-SVM V2

- Accuracy: **76.046%**
- Balanced accuracy: **76.069%**
- Macro-F1: **76.015%**

Class F1:

- LYING: **0.745**
- SITTING: **0.704**
- STANDING: **0.832**

The RBF-SVM was retained as the clean classical stationary benchmark.

## 4. PAMAP2 Forensic Audit

The audit found:

- 5,640 windows
- subjects 1–8
- 0 non-finite windows
- no class-count explanation for the difficult subject

Subject 5 showed unusually high gyroscope activity while its accelerometer magnitude statistics were relatively normal:

- Subject 5 gyro magnitude mean: **0.4641**
- Subject 5 gyro median: **0.1888**
- Subject 5 gyro 75th percentile: **0.7935**
- Subject 5 gyro 95th percentile: **1.6006**

This identifies Subject 5 as an unusual sensor/domain case, but does **not** establish that the dataset is invalid.

## 5. PAMAP2 Model-Suite Experiment

The same subject-independent framework was used to compare:

- RBF-SVM
- ANN/MLP
- Tiny CNN
- BiLSTM
- BiGRU

Fold accuracies:

**RBF-SVM:** 76.91%, 78.00%, 74.40%, 53.60%, 89.35%

**ANN/MLP:** 86.91%, 80.96%, 71.52%, 39.61%, 82.20%

**Tiny CNN:** 94.78%, 82.79%, 65.54%, 50.97%, 79.40%

**BiLSTM:** 76.10%, 63.05%, 60.06%, 44.46%, 76.10%

**BiGRU:** stopped manually because CPU training became excessively slow. Only Fold 1 (**94.71%**) and Fold 2 (**81.38%**) completed. No overall BiGRU score is reported.

The major observation is the large variation between held-out subjects. For example, Tiny CNN ranged from **94.78% to 50.97%**.

## 6. Main Finding

Different model families repeatedly show:

1. high performance on some held-out subjects;
2. substantial degradation on other held-out subjects;
3. persistent stationary-state confusion;
4. easier recognition of dynamic activities.

Because this pattern appears across substantially different algorithms, it is unlikely to be explained solely by model architecture.

The evidence therefore supports a **dataset/domain/task-design contribution** to the difficulty.

## 7. SafeBand Activity Taxonomy

The original RESTING label is ambiguous for SafeBand.

A more useful activity vocabulary is:

```text
ACTIVITY STATE
├── STATIONARY
│   ├── SITTING
│   ├── STANDING
│   └── LYING / SLEEPING
│
├── DYNAMIC
│   ├── WALKING
│   ├── RUNNING
│   └── STAIRS
│
└── OTHER / UNKNOWN
```

Fall/emergency detection remains separate:

```text
SafeBand
├── Activity Recognition
└── Emergency Detection
          ↓
      Sensor Fusion
          ↓
       Risk Score
```

Sleeping should not simply be treated as generic low-motion activity; it is a prolonged behavioral/physiological state that can benefit from physiological sensing as well.

## 8. Revised Dataset Strategy

### Layer 1 — Public datasets

Retain BITS2, PAMAP2 and other suitable public datasets for:

- external benchmarking
- pretraining
- preprocessing validation
- robustness/domain-shift studies

### Layer 2 — SafeBand-specific dataset

Collect controlled data using the actual or closely matched SafeBand sensor configuration.

Priority activities:

- sitting
- standing
- lying
- walking
- running
- stairs
- sit → stand
- stand → sit
- other/unknown movements

Fall/emergency events remain separately labeled.

### Layer 3 — Hardware-domain validation

After the prototype is available:

- collect synchronized sensor streams;
- preserve timestamps;
- preserve subject/session IDs;
- record the sensor configuration;
- test on previously unseen subjects/sessions;
- evaluate realistic wrist-worn conditions.

## 9. Evaluation Principle

Subject-independent evaluation should remain the primary generalization protocol.

The project should avoid using random overlapping windows from the same subject in both training and test as the main result.

The objective is not merely maximum classification accuracy. The activity model should provide reliable evidence to the downstream **sensor-fusion risk engine**.

## 10. Current Decision

**Retain/freeze:**
- BITS2 ACC-only DL V1 as the strongest current activity baseline.
- BITS2 cross-domain experiments as domain-shift evidence.
- PAMAP2 RBF-SVM V2 as the reproducible stationary benchmark.
- PAMAP2 forensic audit.
- Fall detection multidataset V5 pipeline.

**Stop for now:**
- arbitrary CNN/RNN/Transformer architecture searches on the same PAMAP2 stationary dataset;
- treating the incomplete BiGRU run as a final result;
- endless hyperparameter tuning for small gains.

**Next meaningful data step:**

> Design and collect a SafeBand-specific stationary/activity dataset using the actual wrist-sensor configuration and intended activity taxonomy.

## 11. Research Interpretation

> The activity-recognition experiments indicate that stationary-state recognition is strongly affected by subject-specific motion and sensor characteristics. Public datasets are valuable for benchmarking, but they do not necessarily represent the exact sensing conditions and activity taxonomy required by SafeBand. Therefore, public datasets should remain external benchmarks while a SafeBand-specific dataset is introduced for final model development and hardware-domain validation.

This is a **data-design and domain-generalization finding**, not evidence that any particular public dataset is unusable.
