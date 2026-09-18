# SafeBand AI — BNO055 ML Benchmark V1 Results

## 1. Purpose

This benchmark evaluates activity classification using the frozen synthetic BNO055 V2.1 window dataset.

The benchmark uses subject-independent evaluation and compares:

- Sensor/channel configurations:
  - ACC
  - ACC + GYRO
  - ACC + GYRO + MAG
- Classical machine-learning models using deterministic window features.
- Small deep-learning models using the raw temporal sensor windows.

The benchmark is a software/synthetic-data study only. It is **not real BNO055 hardware validation**.

---

## 2. Dataset and evaluation protocol

Dataset:

```text
datasets/synthetic/bno055/bno055_windows_v2_1.npz
```

Preflight results:

- Windows: 5,040
- Window shape: `(5040, 9, 200)`
- Subjects: 20
- Classes: 9
- Samples per class: 560
- Event-overlap windows: 433
- Nonfinite values: 0
- Timing: 100 Hz
- Quaternion norm: approximately 1.0

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

Evaluation:

- 5-fold `StratifiedGroupKFold`
- Grouped by subject
- Test folds are kept outside model selection
- Preprocessing is fitted on outer training data only
- Pooled outer predictions are reported
- Event-overlap windows are reported separately

---

# 3. Sensor Ablation — Random Forest

The first experiment compares sensor modalities using the same Random Forest feature-based classifier.

| Configuration | Accuracy | Balanced Accuracy | Macro-F1 | Event Macro-F1 |
|---|---:|---:|---:|---:|
| ACC | 69.38% | 69.38% | 69.27% | 24.33% |
| **ACC + GYRO** | **72.54%** | **72.54%** | **72.51%** | 24.18% |
| ACC + GYRO + MAG | 72.10% | 72.10% | 71.87% | 22.71% |

### ACC-only

Fold results:

| Fold | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 1 | 70.34% | 70.34% | 70.11% |
| 2 | 66.77% | 66.77% | 66.60% |
| 3 | 64.98% | 64.98% | 64.85% |
| 4 | 70.93% | 70.93% | 69.82% |
| 5 | 73.91% | 73.91% | 73.81% |

Pooled:

```text
Accuracy             69.38%
Balanced Accuracy    69.38%
Macro-F1             69.27%
Event Macro-F1       24.33%
```

### ACC + GYRO

Fold results:

| Fold | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 1 | 72.52% | 72.52% | 72.59% |
| 2 | 71.92% | 71.92% | 71.32% |
| 3 | 71.33% | 71.33% | 71.91% |
| 4 | 72.92% | 72.92% | 72.05% |
| 5 | 74.01% | 74.01% | 73.87% |

Pooled:

```text
Accuracy             72.54%
Balanced Accuracy    72.54%
Macro-F1             72.51%
Event Macro-F1       24.18%
```

Relative to ACC-only:

```text
Accuracy      +3.16 percentage points
Macro-F1      +3.24 percentage points
```

The fold range also becomes narrower:

```text
ACC:          64.98% — 73.91%
ACC + GYRO:   71.33% — 74.01%
```

### ACC + GYRO + MAG

Fold results:

| Fold | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| 1 | 71.92% | 71.92% | 71.70% |
| 2 | 70.63% | 70.63% | 70.12% |
| 3 | 69.84% | 69.84% | 70.30% |
| 4 | 74.01% | 74.01% | 73.14% |
| 5 | 74.11% | 74.11% | 73.85% |

Pooled:

```text
Accuracy             72.10%
Balanced Accuracy    72.10%
Macro-F1             71.87%
Event Macro-F1       22.71%
```

Compared with ACC + GYRO:

```text
Accuracy      -0.44 percentage points
Macro-F1      -0.64 percentage points
Event F1      -1.47 percentage points
```

### Sensor-ablation interpretation

Within this frozen synthetic benchmark:

- Adding GYRO to ACC produces a clear improvement.
- Adding MAG on top of ACC + GYRO does not improve the Random Forest activity classifier.
- The magnetometer should therefore not be claimed as an activity-classification improvement from this experiment.
- This result does not establish that the magnetometer has no value elsewhere in the SafeBand system; this benchmark only evaluates its contribution to this activity-classification pipeline.

---

# 4. Deep-Learning Model Comparison

The neural models were evaluated using the strongest sensor configuration identified by the initial RF ablation: ACC + GYRO.

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Event Macro-F1 |
|---|---:|---:|---:|---:|
| **RF** | **72.54%** | **72.54%** | **72.51%** | **24.18%** |
| CNN | 61.03% | 61.03% | 60.76% | 17.41% |
| TCN | 63.27% | 63.27% | 62.99% | 21.33% |

## CNN

Fold results:

```text
Fold 1: 56.45% accuracy, 54.76% macro-F1
Fold 2: 65.38% accuracy, 64.09% macro-F1
Fold 3: 60.32% accuracy, 59.43% macro-F1
Fold 4: 62.20% accuracy, 60.81% macro-F1
Fold 5: 60.81% accuracy, 60.62% macro-F1
```

Pooled:

```text
Accuracy             61.03%
Balanced Accuracy    61.03%
Macro-F1             60.76%
Event Macro-F1       17.41%
```

Relative to RF:

```text
Accuracy      -11.51 percentage points
Macro-F1      -11.75 percentage points
Event F1       -6.77 percentage points
```

## TCN

Fold results:

```text
Fold 1: 60.81% accuracy, 60.33% macro-F1
Fold 2: 68.06% accuracy, 66.57% macro-F1
Fold 3: 64.19% accuracy, 64.21% macro-F1
Fold 4: 59.72% accuracy, 58.20% macro-F1
Fold 5: 63.59% accuracy, 64.33% macro-F1
```

Pooled:

```text
Accuracy             63.27%
Balanced Accuracy    63.27%
Macro-F1             62.99%
Event Macro-F1       21.33%
```

Relative to RF:

```text
Accuracy      -9.27 percentage points
Macro-F1      -9.52 percentage points
Event F1      -2.85 percentage points
```

The TCN also required substantially more computation. Its five folds took approximately 399–680 seconds each.

---

# 5. Classical ML Model Suite

After the initial RF/CNN/TCN benchmark, a dedicated classical ML comparison was run using the same:

```text
ACC + GYRO
```

configuration and the same subject-independent 5-fold protocol.

## Results

| Classical Model | Accuracy | Balanced Accuracy | Macro-F1 | Event Macro-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 71.13% | 71.13% | 70.85% | 21.88% |
| k-NN | 59.74% | 59.74% | 60.01% | 17.95% |
| Linear SVM | 70.08% | 70.08% | 69.54% | 21.40% |
| Decision Tree | 67.02% | 67.02% | 67.11% | 19.58% |
| Extra Trees | 71.53% | 71.53% | 71.40% | 22.97% |
| RBF SVM | 71.07% | 71.07% | 71.02% | 23.22% |
| HistGradientBoosting | 72.50% | 72.50% | **72.54%** | 23.10% |
| **Random Forest** | **72.54%** | **72.54%** | 72.51% | **24.18%** |

## Logistic Regression

```text
Accuracy             71.13%
Balanced Accuracy    71.13%
Macro-F1             70.85%
Event Macro-F1       21.88%
```

## k-NN

```text
Accuracy             59.74%
Balanced Accuracy    59.74%
Macro-F1             60.01%
Event Macro-F1       17.95%
```

## Linear SVM

```text
Accuracy             70.08%
Balanced Accuracy    70.08%
Macro-F1             69.54%
Event Macro-F1       21.40%
```

## Decision Tree

```text
Accuracy             67.02%
Balanced Accuracy    67.02%
Macro-F1             67.11%
Event Macro-F1       19.58%
```

## Extra Trees

```text
Accuracy             71.53%
Balanced Accuracy    71.53%
Macro-F1             71.40%
Event Macro-F1       22.97%
```

## RBF SVM

```text
Accuracy             71.07%
Balanced Accuracy    71.07%
Macro-F1             71.02%
Event Macro-F1       23.22%
```

## HistGradientBoosting

```text
Accuracy             72.50%
Balanced Accuracy    72.50%
Macro-F1             72.54%
Event Macro-F1       23.10%
```

---

# 6. Complete Model Comparison

Using ACC + GYRO:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| k-NN | 59.74% | 60.01% |
| CNN | 61.03% | 60.76% |
| TCN | 63.27% | 62.99% |
| Decision Tree | 67.02% | 67.11% |
| Linear SVM | 70.08% | 69.54% |
| RBF SVM | 71.07% | 71.02% |
| Logistic Regression | 71.13% | 70.85% |
| Extra Trees | 71.53% | 71.40% |
| HistGradientBoosting | 72.50% | **72.54%** |
| **Random Forest** | **72.54%** | 72.51% |

The Random Forest and HistGradientBoosting results are essentially tied at the displayed precision:

```text
RF:
Accuracy   72.54%
Macro-F1   72.51%

HGB:
Accuracy   72.50%
Macro-F1   72.54%
```

No statistical significance test was performed, so a difference of a few hundredths of a percentage point should not be treated as meaningful.

---

# 7. Event-Overlap Subset

The benchmark separately evaluates 433 windows overlapping marked events.

Results:

| Model | Event Macro-F1 |
|---|---:|
| k-NN | 17.95% |
| CNN | 17.41% |
| Decision Tree | 19.58% |
| TCN | 21.33% |
| Linear SVM | 21.40% |
| Logistic Regression | 21.88% |
| HistGradientBoosting | 23.10% |
| Extra Trees | 22.97% |
| RBF SVM | 23.22% |
| **Random Forest** | **24.18%** |
| ACC-only RF | 24.33% |

The event-overlap subset should not be interpreted as the final fall/emergency detector. The SafeBand architecture treats emergency/fall detection as a separate branch from ordinary activity recognition.

---

# 8. Current Benchmark Findings

## 8.1 Sensor finding

The strongest tested RF configuration is:

```text
ACC + GYRO
```

Adding GYRO to ACC improved pooled accuracy by 3.16 percentage points.

Adding MAG to ACC + GYRO did not improve the activity-classification result.

## 8.2 Model-family finding

For this synthetic BNO055 benchmark:

- Classical ML substantially outperformed the tested CNN and TCN.
- The strongest classical models were Random Forest and HistGradientBoosting.
- RBF SVM and Extra Trees were competitive but lower.
- Simple linear models were useful baselines but did not exceed the strongest tree ensembles.
- k-NN and the tested neural models were substantially weaker.

## 8.3 Computational finding

Random Forest completed each fold in roughly 13–14 seconds for ACC + GYRO.

The TCN required roughly 399–680 seconds per fold while producing lower performance.

Therefore, the tested TCN does not provide a practical performance advantage in this benchmark.

---

# 9. Benchmark Decision

The current synthetic-data benchmark configuration to carry forward is:

```text
BNO055 activity baseline
        ↓
ACC + GYRO
        ↓
Classical window features
        ↓
Random Forest
```

HistGradientBoosting is retained as a near-tied alternative/reference because its pooled macro-F1 is 72.54%.

No further architecture tuning is justified on the frozen V2.1 synthetic dataset based on these experiments.

---

# 10. Limitations

This benchmark has several important limitations.

1. The dataset is synthetic BNO055-shaped data, not recordings from a physical BNO055 mounted on a human wrist.
2. Results therefore measure performance in the simulated sensor domain.
3. The benchmark contains nine classes including FALL and posture transitions, whereas the eventual SafeBand architecture separates ordinary activity recognition from emergency/fall detection.
4. The event-overlap subset contains only 433 windows and represents a difficult temporal subset; it is not a standalone emergency-detection benchmark.
5. No significance testing or confidence intervals were calculated.
6. The benchmark does not establish real-world generalization to new users, hardware mounting variations, environmental magnetic disturbances, or sensor noise beyond what is represented in the synthetic generator.
7. The magnetometer result is specific to this classifier/feature pipeline and should not be generalized to the overall usefulness of BNO055 magnetometer data.

---

# 11. Next Validation Stage

The synthetic benchmark should now be treated as a software baseline.

The next meaningful validation is with real BNO055 recordings:

```text
Real BNO055
    ↓
ACC + GYRO (+ MAG retained as auxiliary data)
    ↓
Synchronized labeled windows
    ↓
Subject-independent evaluation
    ↓
Compare against synthetic-trained baseline
    ↓
Hardware-domain validation
```

The final SafeBand system should then integrate activity evidence with the separate fall/emergency detector and other sensor branches through the planned sensor-fusion/risk engine.

---

## Status

**BNO055 Synthetic ML Benchmark V1: COMPLETE**

- Dataset V2.1: frozen
- Dataset audit: passed
- Sensor ablation: complete
- Classical ML comparison: complete
- CNN comparison: complete
- TCN comparison: complete
- Current classical baseline: RF + ACC/GYRO
- Real-hardware validation: pending
