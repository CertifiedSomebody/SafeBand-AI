# SafeBand AI — BME680 AIRWISE V1 Experiment Documentation

## 1. Experiment Overview

**Branch:** BME680 environmental sensing / IAQ classification  
**Dataset:** AIRWISE v1.0.0  
**Task:** Multi-class indoor IAQ classification  
**Experiment status:** **FROZEN BASELINE**

This experiment establishes the first machine-learning benchmark for the SafeBand AI BME680 environmental-sensing branch. The objective is to determine whether BME680-derived environmental measurements can support classification of indoor air-quality categories.

The experiment uses AIRWISE data collected with Bosch BME680 sensors and evaluates several supervised machine-learning models using a chronological train/validation/test protocol.

---

## 2. Dataset

AIRWISE v1.0.0 contains minute-level indoor measurements from three environments:

- Office
- Kitchen
- Hallway

The dataset contains:

- Temperature
- Pressure
- Relative humidity
- Gas resistance
- Sensor/quality-related information
- IAQ-related class annotations

### Dataset size

| Split | Rows |
|---|---:|
| Training | 87,922 |
| Validation | 29,307 |
| Held-out test | 29,309 |
| **Total** | **146,538** |

The split is performed **chronologically within each location**, using a 60/20/20 allocation.

This prevents future observations from being used to train the model while evaluating earlier observations.

---

## 3. Prediction Target

The V1 task predicts the AIRWISE `IAQ_class` label.

The dataset contains seven IAQ classes:

- Excellent
- Good
- Lightly polluted
- Moderately polluted
- Heavily polluted
- Severely polluted
- Extremely polluted

The classes are not uniformly distributed. Therefore, accuracy alone is not considered sufficient for evaluating the classifier.

The primary evaluation measures include:

- Accuracy
- Balanced accuracy
- Macro precision
- Macro recall
- Macro F1
- Confusion matrix

---

## 4. Feature Engineering

The V1 pipeline uses **24 features**.

### Direct sensor measurements

- `temperature_c`
- `pressure_hpa`
- `humidity_rh`
- `gas_resistance_ohms`
- `heat_stable`

### Gas and environmental transformations

- `gas_log10`
- `temp_humidity_product`
- `temp_humidity_gap`

### Temporal features

For temperature, pressure, humidity, and gas resistance:

- First difference
- Percentage change
- 5-minute rolling mean
- 5-minute rolling standard deviation

The complete feature set is recorded in the experiment report.

---

## 5. Models Compared

Four classifiers were evaluated:

1. Logistic Regression
2. Random Forest
3. Extra Trees
4. HistGradientBoosting (HGB)

Model selection was performed using the validation set.

The selected model was:

**HistGradientBoosting (HGB)**

---

## 6. Validation Results

The validation results showed substantially different behavior between the candidate models.

### Logistic Regression

- Accuracy: **84.36%**
- Balanced accuracy: **68.50%**
- Macro precision: **41.85%**
- Macro recall: **51.37%**
- Macro F1: **43.23%**

### Random Forest

- Accuracy: **98.69%**
- Balanced accuracy: **98.71%**
- Macro precision: **57.08%**
- Macro recall: **59.23%**
- Macro F1: **58.08%**

### Extra Trees

- Accuracy: **95.09%**
- Balanced accuracy: **89.26%**
- Macro precision: **64.73%**
- Macro recall: **66.95%**
- Macro F1: **65.52%**

### HistGradientBoosting

- Accuracy: **99.05%**
- Balanced accuracy: **98.99%**
- Macro precision: **55.36%**
- Macro recall: **59.39%**
- Macro F1: **57.17%**

HGB was selected according to the experiment's validation model-selection procedure.

---

## 7. Held-Out Test Results

After model selection, the selected HGB model was evaluated on the untouched chronological test set.

| Metric | Held-out test |
|---|---:|
| Accuracy | **91.40%** |
| Balanced accuracy | **79.47%** |
| Macro precision | **77.37%** |
| Macro recall | **79.47%** |
| Macro F1 | **78.24%** |

### Confusion matrix

The held-out test confusion matrix contains predictions across all seven IAQ classes and is preserved in the generated experiment report.

The matrix demonstrates that the model distinguishes several classes well, while some neighboring pollution categories remain more difficult to separate.

---

## 8. Validation vs. Test Behavior

A notable characteristic of this experiment is the difference between validation and held-out test performance.

The selected HGB model achieved approximately:

- **98.99% balanced accuracy on validation**
- **79.47% balanced accuracy on held-out test**

This indicates a meaningful distribution/temporal shift between the development period and the later held-out period.

This behavior is important and is retained in the results rather than being hidden through additional tuning.

The held-out test result should therefore be treated as the primary indication of expected performance on unseen chronological AIRWISE data.

---

## 9. Interpretation

The V1 experiment establishes that BME680 environmental measurements contain useful information for predicting the AIRWISE IAQ categories.

At the same time, the held-out test results show that the problem is not solved by the current feature set and model alone.

The result should be interpreted as a **dataset-specific IAQ classification benchmark**, not as evidence that the SafeBand can currently determine a user's personal health or emergency status from BME680 measurements.

In particular:

- AIRWISE is an indoor environmental dataset.
- The target is an IAQ classification label.
- The experiment does not establish clinical thresholds.
- The experiment does not establish personal exposure risk.
- The experiment does not replace actual SafeBand hardware validation.

---

## 10. Why the Experiment Is Frozen

The purpose of V1 is to establish a reproducible baseline.

The following elements are now fixed for this experiment:

- AIRWISE dataset
- Dataset audit
- 24-feature representation
- Chronological 60/20/20 split
- Candidate model comparison
- Validation-based model selection
- Untouched held-out test evaluation
- Confusion-matrix analysis

Further small increases in AIRWISE benchmark performance are not currently considered the primary project objective.

The next meaningful validation step is **SafeBand hardware-domain testing using the actual BME680 sensor configuration**.

---

## 11. Reproducibility and Path Handling

The AIRWISE extraction contained an additional nested dataset directory:

`datasets/raw/AIRWISE/AIRWISE-v1.0.0/data/indoor/sensor_1min`

The initial versions of the preparation and training commands assumed:

`datasets/raw/AIRWISE/data/indoor/sensor_1min`

This caused file-not-found errors.

The scripts were subsequently corrected and the complete pipeline was executed successfully.

### Engineering rule for future SafeBand tools

All future `tools/*.py` scripts must:

1. Work when invoked from the repository root.
2. Resolve the repository root explicitly.
3. Avoid assuming archive extraction structure when practical.
4. Be tested by actually running the command, not only by syntax-checking or compiling the script.

Typical invocation:

```powershell
python tools\prepare_airwise_bme680.py
python tools\train_airwise_bme680.py
```

---

## 12. Final V1 Status

**BME680 AIRWISE V1: COMPLETE — BASELINE FROZEN**

The experiment provides a documented and reproducible starting point for the SafeBand BME680 environmental-sensing branch.

### Key final result

**HGB held-out chronological test:**

- Accuracy: **91.40%**
- Balanced accuracy: **79.47%**
- Macro precision: **77.37%**
- Macro recall: **79.47%**
- Macro F1: **78.24%**

These values should be used as the V1 reference point for future BME680 experiments.

---

## 13. Source Artifacts

Recommended repository artifacts:

```text
datasets/
├── raw/
│   └── AIRWISE/
└── processed/
    └── airwise_bme680/

ai/
└── bme680_features.py

tools/
├── audit_airwise.py
├── prepare_airwise_bme680.py
├── train_airwise_bme680.py
└── predict_airwise_bme680.py

docs/
├── BME680_AIRWISE_V1.md
├── BME680_AIRWISE_V1_EXPERIMENT_LOG.md
└── BME680_AIRWISE_V1_REQUIREMENTS.md

models/
└── airwise_bme680/
```

The generated JSON report and test predictions should also be retained as experiment artifacts where applicable.
