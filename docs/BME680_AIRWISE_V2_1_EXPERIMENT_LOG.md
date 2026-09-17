# SafeBand AI — BME680 AIRWISE V2.1 Experiment Log

## 1. Experiment Identity

- **Project:** SafeBand AI
- **Sensor:** Bosch BME680
- **Dataset:** AIRWISE v1.0.0
- **Experiment:** BME680 AIRWISE V2.1
- **Status:** Evaluated; **not promoted** over V1
- **Primary purpose:** Test whether expanded temporal and nonlinear feature engineering improves environmental pollution-class recognition without target leakage.

---

## 2. Dataset

The AIRWISE preparation pipeline produced:

- **146,538 rows**
- **3 locations:** office, hallway, kitchen
- **Date range:** 2025-11-14 to 2025-12-18
- **Input:** minute-level BME680-derived environmental data
- **Processed file:**
  `datasets/processed/airwise_bme680/airwise_bme680_features.csv`

Class distribution:

| IAQ class | Rows |
|---|---:|
| Extremely polluted | 68,159 |
| Severely polluted | 34,904 |
| Excellent | 13,250 |
| Moderately polluted | 8,042 |
| Lightly polluted | 7,986 |
| Heavily polluted | 7,641 |
| Good | 6,556 |

The three locations contain approximately 48.8k rows each.

---

## 3. V2.1 Objective

V2.1 was created to investigate whether the V1 model could be improved using richer representations of the BME680 time series.

The experiment added:

- causal temporal statistics
- temporal differences
- percentage changes
- gas-resistance logarithmic representation
- environmental interaction features
- cyclic temporal features
- expanded nonlinear model capacity

The objective was **not** to force a target accuracy. Model promotion was based on held-out generalization performance.

---

## 4. Leakage Controls

The following controls were explicitly enforced:

1. `IAQ_class` was never used as an input feature.
2. `IAQ_proxy` was excluded from the primary V2.1 model.
3. Future temporal values were not used to construct features.
4. The chronological split was preserved.
5. The test set was not used for model selection.
6. Feature engineering was checked for forbidden columns.
7. Artifact validation confirmed the final model contract.

The artifact validator returned:

```text
test_used_for_selection       false
iaq_proxy_used_as_input      false
future_temporal_values_used  false
label_column_used_as_input   false
forbidden_feature_columns    []
status                        PASS
```

These checks are important because an apparently large performance improvement caused by label-derived information would not constitute a valid model improvement.

---

## 5. Split Protocol

The established AIRWISE protocol uses a:

**60/20/20 chronological split within each location**

This preserves the temporal nature of the environmental data and prevents later observations from being used to train the model for earlier observations.

The test set is therefore a future chronological holdout relative to training data.

---

## 6. V2.1 Model Selection

Candidate models were evaluated using the validation set.

Selection priority:

1. validation macro-F1
2. validation balanced accuracy
3. validation accuracy

The selected V2.1 model was:

**Extra Trees**

with:

- **91 input features**
- validation macro-F1: **0.4096989**

The relatively low validation macro-F1, despite strong performance on common classes, indicates that class imbalance and separation among several pollution categories remain important difficulties.

---

## 7. Held-Out Test Results

V2.1 final test results:

| Metric | Result |
|---|---:|
| Accuracy | **90.47%** |
| Balanced accuracy | **77.64%** |
| Macro precision | **74.51%** |
| Macro recall | **77.64%** |
| Macro F1 | **75.85%** |

Per-class F1:

| Class index | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0 | 98.22% | 94.00% | 96.06% |
| 1 | 97.14% | 96.50% | 96.82% |
| 2 | 79.95% | 79.09% | 79.51% |
| 3 | 63.75% | 78.05% | 70.18% |
| 4 | 58.89% | 58.56% | 58.72% |
| 5 | 48.47% | 62.11% | 54.45% |
| 6 | 75.16% | 75.20% | 75.18% |

The confusion matrix shows that the major remaining errors occur among several intermediate pollution categories rather than the two dominant classes.

---

## 8. Comparison With BME680 V1

Previously established V1 held-out test performance:

- Accuracy: approximately **91.40%**
- Balanced accuracy: approximately **79.47%**
- Macro precision: approximately **77.37%**
- Macro recall: approximately **79.47%**
- Macro F1: approximately **78.24%**

V2.1:

- Accuracy: **90.47%**
- Balanced accuracy: **77.64%**
- Macro precision: **74.51%**
- Macro recall: **77.64%**
- Macro F1: **75.85%**

### Interpretation

V2.1 **did not improve held-out generalization** over V1.

In particular, V2.1 was lower on:

- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1

Therefore V2.1 is retained as an experiment and negative/ablation result, while **V1 remains the primary BME680 reference model**.

---

## 9. Research Interpretation

The experiment demonstrates that adding temporal and nonlinear features does not automatically improve cross-temporal environmental classification.

The result is useful because the experiment was performed with explicit leakage controls and an untouched chronological test set. The lower V2.1 result therefore should not be hidden or replaced by a more favorable metric.

The current evidence supports:

> BME680 AIRWISE V1 remains the primary Phase-1 reference model. V2.1 explored richer temporal and nonlinear representations but did not outperform V1 on the held-out chronological test set and was consequently not promoted.

---

## 10. Why We Do Not Chase “95% Accuracy”

The project should not treat an arbitrary 95% classification-accuracy threshold as the definition of research quality.

For this seven-class environmental classification task, accuracy alone can obscure poor performance on less frequent classes. Balanced accuracy and macro-F1 are therefore essential companion metrics.

Research-quality reporting should emphasize:

- reproducible preprocessing
- leakage-free feature construction
- temporal/subject/environment-aware validation where appropriate
- untouched test evaluation
- class-balanced metrics
- per-class performance
- transparent negative results
- later validation on actual SafeBand hardware

---

## 11. Phase-1 Decision

### Promoted

`models/bme680_airwise_v1/`

### Retained as experiment

`models/bme680_airwise_v2/`

### Current BME680 status

**FROZEN FOR PHASE 1**

Further AIRWISE hyperparameter tuning is not currently justified by the V2.1 evidence.

The next meaningful improvement should come from:

1. actual BME680 hardware recordings from the SafeBand prototype,
2. synchronized multi-sensor data,
3. hardware-domain validation,
4. and eventual sensor-fusion integration.

---

## 12. Reproduction Commands

Prepare the dataset:

```powershell
python tools\prepare_airwise_bme680.py
```

Audit:

```powershell
python tools\audit_airwise_v2.py
```

Train V2.1:

```powershell
python tools\train_airwise_bme680_v2.py
```

Validate artifacts:

```powershell
python tools\validate_bme680_v2.py
```

Expected processed dataset:

```text
datasets\processed\airwise_bme680\airwise_bme680_features.csv
```

---

## 13. Final Status

**BME680 AIRWISE V2.1 — COMPLETE / NOT PROMOTED**

V2.1 is a valid, leakage-controlled experiment. It does not replace the established V1 model.

The branch is considered sufficiently investigated for the current Phase-1 stage, allowing development effort to move toward hardware validation and SafeBand sensor-fusion integration.
