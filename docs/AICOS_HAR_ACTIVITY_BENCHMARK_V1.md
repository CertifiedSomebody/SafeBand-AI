# AICOS-HAR Activity Recognition — Benchmark & Decision Log V1

## 1. Dataset audit

AICOS-HAR contains 5,231 acquisitions from 106 participants, with accelerometer, gyroscope, magnetometer and barometer data and 11 activity labels used in this benchmark.

Current ACC window dataset:
- Windows: 346,788
- Shape: `(346788, 3, 100)`
- Sampling target: 50 Hz
- Window: 2 s
- Step: 1 s
- Participants: 106
- Unique acquisitions represented: 5,205
- Official test windows: 97,425
- Training/development windows: 249,363

Raw-window audit V1 passed:
- NaN values: 0
- Inf values: 0
- Windows containing non-finite values: 0
- Zero-standard ACC channels: 0
- Undefined correlation-risk windows: 0

The raw windows are therefore not the source of the earlier `np.corrcoef` warnings. Those warnings originate downstream during handcrafted feature calculation.

## 2. Current benchmark

### Classical ACC-only benchmark

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| ExtraTrees | 64.755% | 45.449% | 43.336% |
| Random Forest | 60.922% | 51.223% | **43.940%** |
| HistGradientBoosting | 63.632% | 46.325% | 42.747% |
| RBF-SVM | 53.401% | **54.111%** | 39.578% |

Interpretation:
- Random Forest is the best ACC-only model by Macro-F1.
- RBF-SVM has the highest balanced accuracy.
- Overall performance is not strong enough to treat the 11-class ACC-only model as a deployment-ready activity recognizer.
- The gap between ordinary accuracy and balanced accuracy reflects both class-frequency effects and the difficulty of the 11-class activity taxonomy.

### TinyCNN ACC-only

Official test:
- Accuracy: 51.648%
- Balanced Accuracy: 52.278%
- Macro-F1: 38.715%

The TinyCNN did not outperform the classical ACC-only models.

Validation Macro-F1 peaked around 0.585, while the official test Macro-F1 was much lower. This indicates a substantial generalization gap between development and official test data.

### Supervised Denoising Autoencoder V1

Architecture:
- ACC XYZ + deterministic acceleration magnitude
- convolutional encoder
- 128-D latent representation
- classification head
- reconstruction decoder
- mild train-only augmentation
- inverse-square-root class weighting
- label smoothing
- training-only normalization
- subject-grouped validation
- early stopping on validation Macro-F1

Best validation Macro-F1:
- Epoch 9: 0.5433

Official test:
- Accuracy: 61.218%
- Balanced Accuracy: 51.306%
- Macro-F1: 43.134%

The DAE did not beat the Random Forest Macro-F1 baseline.

The falling training loss after epoch 9 did not justify continuing training: validation Macro-F1 declined/oscillated after its best value. Early stopping was therefore behaving correctly.

## 3. Current conclusion

The ACC-only AICOS-HAR 11-class problem is difficult.

The experiments do not currently support the conclusion that a larger/deeper neural network will solve the problem. Classical models, TinyCNN and supervised DAE have not produced a clear improvement over the classical baseline.

The next improvement should therefore come from **better sensor information and better task formulation**, not blind model tuning.

## 4. Recommended next experiments

### Priority 1 — ACC + Gyroscope

This is the highest-value next experiment.

AICOS-HAR contains inertial information, while the current benchmark deliberately uses ACC only. The SafeBand hardware also has a BNO055, which provides acceleration and gyroscope information.

Next benchmark:
- ACC + gyro
- resample both to a common 50 Hz grid
- same 2 s / 1 s window contract
- preserve official test partition
- compare a handcrafted classical model and a small temporal CNN
- report accuracy, balanced accuracy, Macro-F1 and per-class F1

This is a meaningful sensor-level improvement rather than another hyperparameter search.

### Priority 2 — Orientation/placement robustness

AICOS-HAR contains many device placements and device types. Placement can substantially change raw axis statistics.

Add orientation-robust features such as:
- acceleration magnitude
- gyro magnitude
- jerk magnitude
- axis-independent energy/statistics
- carefully defined frequency-band energy

Do not simply add dozens of features without an ablation. Keep a clear comparison against the ACC-only baseline.

### Priority 3 — Use AICOS-HAR as an external robustness benchmark for SafeBand

The project's final activity vocabulary does not need to mirror every AICOS-HAR label.

A higher-level SafeBand evaluation can group labels such as:
- STATIONARY: Sitting / Standing / LyingDown
- DYNAMIC: Walking / Running / Upstairs / Downstairs / RampUp / RampDown
- TRANSIT: LiftUp / LiftDown

This must be treated as a separate task from the 11-class benchmark. Do not silently mix the two scores.

### Priority 4 — Only then consider sensor fusion

Once ACC+gyro is benchmarked, add magnetometer only if it provides measurable benefit.

Barometer is particularly interesting for stairs/ramps/lifts, but it should be evaluated as task-specific evidence rather than automatically fed into a giant model.

## 5. What should NOT be done next

- Do not keep training the DAE to epoch 40 just because training loss is falling.
- Do not repeatedly tune learning rate/architecture without adding new information.
- Do not overwrite the current ACC-only baseline.
- Do not touch the official test partition during model selection.
- Do not claim improvement unless Macro-F1/balanced accuracy improves on the same untouched test protocol.
- Do not treat ordinary accuracy alone as the main metric for this 11-class imbalanced task.
- Do not interpret the clean raw-window audit as proof that the entire preprocessing pipeline is correct; handcrafted feature generation still needs its NaN-safe implementation audited.

## 6. Frozen baseline for future comparison

Current best ACC-only reference:

**Random Forest**
- Accuracy: 60.922%
- Balanced Accuracy: 51.223%
- Macro-F1: 43.940%

Current DAE V1:
- Accuracy: 61.218%
- Balanced Accuracy: 51.306%
- Macro-F1: 43.134%

The DAE is therefore **not selected as the deployment model**.

The next model must be compared directly against these frozen numbers.

## 7. Decision

**Freeze AICOS-HAR ACC-only benchmark V1.**

Move to:

> **AICOS-HAR ACC + Gyroscope benchmark V2**

The goal is not to chase an arbitrary accuracy number. The goal is to determine whether adding the sensor modality that SafeBand actually has produces a real, reproducible improvement in subject-independent activity recognition.
