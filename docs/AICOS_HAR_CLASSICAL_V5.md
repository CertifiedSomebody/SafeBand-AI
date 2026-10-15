# AICOS-HAR Classical ML Benchmark V5

## Purpose

Test the hypothesis that classical machine-learning models may be better suited than raw-temporal neural networks for the current ACC-only AICOS-HAR activity-recognition task.

## Why this experiment exists

The previous benchmark produced the following official-test results:

| Model | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| ExtraTrees | 64.755% | 45.449% | 43.336% |
| Random Forest | 60.922% | 51.223% | **43.940%** |
| HistGradientBoosting | 63.632% | 46.325% | 42.747% |
| RBF-SVM | 53.401% | **54.111%** | 39.578% |
| TinyCNN | 51.648% | 52.278% | 38.715% |
| DAE V1 | 61.218% | 51.306% | 43.134% |

The V5 experiment broadens the classical baseline without changing the dataset split or touching the official test data during model development.

## Feature contract

The benchmark uses 38 deterministic ACC features from each `(3, 100)` window:

- per-axis mean, standard deviation, minimum, maximum and RMS
- acceleration-magnitude statistics and percentiles
- mean absolute acceleration, magnitude range
- jerk/magnitude-difference statistics
- difference energy
- magnitude zero-crossing count
- pairwise axis correlations
- three magnitude FFT energy bands

Correlation is computed with an explicit zero-variance guard. Non-finite feature values cause the trainer to fail rather than being silently converted.

## Evaluation protocol

- 346,788 total windows
- 249,363 training/development windows
- 97,425 official test windows
- 106 participants
- official `Fold=test` remains untouched until final evaluation
- fixed random seed: 42
- primary comparison metric: Macro-F1
- secondary metrics: balanced accuracy and accuracy

## Models

The benchmark uses fixed, non-test-tuned configurations for:

1. Logistic Regression
2. Linear SVM
3. Decision Tree
4. Random Forest
5. ExtraTrees
6. HistGradientBoosting
7. RBF-SVM

## Interpretation rule

A classical model is considered a meaningful improvement only if it improves Macro-F1 on the same official test partition. A higher ordinary accuracy alone is not sufficient.

If classical ML remains ahead of the neural models, that is a valid project finding. The next improvement should then come from adding sensor information (especially gyroscope) rather than repeatedly increasing neural-network complexity.
