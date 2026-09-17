# Walking HR Dataset Acquisition Plan

## Purpose

PPG V4.3 identified **walking** as the dominant weakness in held-out
evaluation. The next experiment therefore adds an **independent
walking-focused dataset** rather than continuing to tune the existing
PTT/MAX30101 reference benchmark.

The objective is to test whether the SafeBand HR pipeline can
learn/generalize under wrist-motion conditions that specifically stress
PPG motion artifacts.

## Selected Dataset

### Wrist PPG During Exercise --- PhysioNet v1.0.0

Official dataset: PhysioNet, **Wrist PPG During Exercise**, version
1.0.0.

This dataset is particularly suitable because it contains:

-   wrist PPG
-   simultaneous accelerometer measurements
-   simultaneous gyroscope measurements
-   chest ECG reference
-   walking records
-   running records
-   low/high resistance cycling records
-   open access
-   WFDB-format recordings
-   ECG R-peak annotations for reference HR

The dataset contains recordings from **8 participants** (3 male, 5
female), aged 22--32, with most activities lasting approximately 4--6
minutes. Signals were sampled at **256 Hz**.

The walking and running records contain raw PPG and motion signals
without additional filtering beyond the Shimmer hardware. The reference
ECG provides a gold-standard HR comparison.

## Why this dataset

PPG V4.3 showed:

-   overall held-out MAE: 4.576 BPM
-   walking MAE: 6.497 BPM
-   walking ±5 BPM: 53.17%
-   walking ±10 BPM: 80.92%
-   walking bias: -6.42 BPM
-   walking R²: -0.884

Walking is therefore the specific failure regime to investigate.

The PhysioNet Wrist PPG During Exercise dataset is useful because it
isolates the same fundamental problem: wrist PPG contaminated by
exercise-related motion, while providing accelerometer, gyroscope, and
ECG reference signals.

## Research Role

This dataset is **not** a replacement for the PTT/MAX30101 benchmark.

It should be treated as an **external walking-stress dataset**:

``` text
PTT / MAX30101 reference
        +
Wrist PPG During Exercise / walking
        ↓
External robustness evaluation
        ↓
Identify walking-specific failure modes
        ↓
Targeted improvement
        ↓
Re-test on untouched walking subjects
```

The sensor hardware is different from both the planned MAX30102 and the
PTT MAX30101-family reference, so results must be reported as
cross-device validation rather than MAX30102 validation.

## Acquisition

PhysioNet provides direct terminal access. Recommended destination:

``` text
datasets/raw/WristPPGExercise/
```

AWS CLI:

``` powershell
aws s3 sync --no-sign-request `
s3://physionet-open/wrist/1.0.0/ `
"datasets/raw/WristPPGExercise"
```

Alternative direct download:

``` text
https://physionet.org/files/wrist/1.0.0/
```

## Required Audit Before Training

Do not immediately train the existing V4.3 model.

First verify:

1.  exact record list
2.  number of walking records
3.  subject count
4.  signal/channel names
5.  sampling rates from `.hea`
6.  PPG channel(s)
7.  accelerometer channels
8.  gyroscope channels
9.  ECG channel
10. ECG R-peak annotation format
11. record durations
12. missing/invalid samples
13. timestamp/alignment assumptions
14. whether the existing V4 feature extractor can be adapted without
    changing feature semantics

## Experimental Rule

The external walking dataset must remain subject-independent.

No subject from the external walking test set may appear in training.

If the dataset is used for fine-tuning, the protocol must explicitly
separate:

-   development/training subjects
-   validation subjects
-   untouched walking test subjects

A cross-dataset test should also be retained where the model is trained
without the walking dataset and evaluated on it.

## Planned Experiments

### Experiment A --- External evaluation

Use the frozen V4.3 model without retraining.

Purpose:

> Measure how the existing PTT/MAX30101-domain model transfers to a
> separate walking dataset.

### Experiment B --- Walking adaptation

If Experiment A confirms the same failure mode, train a controlled
walking-adapted model.

Compare:

-   existing V4.3 features
-   PPG-only
-   PPG + motion
-   walking-specific training augmentation
-   subject-independent validation

### Experiment C --- Final external test

After all decisions are frozen, evaluate once on untouched walking
subjects.

Primary metrics:

-   MAE
-   RMSE
-   R²
-   within ±3 BPM
-   within ±5 BPM
-   within ±10 BPM
-   bias
-   median absolute error
-   P90/P95 absolute error
-   worst record
-   subject-wise performance

## Important Constraint

This dataset should be used to improve the **walking weakness**, not to
repeatedly tune against the same test subjects.

The final external test must remain untouched until the
model-development decisions are complete.

## Source

Jarchi, D. and Casson, A. J. (2017), "Description of a Database
Containing Wrist PPG Signals Recorded during Physical Exercise with Both
Accelerometer and Gyroscope Measures of Motion", Data 2(1), 1.

PhysioNet dataset: Wrist PPG During Exercise, version 1.0.0.
