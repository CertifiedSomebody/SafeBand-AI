# SafeBand Activity Recognition — Cross-Dataset Validation V1.2

## Purpose

This experiment tests whether an ACC-only activity representation transfers between **FORTH-TRACE** and **BITS2** without using target-domain data for model selection.

The experiment is intentionally conservative. Only genuinely shared labels are used:

- `SITTING`
- `WALKING`

No semantic substitutions are made between datasets. In particular:

- `RESTING` is **not** mapped to `STANDING`.
- `RUNNING` is **not** mapped to `STAIRS`.

## Common feature contract

Both domains use the same 37 engineered accelerometer features from `ai/common_acc_features.py`.

Target representation:

- 20 Hz
- 40 samples/window
- 2.0 s window
- 20 samples/1.0 s step
- ACC x/y/z + magnitude-derived time/frequency features

### FORTH-TRACE preprocessing correction

FORTH-TRACE is recorded at 51.2 Hz. A target 2.0 s window therefore corresponds to approximately 102 raw samples. V1 incorrectly attempted to take 40 raw samples (~0.78 s) and then resample them, which could produce zero windows.

V1.2 fixes this by taking approximately 102 raw samples per target 40-sample window and approximately 51 raw samples per 1-second step, then resampling each window to exactly 40 samples at 20 Hz.

The builder now fails fast instead of silently saving an empty feature artifact.

## Model selection

Source-domain model selection uses 5-fold `StratifiedGroupKFold`, grouped by subject. The target domain is never used for model selection.

Models:

- ExtraTrees
- Random Forest
- RBF-SVM
- HistGradientBoosting for FORTH within-domain CV

Primary selection metric:

1. macro-F1
2. balanced accuracy
3. accuracy

Estimators are cloned for every fold so no fitted state is reused between folds.

## Cross-domain evaluation

Two directions are evaluated:

1. FORTH-TRACE → BITS2
2. BITS2 → FORTH-TRACE

The selected source model is retrained on the full source domain and evaluated once on the untouched target domain.

The transfer report additionally records:

- majority-class baseline accuracy
- accuracy delta versus majority baseline
- chance balanced accuracy (0.50 for the binary task)
- predicted class counts
- whether the model collapsed to one predicted class

This prevents a high raw accuracy caused by class imbalance from being mistaken for useful transfer.

## Current observed result

The first valid V1.2 run produced severe cross-domain degradation:

| Direction | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---:|---:|---:|
| FORTH-TRACE → BITS2 | 40.15% | 50.66% | 29.94% |
| BITS2 → FORTH-TRACE | 62.71% | 50.00% | 38.54% |

FORTH-TRACE → BITS2 strongly favored `SITTING`, while BITS2 → FORTH-TRACE predicted `WALKING` for the target set. These are domain-shift diagnostics, not deployment-quality SafeBand performance.

The result is consistent with the broader HAR literature: sensor placement, sampling characteristics, acquisition protocols, and other dataset heterogeneities can materially affect cross-dataset activity recognition. citeturn0search0turn0search1

## Interpretation for SafeBand

The project should **not** claim that the ~99% FORTH-TRACE within-domain score is a universal activity-recognition result.

Instead:

- BITS2 remains the project-specific activity benchmark for the current wrist-device domain.
- FORTH-TRACE remains an external benchmark.
- Cross-dataset transfer is retained as a domain-generalization stress test.
- Real SafeBand hardware data remain the decisive deployment-domain validation.

The purpose of this experiment is therefore to learn where generalization fails and to avoid overclaiming from a single-dataset score.


## V1.2 corrections after first real execution

The distributed FORTH-TRACE CSVs can encode activity labels as numeric codes rather than the textual names shown in the README. The common-label builder therefore accepts both forms: codes 2/3 map to SITTING and 4/5 map to WALKING. The builder also refuses to replace the existing artifact until a complete non-empty two-class build succeeds. The CV and transfer scripts verify the V1.2 audit contract so a stale artifact cannot silently be reused after a failed rebuild. The batch runner stops immediately on a build/CV error.

The cross-domain diagnostics intentionally expose majority-class collapse rather than hiding it. This is consistent with published wearable HAR work showing that sensor position, sampling rate, measurement units, and acquisition protocols can create substantial heterogeneity across datasets.
