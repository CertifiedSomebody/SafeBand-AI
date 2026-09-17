# PPG V4.1 Experiment Log

## Why V4.1 exists

The first V4 run successfully produced a MAX30101-domain HR model, but its
implementation needed a stronger audit layer before being treated as the
final V4 pipeline.

V4.1 therefore:

- keeps the original V4 experiment as historical evidence,
- fixes deterministic subject ordering,
- adds explicit preparation validation,
- adds complete held-out diagnostics,
- records test predictions,
- records interpretable regression metrics,
- preserves the validation-only calibration rule.

## First V4 result

The first V4 report selected HGB on validation MAE.

Historical first-V4 values:

- validation MAE: 1.467 BPM
- held-out test MAE: 4.170 BPM

These values belong to the original V4 split and must not be silently mixed
with V4.1 results.

## V4.1 experiment

V4.1 is a fresh experiment because the deterministic split definition changed
from lexical subject ordering to numeric subject ordering.

The expected split for 22 subjects is:

- train: s1–s13
- validation: s14–s17
- test: s18–s22

All three partitions are subject-disjoint.

## Evaluation philosophy

Primary model-selection metric:

```text
validation MAE
```

Final research evaluation:

```text
held-out subject test MAE
held-out subject RMSE
held-out subject R²
within ±3 BPM
within ±5 BPM
within ±10 BPM
bias
median absolute error
P90 absolute error
maximum absolute error
```

Activity and subject diagnostics are required before drawing conclusions.

## Calibration

Validation-only affine calibration is evaluated.

If:

```text
calibrated validation MAE < raw validation MAE
```

then calibration is enabled.

Otherwise it is disabled.

No test labels are used to decide this.

## Interpretation

The model should not be called "accurate" from one score alone.

A healthy result should show:

- low held-out MAE,
- RMSE not dramatically larger than MAE,
- high within-tolerance percentages,
- bias reasonably near zero,
- no single activity/subject dominating the error,
- a meaningful improvement over the constant train-mean baseline.

A large validation-to-test degradation is a signal for further generalization
analysis, not something to hide.

## Next hardware stage

After V4.1, collect actual MAX30102 recordings.

The MAX30102 dataset should be evaluated independently before any calibration is
fitted. Calibration/fine-tuning must use a clearly defined development subset,
while a separate MAX30102 hold-out set remains untouched for final hardware
validation.
