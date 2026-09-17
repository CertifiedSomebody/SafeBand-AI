# PPG Walking V5.2 Experiment Log

## Starting point

V4.3 walking benchmark:
- MAE 6.497 BPM
- ±5 BPM 53.17%
- ±10 BPM 80.92%
- bias -6.424 BPM
- R² -0.884

V5.1:
- 1,097 windows / 6 subjects
- PPG+ACC+gyro was best on its single validation subject
- unrestricted validation calibration generalized catastrophically to S9

## V5.2 hypothesis

The walking weakness may be driven by subject-specific motion/PPG relationships. More robust grouped validation should reveal whether a representation generalizes before we touch the final test subject.

## Current experiment

Development: S1, S2, S3, S6, S8
Test: S9

5-fold GroupKFold over development subjects.

### Result

Pending execution.

Do not claim improvement until S9 held-out metrics are available.

## Decision rule

The selected model is the variant/model/calibration combination with the lowest mean out-of-fold MAE. The final comparison is then made against the untouched S9 result and the earlier V4.3 walking benchmark.
