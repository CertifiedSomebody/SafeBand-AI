# PPG V4.3 Final Results Addendum

## Status

**PPG V4.3 is frozen as the current MAX30101-domain reference
benchmark.**

The next development target is walking robustness using an independent
external walking dataset.

## Dataset and Split

V4.3 contains:

-   15,982 windows
-   22 subjects
-   159 features
-   93 PPG features
-   66 motion features
-   activities: run, sit, walk

The subject-independent protocol uses numeric 60/20/20 splitting:

-   train: S1--S13
-   validation: S14--S17
-   held-out test: S18--S22

There is no subject overlap between splits.

## Held-Out Test Result

On the untouched S18--S22 test subjects:

  Metric                            Result
  ------------------------ ---------------
  Windows                            3,612
  MAE                        **4.576 BPM**
  RMSE                           5.955 BPM
  R²                                 0.591
  Within ±3 BPM                     37.15%
  Within ±5 BPM                 **69.57%**
  Within ±10 BPM                **92.47%**
  Bias                          -4.407 BPM
  Median absolute error          3.790 BPM
  P90 absolute error             8.919 BPM
  P95 absolute error            11.576 BPM
  Maximum absolute error        34.914 BPM

## Baseline Comparison

The constant train-mean baseline produced:

-   test MAE: 8.048 BPM
-   test R²: -0.061

The direct PPG HR baseline produced:

-   test MAE: 5.297 BPM
-   test R²: -0.677

The V4.3 learned model therefore improves the held-out MAE over both
baselines.

## Controlled PPG vs Motion Ablation

Validation results:

  Configuration               MAE    RMSE          R²
  --------------- --------------- ------- -----------
  PPG-only          **4.600 BPM**   7.424   **0.778**
  PPG + motion          5.456 BPM   7.791       0.756

This experiment does **not** support the assumption that adding motion
features automatically improves HR regression.

The PPG-only configuration performed better on validation. Motion
features should therefore be treated as an experimental component rather
than an assumed improvement.

## Activity-Level Result

  Activity               MAE       ±5 BPM      ±10 BPM        Bias
  ---------- --------------- ------------ ------------ -----------
  Run          **2.688 BPM**       89.19%       97.59%       -2.30
  Sit              4.548 BPM       66.34%       98.84%       -4.50
  Walk         **6.497 BPM**   **53.17%**   **80.92%**   **-6.42**

Walking is the clear weakness.

Its R² is -0.884, indicating that the current model does not explain
held-out walking HR variation reliably enough.

## Subject-Level Result

Held-out subject MAE:

  Subject           MAE
  --------- -----------
  S18         3.397 BPM
  S19         6.216 BPM
  S20         6.149 BPM
  S21         3.803 BPM
  S22         3.272 BPM

The degradation is not uniform across people, indicating subject/domain
variability in addition to activity-specific difficulty.

## Worst Records

The largest errors occur in walking:

-   S19 walking: MAE 9.921 BPM
-   S20 walking: MAE 10.725 BPM

S20 walking has approximately -10.56 BPM bias and a P90 absolute error
of approximately 19.48 BPM.

These records motivate a dedicated walking robustness experiment rather
than further generic benchmark tuning.

## Interpretation

The V4.3 result is strong enough to establish a reproducible reference
pipeline, but it should **not** be described as production-ready
MAX30102 HR accuracy.

The benchmark is based on MAX30101-domain data. The planned SafeBand
hardware uses MAX30102.

The current evidence supports the following statement:

> V4.3 establishes a subject-independent MAX30101-domain HR estimation
> reference with 4.576 BPM held-out MAE, while identifying walking as
> the dominant remaining failure regime.

## Next Experiment

The next experiment will use the public **Wrist PPG During Exercise**
dataset from PhysioNet.

It contains wrist PPG, accelerometer, gyroscope, and chest ECG reference
data during walking, running, and cycling. It contains 8 participants
and uses 256 Hz sampling.

The walking subset is especially relevant because it directly stresses
wrist PPG under exercise motion.

The dataset will be used first for external evaluation, followed by
controlled walking adaptation only if the external evaluation confirms
the failure mode.

## Scientific Guardrail

The external walking test must remain subject-independent and untouched
during model selection.

No claim of improvement should be made until the adapted model is
evaluated on an untouched walking test set.

## Final V4.3 Conclusion

**V4.3 benchmark: COMPLETE.**

**Walking robustness investigation: NEXT.**

**MAX30102 hardware validation: AFTER walking robustness work.**
