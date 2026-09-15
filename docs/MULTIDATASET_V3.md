# V3 Experiment Design

## Baseline

The baseline is the V2 equal-dataset-weighted training setup. Each dataset
contributes exactly half of the total training sample weight.

## Signal transformations

### Orientation

A random axis-angle rotation is applied with maximum angle 12 degrees. This
models small wrist/sensor orientation differences without inventing a new
activity.

### Magnitude + noise

The complete 3-axis signal is multiplied by a random factor in [0.90, 1.10].
Gaussian noise is then added with standard deviation equal to 0.5–1.5% of
the window's signal standard deviation.

### Temporal jitter

A nearby 60-sample window is selected from the same recording, with a start
shift of at most 6 samples (300 ms at 20 Hz). This addresses dependence on
exact window boundaries.

### Combined

Three independent synthetic copies are created from each training window.
This deliberately increases training-set size, but dataset-level sample
weights are recomputed from the resulting counts so BITS-2 and SisFall still
have equal total training weight.

## Why no arbitrary feature noise

The V6 feature set contains physically meaningful morphology measurements:
peak prominence, pre/post impact statistics, energy ratios, impulse area,
tilt change and spectral entropy. Modifying those numbers independently
would break their physical relationships. Therefore V3 modifies the signal
first and then recomputes all 84 features.

## Why validation/test remain untouched

Any augmentation of validation or test data would change the evaluation
distribution and make comparisons with V2 less defensible. Only training
subjects are augmented.

## Model selection

For every recipe, four classical models are compared. The primary validation
criterion is:

`0.5 * BITS2_F1 + 0.5 * SisFall_F1`

Balanced accuracy is used as a tie-break. The held-out test set is not used
during selection.

## Safety interpretation

A higher fall recall with a large false-positive rate is not automatically
better for SafeBand. The eventual system needs temporal confirmation to turn
window-level evidence into a real alarm event. V3 therefore measures whether
augmentation improves the underlying probability signal before that temporal
stage.
