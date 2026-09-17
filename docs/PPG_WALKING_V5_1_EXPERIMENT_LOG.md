# PPG Walking V5.1 Experiment Log

## V4.3 Baseline Context

V4.3 identified walking as the dominant weakness:
- walking MAE: 6.497 BPM
- ±5 BPM: 53.17%
- ±10 BPM: 80.92%
- bias: -6.424 BPM
- R²: -0.884

## W5 Preparation

The external PhysioNet walking dataset was prepared with:
- 8 s windows
- 2 s shift
- 256 Hz
- ECG R-peak-derived HR target

Current preparation result:
- 6 subjects
- 6 walking records
- 1,097 valid windows
- 70 extracted features
- 100% of generated windows valid

## V5.1 Plan

Run controlled representation/model comparisons:
- PPG-only
- PPG + gyro
- PPG + low-noise ACC + gyro

Models:
- Ridge
- RF
- Extra Trees
- HGB

Select by validation MAE.

Then evaluate once on the untouched test subject.

## Result

Pending execution.

Do not edit this section with a claimed improvement until the held-out result exists.
