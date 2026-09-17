# SafeBand AI — PPG Walking V5.3 FINAL

## Purpose
Final walking-HR experiment before moving toward actual MAX30102 hardware validation. The goal is robust subject-independent HR estimation, not another dataset-specific score.

## Protocol
- Dataset: Wrist PPG During Exercise
- Walking only
- 8 s window
- 2 s shift
- 256 Hz
- PPG bandpass: 0.5–5 Hz
- HR search range: 40–220 BPM
- ECG waveform is NEVER an input; ECG R-peaks define the HR target.
- S9 is completely untouched during development.
- Five development subjects are evaluated with 5-fold GroupKFold.
- Model selection uses mean development out-of-fold MAE.
- No affine calibration.
- Temporal stabilization: alpha=0.35; maximum allowed frame-to-frame change=25 BPM.

## Explicit configurations
1. PPG signal candidates: spectral, peak, autocorrelation and agreement.
2. PPG + motion quality: adds accelerometer/gyro magnitude statistics.
3. Same feature family with model comparison.

## Why
V5.2 showed severe cross-subject failure and calibration pushing against its slope bound. V5.3 therefore makes the physiological signal-derived HR candidates primary and uses ML only as a bounded estimator, followed by conservative temporal stabilization.

## SafeBand interpretation
Every 2 seconds the band can estimate HR from the latest 8-second PPG window. Temporal stabilization is intended to prevent implausible jumps, not to hide poor signal quality.

## Acceptance
Do not declare improvement unless the untouched S9 test improves against the frozen V4.3/V5 baselines on MAE and tolerance metrics without unacceptable bias. After this experiment, prioritize actual MAX30102 recordings and hardware-domain validation.
