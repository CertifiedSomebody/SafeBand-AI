# SafeBand PPG Walking V5

## Purpose

V4.3 identified walking as the dominant weakness of the held-out PTT/MAX30101-domain HR experiment. The V5 branch therefore targets **walking robustness specifically** using the independent PhysioNet **Wrist PPG During Exercise v1.0.0** dataset.

The dataset contains wrist PPG, chest ECG reference, gyroscope, low-noise accelerometer, wide-range accelerometer, and magnetometer. All signals are sampled at 256 Hz; ECG R peaks were manually identified and supplied as annotations. Walking/running records contain raw PPG and motion signals without additional filtering beyond the Shimmer hardware. citeturn0search0turn0search1

## Why a new feature space

V4.3 uses six PTT optical channels (distal/proximal red/IR/green). The Wrist dataset contains one wrist PPG channel. Replicating the single wrist PPG into six artificial channels would create fabricated information and is prohibited.

V5 therefore uses an honest **common single-PPG + motion representation**:

- one PPG channel
- 3-axis accelerometer
- 3-axis gyroscope
- per-window normalization
- 0.5–5 Hz PPG filtering
- spectral HR features
- peak HR features
- PPG AC/DC
- motion magnitude/statistics/spectrum
- PPG↔motion coupling

The model family and selection protocol remain aligned with V4.3: Ridge, Random Forest, Extra Trees, HistGradientBoosting; validation-MAE selection; validation-only affine calibration; untouched subject-level test.

## Window protocol

- Window: 8 seconds
- Shift: 2 seconds
- Sampling: 256 Hz native
- Target: ECG R-peak-derived mean instantaneous HR within each window
- Input: PPG + motion only
- ECG waveform: never an input feature

The database's ECG R peaks are explicitly provided for gold-standard exercise HR comparison. citeturn0search0

## Experimental stages

### W1 — External characterization

Before using the walking dataset for adaptation, run the prepared data through a controlled baseline/model experiment. This establishes how difficult the external walking domain is.

### W2 — Walking-specific model

Train V5 only on the walking dataset using subject-independent 60/20/20 splitting. The held-out walking subjects are not used during model selection or calibration.

### W3 — SafeBand integration

Only after W2 should the resulting features/model be considered for later comparison with actual MAX30102 recordings.

## Important interpretation

V5 is **not** a MAX30102 validation result. It is an external walking-domain experiment intended to diagnose and improve robustness before hardware validation.

## Source

PhysioNet: Wrist PPG During Exercise v1.0.0. Official description states that the database contains wrist PPG during walking, running and bike riding, simultaneous accelerometer/gyroscope motion measurements, and a reference chest ECG; it contains records from 8 participants and uses 256 Hz sampling. citeturn0search0
