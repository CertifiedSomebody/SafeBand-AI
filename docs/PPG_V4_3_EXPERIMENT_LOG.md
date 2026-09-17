# PPG V4.3 FINAL — Experiment Log

## Version lineage

**V3:** PPG-DaLiA / Empatica E4 BVP reference.  
**V4:** PTT / MAX30101-domain reference with six PPG channels + IMU.  
**V4.1:** corrected subject ordering/preparation workflow.  
**V4.2:** diagnostics and baseline/ablation infrastructure.  
**V4.3:** final controlled experiment for the current reference dataset.

## V4.1/V4.2 evidence retained

V4.2 contained:
- 15,982 windows
- 22 subjects
- 159 features
- 60/20/20 numeric subject split
- HGB selected by validation MAE
- held-out MAE 4.576 BPM
- RMSE 5.955 BPM
- R² 0.591
- ±5 BPM 69.57%
- ±10 BPM 92.47%
- bias −4.407 BPM

Activity analysis identified walking as the principal weakness:
walk MAE 6.497 BPM and bias −6.424 BPM.

The worst records were s19_walk and s20_walk.

## V4.3 research question

Can we determine, without changing the dataset or evaluation protocol,
whether motion-derived features provide measurable value beyond PPG-only
features?

## Controlled comparisons

1. Constant train-mean baseline.
2. PPG spectral/peak HR baseline.
3. Ridge / RF / Extra Trees / HGB using all established features.
4. HGB using PPG-only features.
5. HGB using PPG + motion features.

## Anti-leakage rules

- No random window split.
- No subject overlap.
- Validation selects models.
- Calibration is validation-only.
- Test is not used for feature selection.
- ECG waveform is never a feature.
- ECG peak annotations provide HR target.
- SpO2 start/end values are not converted into continuous window labels.

## Final-stage boundary

V4.3 is the final benchmark-analysis stage. If it does not resolve the
walking/generalization limitation, further benchmark tuning should stop.

The next scientifically meaningful step is actual MAX30102 recording, because
the project's deployment sensor is MAX30102 and PTT uses MAX30101-family hardware.
