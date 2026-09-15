# SafeBand PPG-DaLiA v1

## Purpose

This package starts the SafeBand MAX30102/PPG work using PPG-DaLiA.

The first objective is **heart-rate estimation from wrist PPG/BVP with wrist accelerometer context**, using ECG-derived heart rate as ground truth.

The official PPG-DaLiA documentation specifies:
- wrist BVP: 64 Hz
- wrist 3-axis ACC: 32 Hz
- ECG-derived HR ground truth
- 8-second windows with 2-second shift
- synchronized and labelled data in `SX.pkl`

## Important methodological rules

1. Audit the actual pickle schema before preparation.
2. Keep subjects completely separated between train/validation/test.
3. Do not use E4-derived `HR.csv` or `IBI.csv` as ground truth.
4. Use the dataset's ECG-derived `label` as HR target.
5. Do not silently resample or normalize sensor units before documenting the transformation.
6. S6 has only about 1.5 hours of valid data and must not be treated as a normal full recording.
7. The first benchmark should establish a robust baseline before introducing deep learning.

## Expected extracted path

```text
datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy/
├── S1/
├── S2/
...
└── S15/
```

## Step 1 — forensic audit

From the repository root:

```powershell
python tools/inspect_ppgdalia.py `
  --root datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy `
  --subjects S1 `
  --json datasets/processed/ppgdalia_audit_S1.json
```

If S1 passes, run all subjects:

```powershell
python tools/inspect_ppgdalia.py `
  --root datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy `
  --json datasets/processed/ppgdalia_audit.json
```

Do not start model training until the audit is reviewed.

## Step 2 — validated window preparation

After the audit passes:

```powershell
python tools/prepare_ppgdalia.py `
  --root datasets/raw/PPG-DaLiA/ppg_dalia/data/PPG_FieldStudy `
  --out datasets/processed/ppgdalia_windows.npz `
  --include-activity
```

This creates:
- BVP windows: `(N, 512)` — 8 s × 64 Hz
- ACC windows: `(N, 256, 3)` — 8 s × 32 Hz
- HR targets: `(N,)`
- subject IDs
- per-subject window indices
- representative activity IDs
- sampling/window metadata

## What comes next

Do **not** immediately choose a neural network.

First benchmark:

1. signal-processing HR estimator from BVP alone
2. BVP + accelerometer motion-quality features
3. classical ML regression
4. subject-independent evaluation
5. error analysis by activity and motion intensity
6. only then consider a compact neural model suitable for SafeBand

The final SafeBand MAX30102 branch should eventually be evaluated on real MAX30102 recordings because PPG-DaLiA uses an Empatica E4 BVP channel rather than the exact MAX30102 hardware.


## Schema correction from S1 audit

The first S1 audit confirmed that the released pickle uses the keys
`Temp` and `Resp` in the chest dictionary rather than the uppercase
`TEMP`/`RESP` spelling used in the prose documentation. The auditor now
accepts those release spellings.

The S1 activity array has 36,848 samples for 4,603 HR windows. This is
exactly 8 activity samples per successive 2-second HR-window start
(4 Hz × 2 s). The preparation script therefore uses the midpoint sample
of each 8-s HR window as the representative activity label rather than
assuming 32 activity samples per HR window.
