# PPG-DaLiA → SafeBand MAX30102 Stage 1

## Why PPG-DaLiA is being used

PPG-DaLiA provides wrist BVP + wrist 3-axis accelerometer data and an ECG-derived HR target.
It is useful for establishing whether our physiological branch can estimate heart rate and how
motion affects optical pulse estimation.

**It is NOT a MAX30102 dataset.**

The released PPG-DaLiA wrist signal comes from an Empatica E4. Its BVP samples are therefore
not assumed to have the same optical path, LED/photodiode characteristics, ADC scale,
sampling behavior, contact mechanics, or signal morphology as a MAX30102 module.

## Conflict policy

We will NOT train a model on E4 BVP and call it a MAX30102 model.

Instead:

1. PPG-DaLiA = algorithmic / research benchmark.
2. Real MAX30102 recordings = hardware-domain validation and later fine-tuning.
3. The interface between them is a generic normalized PPG representation, not raw-value equivalence.
4. Sensor-specific calibration and preprocessing will be isolated in adapters.
5. The final SafeBand model must be tested on recordings produced by the actual MAX30102 hardware.

## Stage 1 benchmark

`tools/benchmark_ppgdalia_hr.py` compares:
- BVP-only FFT HR
- BVP-only peak/IBI HR
- classical ML using compact BVP + motion-quality features

Evaluation is subject-independent. Test subjects are not used for model selection.

## Run

First do a smoke test:

```powershell
python tools\benchmark_ppgdalia_hr.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --out models\ppgdalia_hr_smoke.json `
  --max-windows 3000
```

If that completes, run the full benchmark:

```powershell
python tools\benchmark_ppgdalia_hr.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --out models\ppgdalia_hr_benchmark.json
```

Do not treat the smoke-test metrics as research results.

## Future MAX30102 recording contract

A SafeBand logger should produce at minimum:

`timestamp_ms, ir`

Preferably:

`timestamp_ms, ir, red, acc_x, acc_y, acc_z`

Run:

```powershell
python tools\check_max30102_recording.py path\to\max30102.csv
```

before using real hardware data.

## What this stage does NOT do

- It does not claim E4 BVP == MAX30102 PPG.
- It does not perform cross-sensor transfer validation.
- It does not produce SpO2 estimates.
- It does not train a final embedded MAX30102 model.
- It does not replace real MAX30102 hardware validation.
