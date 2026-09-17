# SafeBand Activity DL V1

This package adds a **deployment-conscious deep-learning benchmark** to the existing SafeBand Activity Recognition pipeline.

## Purpose

The V4 classical ML model is retained as the baseline. DL is tested as a separate hypothesis:

> Can a compact temporal neural network learn useful activity representations directly from the 3-axis ACC waveform and improve subject-independent activity recognition?

There is **no guaranteed 90–95% target** in the code. A target is a project requirement, not evidence. The experiment must earn any claimed improvement on held-out subjects.

## Models

- `tiny_cnn`: compact 1D CNN for 3-axis, 40-sample (~2 s) ACC windows.
- `cnn_lstm`: compact CNN+GRU temporal model for optional comparison.

## Evaluation

- BITS2 only
- RESTING / SITTING / WALKING / RUNNING
- FALL remains a separate safety detector
- 5-fold `StratifiedGroupKFold`
- `subject_id` is the grouping variable
- normalization is fitted only on training subjects
- early stopping uses only the current training fold's validation split
- pooled and fold-level accuracy, balanced accuracy, macro-F1, reports and confusion matrices are saved

This is intentionally different from the engineered-feature V4 model: DL receives the raw temporal ACC window instead of the 38 handcrafted features.

## Run

From the SafeBand repo root:

```powershell
python tools\build_activity_raw_windows.py
python tools\train_activity_dl_v1.py
```

Optional CNN+GRU comparison:

```powershell
python tools\train_activity_dl_v1.py --model cnn_lstm
```

If PyTorch is missing:

```powershell
pip install torch numpy scikit-learn
```

## Decision rule

Do not replace V4 merely because DL is newer. Compare:

1. macro-F1
2. balanced accuracy
3. class-wise recall/F1, especially RESTING/SITTING
4. fold stability
5. model size
6. inference latency
7. embedded feasibility

If DL does not produce a meaningful, reproducible improvement, V4 remains the deployment baseline.

## Research history

- V3 confusion-guided specialist: rejected; only +0.08 percentage-point macro-F1 and inconsistent fold behavior.
- V4 strong classical ML: current ACC baseline.
- DL V1: controlled raw-waveform experiment.

The purpose is to improve the model scientifically, not to manufacture a high number through leakage or subject-dependent evaluation.
