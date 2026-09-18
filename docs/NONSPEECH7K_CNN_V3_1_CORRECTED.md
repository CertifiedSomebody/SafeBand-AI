# SafeBand AI — Nonspeech7k CNN V3.1 Corrected Benchmark

## Objective

V3.1 is a corrective benchmark after V3 produced:

- mean accuracy: 51.69%
- mean balanced accuracy: 48.69%
- mean macro-F1: 43.24%
- best epoch: 1 in all five folds

The objective is to determine whether the log-mel CNN can learn the task under a sound grouped validation/training procedure.

The >90% project target is retained as an engineering target, not a guaranteed result.

## Frozen dataset

Training reference:

- 6,283 clean recordings
- 1,899 unique File IDs
- 7 labels:
  - breath
  - cough
  - crying
  - laugh
  - screaming
  - sneeze
  - yawn

The official 725-recording test set is untouched.

## Input representation

Each WAV is:

1. decoded as mono;
2. resampled to 16 kHz;
3. converted to a 64-bin mel spectrogram;
4. converted to dB;
5. clipped to [-80, 0] dB;
6. padded or center-cropped to 128 frames.

Final tensor:

`(1, 64, 128)`

## Evaluation protocol

Outer evaluation:

`5-fold StratifiedGroupKFold`

Grouping variable:

`file_id`

This prevents recordings belonging to the same File ID group from appearing in both the outer training and outer test portions.

Inner validation:

- grouped `GroupShuffleSplit`
- 50 deterministic candidate splits
- candidates must contain every class in both train and validation
- candidate with validation class proportions closest to the outer-training distribution is selected

This fixes the arbitrary inner split problem without leaking groups.

## Training

- SmallLogMelCNN
- AdamW
- learning rate 1e-3
- weight decay 1e-4
- batch size 64
- maximum 35 epochs
- minimum 8 epochs before early stopping
- patience 7
- checkpoint criterion: validation macro-F1
- class-balanced cross-entropy weights calculated from inner training data only
- normalization mean/std calculated from inner training data only
- seed 42
- DataLoader workers = 0 for reproducibility on Windows

## Diagnostics

Every epoch reports:

- training loss
- validation macro-F1

Every fold reports:

- inner train/validation class counts
- validation class-proportion error
- best epoch
- outer-test accuracy
- outer-test balanced accuracy
- outer-test macro-F1
- confusion matrix
- complete training history

The official test set is never used.

## Commands

Run from repository root:

```powershell
python tools\audit_nonspeech7k_cnn_v3_1.py --check-all
python tools\build_nonspeech7k_logmel_v3_1.py
python tools\train_nonspeech7k_cnn_v3_1.py
```

For a quick preflight instead of decoding all files:

```powershell
python tools\audit_nonspeech7k_cnn_v3_1.py
```

## Important engineering notes

The cache builder intentionally uses an object array while constructing File IDs before explicitly converting to strings. This prevents accidental one-character Unicode truncation.

No raw dataset files are modified.

No official test data is copied into the training cache.

No result from V3.1 should be described as >90% unless the actual measured outer evaluation supports it.
