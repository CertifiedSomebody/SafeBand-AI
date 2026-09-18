# SafeBand AI — Nonspeech7k Log-Mel CNN Benchmark V3

## Purpose
Move from summary statistics to direct time-frequency learning using a compact CNN over log-mel spectrograms.

The project engineering target is **>90% accuracy**. It is not a guaranteed result and cannot override the measured leakage-controlled metrics.

## Frozen data
- 6,283 clean training recordings
- 1,899 unique File IDs used as groups
- 7 classes: breath, cough, crying, laugh, screaming, sneeze, yawn
- official 725-recording test set remains untouched

## Representation
- mono
- 16 kHz
- 64 mel bins
- FFT 512
- hop 160
- fixed 128-frame crop/pad
- dB clipped to [-80, 0]
- tensor: `(N, 1, 64, 128)`

## Leakage controls
- `file_id` is explicitly read as string and used only as a group identifier.
- Manifest `audio_path` is authoritative.
- Official test is not read.
- Outer evaluation uses `StratifiedGroupKFold`.
- Inner validation is grouped and comes only from the outer training fold.
- normalization statistics are fitted only on the inner training portion.
- class weights are fitted only on the inner training portion.
- best epoch is selected using validation macro-F1.
- outer test remains unseen during model selection.

## Model
Small 2-D CNN with four convolutional stages, batch normalization, ReLU, pooling, dropout, global average pooling and a 7-class head.

## Commands
```powershell
python tools\audit_nonspeech7k_cnn_v3.py --check-all
python tools\build_nonspeech7k_logmel_v3.py
python tools\train_nonspeech7k_cnn_v3.py
```

## Metrics
Accuracy, balanced accuracy, macro-F1 and confusion matrices are reported. The final decision must consider all of them.

## Runtime discipline
The cache is created once. Training reads the cache and does not repeatedly decode WAV files. Windows DataLoader uses `num_workers=0` for reliability.

## If the 90% target is not reached
Do not perform uncontrolled hyperparameter sweeps. Use the confusion matrix and fold variance to choose a specific next experiment, such as augmentation or temporal crop strategy.

## Status
Prepared; execution pending.
