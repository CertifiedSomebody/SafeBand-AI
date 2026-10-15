
# SafeBand AI — AICOS-HAR Autoencoder V1

This package adds one controlled experiment after the AICOS-HAR ACC-only classical/TinyCNN benchmark.

## Experiments

1. `audit_aicos_har_raw_windows_v1.py`
   - Checks NPZ windows for NaN/Inf.
   - Counts zero-variance ACC axes.
   - Quantifies windows where `np.corrcoef`-style correlation features are mathematically undefined.
   - Does not modify the dataset.

2. `train_aicos_har_dae_v1.py`
   - Supervised denoising convolutional autoencoder.
   - Input: ACC x/y/z + deterministic acceleration magnitude.
   - Encoder learns a 128-D latent representation.
   - Classification head is optimized jointly with reconstruction.
   - Mild train-only augmentation.
   - Inverse-square-root class weighting.
   - Training-only normalization.
   - Subject-grouped validation using StratifiedGroupKFold.
   - Official `test` partition is never used for training, validation selection, or normalization.
   - Early stopping monitors validation Macro-F1.
   - Saves model, normalization parameters, labels, and JSON report.

## Important

No accuracy is guaranteed. The goal is to give the autoencoder a materially stronger and cleaner chance than the previous plain TinyCNN without changing the official test protocol.

## Commands

Audit first:

```powershell
python tools\audit_aicos_har_raw_windows_v1.py --npz datasets\processed\aicos_har\aicos_har_acc_windows_v4.csv.npz --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv --out reports\aicos_har_raw_window_audit_v1.json
```

Then train:

```powershell
python tools\train_aicos_har_dae_v1.py --npz datasets\processed\aicos_har\aicos_har_acc_windows_v4.csv.npz --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv --out models\aicos_har\dae_v1
```

Do not tune based on the official test result. The test partition remains the final untouched comparison against ExtraTrees, RF, HGB, SVM and TinyCNN.
