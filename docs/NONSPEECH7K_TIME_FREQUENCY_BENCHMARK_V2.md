# SafeBand AI — Nonspeech7k Time-Frequency Benchmark V2

## Objective
V2 tests richer audio representations after the V1 handcrafted-feature baseline. The target is a materially stronger audio-event reference model; **90%+ accuracy is an engineering target, not a guaranteed result**. We do not alter the official test set or leak it into model selection.

## Dataset
Clean Nonspeech7k training manifest:
- 6,283 recordings
- 1,899 unique File IDs
- 7 labels: breath, cough, crying, laugh, screaming, sneeze, yawn
- official 725-recording test remains untouched.

## V1 limitation addressed
The V1 feature baseline used compact global/frame statistics. V2 uses:
- 64-bin log-mel spectrogram statistics
- 20 MFCC coefficients
- first and second temporal derivatives
- percentile/statistical summaries
- RMS and duration

All audio is converted to mono and resampled to 16 kHz for a consistent representation.

## Leakage controls
- File ID is used only as a grouping variable, never as a feature.
- `audio_path` is metadata, never a feature.
- `StratifiedGroupKFold` is used.
- Official test is not touched.
- The manifest's actual `audio_path` column is authoritative.
- Relative paths are resolved from the repository root.
- No raw files are modified.
- The script refuses to guess paths from `file_id`.

## Preflight
Run:
```powershell
python tools\audit_nonspeech7k_tf_v2.py
```

Then:
```powershell
python tools\prepare_nonspeech7k_tf_v2.py
```

Then:
```powershell
python tools\benchmark_nonspeech7k_tf_v2.py
```

## Models
- balanced Logistic Regression
- balanced Extra Trees
- balanced Random Forest
- HistGradientBoosting

This is a feature benchmark first. A CNN is intentionally deferred until the representation baseline is verified.

## Acceptance criteria
The requested engineering target is >90% accuracy. However:
- no result is inflated or selected solely because it crosses 90%;
- balanced accuracy and macro-F1 must also be reported;
- if V2 remains below 90%, that is recorded honestly and the next stage is a learned log-mel CNN rather than repeated arbitrary classical tuning.

## Reproducibility
All tools are root-safe and validate the actual manifest schema before processing. The feature extractor reports the exact number of processed WAVs and fails loudly on unresolved files.

## Status
V2 package prepared; results pending execution.
