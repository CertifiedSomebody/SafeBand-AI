# SafeBand AI — Nonspeech7k Audio Feature Benchmark V1

## Purpose
This stage converts the leakage-safe Nonspeech7k training manifest into reproducible audio-level features and benchmarks lightweight classical classifiers. It is a **reference benchmark for the INMP441 audio evidence branch**, not final hardware validation.

## Dataset contract
Input: `datasets/processed/nonspeech7k/train_clean_manifest.csv`

The official Nonspeech7k test manifest remains untouched and is **not used in cross-validation**.

Grouping key: `file_id`. Recordings sharing a File ID stay in the same fold to reduce source leakage.

## Feature set
The baseline uses waveform-level and short-time features:
- sample rate and duration
- RMS, peak amplitude and crest factor
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- frame-level RMS statistics
- frame-level zero-crossing statistics

No class name, source name, filename pattern, or metadata-derived label information is used as a feature.

## Evaluation
Five-fold GroupKFold by `file_id` is used. The official test set is reserved for a later final evaluation.

Models:
- Logistic Regression
- Extra Trees
- Random Forest
- HistGradientBoosting

Metrics:
- accuracy
- balanced accuracy
- macro-F1
- confusion matrix

No model is declared the final deployment model from this stage alone.

## Commands
From repository root:

```powershell
python toolsudit_nonspeech7k_features_v1.py
python tools\prepare_nonspeech7k_features_v1.py
python toolsenchmark_nonspeech7k_audio_v1.py
```

## Expected outputs
```text
datasets/processed/nonspeech7k/train_features_v1.csv
reports/nonspeech7k_audio_v1/benchmark_report.json
```

## Research rules
1. Raw Nonspeech7k files remain immutable.
2. The official test set remains untouched.
3. File-ID grouping is mandatory for benchmark splits.
4. This benchmark does not claim INMP441 hardware performance.
5. The seven original classes remain intact initially:
   `breath, cough, crying, laugh, screaming, sneeze, yawn`.
6. Audio output will eventually become an evidence vector for sensor fusion rather than a direct emergency decision.
7. Further dataset additions should be evaluated separately rather than silently merged into this benchmark.
