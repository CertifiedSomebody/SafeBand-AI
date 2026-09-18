# SafeBand AI — Nonspeech7k Dataset Audit & Preparation V1

## Purpose

This document records the first controlled preparation stage for the SafeBand AI INMP441 audio branch.

The objective is **not** to train a model yet. It is to establish a reproducible, leakage-aware dataset contract before feature extraction and model benchmarking.

## Dataset location

Expected repository root:

```text
datasets/raw/Nonspeech7k/
```

The supplied extraction has the WAVs under paths such as:

```text
datasets/raw/Nonspeech7k/train/train/*.wav
datasets/raw/Nonspeech7k/test/test/*.wav
```

The scripts recursively discover WAV files and therefore do not hard-code the duplicated `train/train` or `test/test` depth.

## Metadata schema observed

### Train

The supplied train metadata contains 6,289 rows and these columns:

```text
Filename
File ID
Duration in ms
Class ID
Classname
augmentation  id
Augmentation  type
source
```

### Test

The supplied test metadata contains 725 rows and these columns:

```text
Filename
File_ID
Durationin ms
Class_id
Classname
Augment Id
Augmentation type
source
```

The preparation code accepts these train/test naming differences explicitly.

## Classes

The seven original classes are normalized to:

```text
breath
cough
crying
laugh
screaming
sneeze
yawn
```

The supplied test metadata contains the spelling `yawm`; this is normalized to `yawn` only in the processed manifest. Raw metadata remains untouched.

## Source distribution

The supplied metadata reports:

```text
Train
Freesound : 2935
Aigei     : 1739
YouTube   : 1615

Test
Freesound : 725
```

The official test set is kept separate and is not used for model selection.

## Critical leakage finding

An audit of `(source, File ID)` found **4 overlapping source/File-ID groups** between the supplied train and test metadata. The raw files are not modified and the official test set is not changed.

For the clean training manifest, every train row belonging to a source/File-ID group appearing in the official test metadata is **quarantined**.

This is intentionally conservative. It prevents recordings associated with the same source identity from entering training when that identity is represented in the official test set.

The quarantined rows are written to:

```text
datasets/processed/nonspeech7k/train_test_overlap_quarantine.csv
```

This makes the exclusion auditable rather than silently deleting data.

## Preparation outputs

The preparation script creates:

```text
datasets/processed/nonspeech7k/
├── train_clean_manifest.csv
├── test_official_manifest.csv
└── train_test_overlap_quarantine.csv
```

The manifests contain normalized labels and repository-relative audio paths.

The raw dataset remains immutable.

## Augmentation policy

Only metadata rows with augmentation ID `0` are retained for the first benchmark preparation.

No synthetic augmentation is generated at this stage.

This keeps the initial benchmark tied to original recordings and avoids introducing a training shortcut before the baseline is understood.

## Leakage-control policy

The first audio benchmark will preserve these boundaries:

1. The supplied official test set remains untouched.
2. Source/File-ID groups appearing in the official test set are excluded from clean training.
3. No audio-derived feature is used to make split decisions after the split is established.
4. Future cross-validation must group by source identity where the metadata supports it.
5. Raw WAV files and original metadata are never overwritten.

## Commands

From the repository root:

### Audit

```powershell
python tools\audit_nonspeech7k.py
```

### Prepare clean manifests

```powershell
python tools\prepare_nonspeech7k.py
```

Optional explicit paths are supported:

```powershell
python tools\audit_nonspeech7k.py --root datasets/raw/Nonspeech7k --train-meta "datasets/raw/Nonspeech7k/metadata of train set .csv" --test-meta "datasets/raw/Nonspeech7k/metadata of test set.csv"
```

## What this stage does NOT claim

This stage does not establish:

- audio classification accuracy;
- INMP441 hardware performance;
- emergency-detection performance;
- scream detection as a medical or safety guarantee;
- robustness to microphone placement, distance, wind, clothing, or real-world noise.

Those require later experiments and, ultimately, real SafeBand hardware recordings.

## Next stage

After this audit/preparation passes, the next controlled stage is:

```text
Clean manifests
      ↓
Audio loading / resampling contract
      ↓
Log-mel / MFCC feature benchmark
      ↓
Subject/source-independent model evaluation
      ↓
Official test evaluation
      ↓
Audio evidence vector
      ↓
Sensor fusion
```

The audio model should initially preserve all seven classes rather than immediately collapsing them into `EMERGENCY / NOT_EMERGENCY`. The fusion layer can later determine how individual audio events contribute to overall risk.

## Engineering decision

**Freeze this dataset audit/preparation stage before training.** If the audit passes, do not alter the raw dataset or silently change the leakage policy while comparing models.
