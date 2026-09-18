# SafeBand AI — Nonspeech7k Best-Fit Audio Model V1

## Purpose

This is the final model-selection step for the Nonspeech7k reference dataset.

The earlier V2 benchmark established that time-frequency handcrafted features substantially outperform the earlier simple feature representation, with HistGradientBoosting reaching the strongest completed V2 result.

Rather than assuming that model is optimal, this script performs a reproducible **best-fit selection** among a small set of strong classical candidates on the same V2 representation.

## Candidates

- HistGradientBoosting
- Random Forest
- Extra Trees
- RBF SVM
- Logistic Regression

## Evaluation

- 5-fold StratifiedGroupKFold
- grouping by File ID
- all 6,283 clean training recordings
- 1,899 File-ID groups
- official 725-recording test set untouched

## Selection rule

The model with the highest **mean Macro-F1** is selected.
Ties are resolved by mean balanced accuracy, then mean accuracy.

This avoids choosing a model merely because the majority class dominates raw accuracy.

## Final artifact

The selected candidate is then refit on all 6,283 approved training recordings and saved as:

`models/nonspeech7k_audio_bestfit_v1/nonspeech7k_audio_bestfit_model.joblib`

Metadata is saved alongside it.

## Command

From repository root:

```powershell
python tools\train_nonspeech7k_bestfit.py
```

## Important

This script produces a deployment artifact after internal CV selection. It does not create a new unbiased test score.

The official Nonspeech7k test set remains untouched.
