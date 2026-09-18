# PAMAP2 Stationary V5 — Patched

This version addresses the V4 runtime failures rather than changing the dataset.

## Main V4 failure fixed
`GridSearchCV` previously received a generator from `StratifiedGroupKFold.split()`, which caused Windows/joblib to raise `TypeError: cannot pickle 'generator' object`.

V5 converts the splits to a list and uses `n_jobs=1` for the SVM search.

## Other corrections
- No sklearn random/un-grouped ANN early stopping.
- CNN receives `(N,6,T)`.
- BiLSTM/BiGRU receive `(N,T,6)` explicitly.
- PyTorch DataLoader uses `num_workers=0`.
- Neural epoch-budget selection uses grouped inner folds.
- Outer test subjects are not used for model selection.
- Normalization is fitted only on training partitions.
- Dataset and feature dimensions are checked before training.

## Caveat
The package is syntax-checked and feature-preflighted. A complete real-data training run must still be performed in the project's Windows venv.
