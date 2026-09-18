# PAMAP2 Stationary V5 PATCHED3

## Execution errors addressed

### Feature contract
The implemented handcrafted feature extractor creates:
- 8 streams × 11 statistics = 88
- 6 aggregate features = 6
- 15 pairwise correlations = 15

Total = **109 features**.

### Standardization API
`standardize_train(X_train, *others)` returns:

```text
transformed_X_train, transformed_other_1, transformed_other_2, ...
```

The previous implementation incorrectly attempted to unpack five values from a three-value call. V5 PATCHED3 uses the correct contract everywhere.

### Windows/joblib
Inner grouped splits are materialized before GridSearchCV. SVM search uses `n_jobs=1`.

### ANN
ConvergenceWarning is expected when the optimizer reaches `max_iter`; it is not treated as an error. Warnings are suppressed to keep the benchmark readable.

### Neural models
The recurrent models explicitly transpose `(N,6,T)` into `(N,T,6)`. DataLoader multiprocessing is disabled for Windows compatibility.

## Scientific protocol
- 5 outer folds
- StratifiedGroupKFold
- group = PeopleId
- grouped inner selection
- outer test untouched
- training-only normalization
- fixed seed 42

## Important interpretation
The model suite is a family comparison, not a license for unlimited architecture tuning. The main question is whether additional model families provide useful information for SafeBand stationary-state recognition.
