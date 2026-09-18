# SafeBand AI — PAMAP2 Stationary Model Suite V5 PATCHED3

This package patches the execution errors observed in the previous V5 run.

## Models
- RBF-SVM
- ANN/MLP
- Tiny 1D CNN
- BiLSTM
- BiGRU

## Critical fixes
1. Feature count is correctly defined as 109:
   88 + 6 + 15 = 109.
2. `standardize_train()` returns exactly `train + one output per extra array`.
   All callers now unpack that API correctly.
3. Inner grouped split generators are materialized with `list(...)`.
4. SVM GridSearchCV uses `n_jobs=1` for Windows-safe execution.
5. ANN ConvergenceWarnings are suppressed because they are warnings, not failures.
6. PyTorch DataLoader uses `num_workers=0`.
7. CNN input remains `(N,6,T)`.
8. LSTM/GRU input is explicitly converted to `(N,T,6)`.
9. NPZ schema, labels, finite values, shape and subject count are validated.
10. Outer test subjects are never used for model selection.
11. Normalization is fitted only on training data.
12. The neural candidate search is bounded to 10/20/30 epochs to avoid an uncontrolled runtime explosion.

## Run

From the repository root:

```powershell
python tools\train_pamap2_stationary_v5.py
```

Optional:

```powershell
python tools\train_pamap2_stationary_v5.py --data datasets\processed\pamap2\stationary_accgyro_windows.npz --out models\pamap2_stationary_v5
```

Output:

```text
models\pamap2_stationary_v5\model_suite_v5_report.json
```

This package is a corrected benchmark implementation; real-data execution must still be performed in the project venv.
