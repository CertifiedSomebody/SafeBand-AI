# AICOS-HAR Stage 6 — Window Validation

This stage is a hard gate before machine-learning training.

## Checks
- NPZ exists and contains `X`.
- Expected tensor shape is `(N, 3, 100)` for 50 Hz, 2 s windows.
- Manifest row count equals the number of windows.
- No NaN/Inf values.
- Every window has an acquisition, participant, activity, fold, and position.
- Official `test` fold exists.
- No participant occurs in multiple official folds.
- If metadata is supplied, every metadata acquisition has at least one window and no unknown acquisition IDs appear.

The validator also detects the legacy V4 output naming issue where `np.savez_compressed()` can create `file.csv.npz` when a `.csv` path was supplied.

Do not train until this report says `PASS`.
