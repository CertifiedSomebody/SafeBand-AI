# SafeBand Activity — Sequence Models V2

This batch tests two sequence architectures on the validated ACC+GYRO tensor:

- Bidirectional LSTM
- Small Transformer Encoder

Input:
`datasets/processed/bits2/activity_accgyro_windows.npz`

Expected:
`X = (N, 6, 40)` with channels ax, ay, az, gx, gy, gz.

## Run LSTM

    python tools/train_activity_lstm_v2.py

## Run Transformer

    python tools/train_activity_transformer_v2.py

Both use the same:
- 5-fold StratifiedGroupKFold by subject
- training-only channel normalization
- inner grouped validation for early stopping
- untouched outer test folds
- fixed seed 42

The LSTM explicitly converts `(N,6,40)` to `(N,40,6)` using `permute(0,2,1)`.
The Transformer uses the same explicit conversion and a learned positional embedding
for the 40 time steps.

## Why both?

LSTM is a natural test for sequential dependencies and temporal order.

The Transformer tests whether attention can model relationships between time steps
more effectively than recurrent recurrence. The implementation is deliberately
small because the window contains only 40 samples; a large Transformer would add
capacity without a strong methodological reason.

Do not select a deployment model from the outer test scores. Compare models only
after both experiments are complete, then select according to the pre-declared
evaluation criterion and resource constraints.
