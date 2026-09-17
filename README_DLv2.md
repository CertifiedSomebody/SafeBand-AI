# SafeBand Activity DL V2

Next experiment after DL V1. It keeps the same 5-fold subject-grouped protocol and training-only normalization, but tests stronger temporal models:
- `tcn`: residual dilated temporal CNN
- `cnn_gru`: CNN + GRU

Run from repo root:
`python tools/train_activity_dl_v2.py --model tcn`
then optionally:
`python tools/train_activity_dl_v2.py --model cnn_gru`

**Critical V1 bug fixed:** tensors are explicitly canonicalized to `(N,3,40)`. Standardization uses `X.mean(axis=(0,2))` and `X.std(axis=(0,2))`; it never calls an invalid 2-axis transpose on a 3-D batch.
