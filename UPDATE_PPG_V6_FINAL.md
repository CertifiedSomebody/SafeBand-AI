# SafeBand PPG V6 FINAL update

This package finalizes the V6.1 PTT/MAX30101 reference pipeline without changing the frozen V4.3 artifacts.

Added:
- bootstrap 95% confidence intervals for MAE/RMSE/±5 BPM tolerance
- Bland–Altman agreement statistics
- artifact-contract validator
- one-command reproducible PTT build/train/validation runner
- explicit final MAX30102 promotion gate

Important: the current PTT result is MAX30101-family reference evidence. MAX30102 validation still requires raw MAX30102 data with a defensible synchronized reference target.
