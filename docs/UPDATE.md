# SafeBand Phase-1 Integration V1.2

This update fixes the integration issues found after applying V1.

## Changes

1. `tools/train_activity_models.py` now defaults to `models/activity_bits2_v2/`, matching the runtime/preflight contract.
2. The activity trainer default report is also written to `models/activity_bits2_v2/report.json`.
3. `tools/preflight_phase1.py` now recognizes both the new activity/fall artifact paths and legacy root-level paths.
4. PPG preflight recognizes both `models/ppg_v4_3/` and the older `models/ppg_v4/` artifact location.
5. Missing optional/untrained model artifacts are reported as `[--]` but do not falsely invalidate the import preflight.
6. `tools/train_airwise_bme680.py` now supplies a fixed seven-class label set to macro metrics/confusion matrices, eliminating the sklearn absent-class warnings without changing the model-selection or held-out evaluation protocol.
7. All three updated tools retain repository-root import handling and are intended to run from the repo root.

## Apply

Copy these three files over the existing files in the repository:

```text
tools/train_activity_models.py
tools/preflight_phase1.py
tools/train_airwise_bme680.py
```

Then run:

```powershell
python tools\train_activity_models.py --input datasets\processed\bits2\activity_windows.csv
python tools\preflight_phase1.py
python -m compileall ai tools
```

Do not delete previous model/report directories. They remain useful as rollback/reference artifacts.
