# AICOS-HAR Final Runbook

## Completed
- Dataset audit
- Acquisition inventory
- Lift/Elevator alias reconciliation
- Sampling audit
- 50 Hz ACC window generation
- Window/manifest validation
- Zero-window inventory

## Current gate
Run the V7 zero-window diagnostic:

```powershell
python tools\audit_aicos_har_zero_window_v7.py --data-dir datasets\raw\AICOS-HAR\AICOS-HAR --metadata datasets\raw\AICOS-HAR\AICOS-HAR\metadata.csv --manifest datasets\processed\aicos_har\aicos_acc_50hz_2s_manifest.csv --out reports\aicos_har_zero_window_v7
```

If every zero-window acquisition is explained by insufficient duration, accept the exclusion and document coverage.

If any is reported as `builder_edge_case`, stop and inspect the builder before training.

## Then
Proceed to the existing V4 classical trainer and TinyCNN trainer from the previous pipeline package. Do not touch the official test fold during model selection.
