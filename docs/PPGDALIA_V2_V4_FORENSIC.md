# PPG-DaLiA V2 ↔ V4 Forensic Verification

## Purpose

Before accepting the temporal HR experiment as an improvement, the numerical
feature representation used by V4 must reproduce the established V2
representation.

The previously observed results are not sufficiently comparable while the
following discrepancy exists:

- Established V2 benchmark: approximately 7.63 BPM MAE
- V4 raw baseline: approximately 8.14 BPM MAE

Feature-name equality alone is not sufficient evidence of numerical equality.

---

## Verification sequence

### 1. Dataset preflight

Run:

```powershell
python tools/ppgdalia_forensic_preflight.py `
  --input datasets/processed/ppgdalia_windows.npz `
  --out models/ppgdalia_forensic_preflight.json