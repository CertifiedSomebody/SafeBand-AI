# SafeBand BNO055 Synthetic V1

This package creates a BNO055-oriented synthetic telemetry dataset for
SafeBand software development.

## Quick start

From the repository root:

```powershell
python tools\generate_bno055_synthetic.py
python tools\audit_bno055_synthetic.py
```

Default output:

`datasets\synthetic\bno055\bno055_synthetic_v1.csv`

See:

`docs\BNO055_SYNTHETIC_DATASET_V1.md`

The generated data are synthetic test data, not real BNO055 measurements.
