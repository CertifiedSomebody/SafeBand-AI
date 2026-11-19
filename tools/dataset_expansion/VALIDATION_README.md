# Milestone 2.10.1 Validation

Validates a generated augmented session against its source.

Checks:
- exact 36-column schema and order
- row count preservation
- activity label preservation
- duplicate rows
- finite numeric values
- `-999` sentinel preservation
- timing stall count/fraction
- contiguous sample index
- basic physical guardrails
- basic motion-magnitude sanity

A JSON report is written to the requested output path.

Example:

```powershell
python tools\dataset_expansion\validate_augmentation.py `
  --source "datasets\real sensor data\walking.csv" `
  --augmented "datasets\generated\walking_aug_001.csv" `
  --output "reports\dataset_expansion\walking_aug_001_validation.json"
```
