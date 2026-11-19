# SafeBand Milestone 2.10.1

Generic session-level diversity augmentation foundation.

## Purpose

This stage creates a reusable augmentation layer for existing SafeBand sessions.
It does NOT generate hazardous events, does NOT alter labels, and does NOT claim
synthetic data is equivalent to real-world validation data.

## Design

- preserves the firmware 36-column schema
- preserves activity labels
- preserves firmware sentinel `-999`
- keeps PPG validity flags untouched
- applies correlated motion drift rather than only independent row noise
- resamples timing from the source session's empirical normal/stall intervals
- writes provenance next to every generated dataset

## Example

```powershell
python tools\dataset_expansionugment_session.py `
  --source "datasetseal sensor data\walking.csv" `
  --output "datasets\generated\walking_aug_001.csv" `
  --config "tools\dataset_expansionugmentation_config.json" `
  --seed 20261118
```

This augmentation tool is one component of the dataset expansion workflow.
Run the end-to-end acceptance sequence described below to validate all stages.

## Milestone 2.11 Acceptance

Run the complete dataset-expansion acceptance sequence (2.11.1–2.11.9) from
the repository root:

```powershell
.\venv\Scripts\python.exe tools\dataset_expansion\run_dataset_expansion_acceptance.py
```

The runner refreshes the individual stage reports, reruns timing, hazard, and
hard-negative validators, and writes the consolidated report to
`reports\dataset_expansion\acceptance\dataset_expansion_acceptance_2.11.9.json`.
It exits nonzero if any stage fails. It validates existing datasets and does
not generate or modify dataset CSVs.

The acceptance stages are:

| Stage | Check |
|---|---|
| 2.11.1 | Dataset inventory |
| 2.11.2 | Schema consistency |
| 2.11.3 | Provenance completeness |
| 2.11.4 | Synthetic-status integrity |
| 2.11.5 | Activity and profile coverage |
| 2.11.6 | Sentinel and validity integrity |
| 2.11.7 | Timing integrity |
| 2.11.8 | Hazard and hard-negative coverage |
| 2.11.9 | Consolidated final acceptance report |
