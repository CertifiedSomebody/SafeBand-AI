# SafeBand Dataset Expansion — 2.10.8 Provenance

This stage creates a consolidated metadata manifest for generated datasets.

## Purpose

The manifest records:

- dataset identity
- source dataset
- source path
- generation stage
- generation tool
- profile
- random seed
- timestamp
- row/column counts
- activity
- synthetic status
- parent dataset
- schema version
- augmentation parameters
- validation report
- provenance file

## Synthetic-status policy

Generated datasets must remain explicitly classified.

`SYNTHETIC_UNVALIDATED`

means the dataset was generated successfully but has not established
real-world validity.

`SYNTHETIC_VALIDATED`

may only be used after an explicit validation stage.

`REAL`

is reserved for directly collected sensor data.

The metadata system does not promote a synthetic dataset to REAL.

## Output

The session manifest is generated under:

`reports/dataset_expansion/provenance/`

Validation reports are also stored there.

## Reproducibility

Existing dataset-level provenance files remain authoritative for
generation parameters and seeds. The session manifest consolidates
those records into one machine-readable index.
