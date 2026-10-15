# AICOS-HAR Activity Recognition — Pipeline Plan V2

## Why this stage exists

The initial inventory contained 5,231 metadata acquisitions and reported
4,457 acquisitions with raw files. A later reconciliation showed that all
5,231 metadata AcquisitionIDs exactly match the inventory AcquisitionIDs.
Therefore the apparent 774-acquisition discrepancy is a raw-path discovery
issue, not an AcquisitionID metadata mismatch.

The next stage audits the actual filesystem hierarchy before any ML windowing.

## Locked principles

- Subject-independent evaluation.
- Preserve the dataset's official fold assignments.
- Do not assume a canonical sampling rate before measuring the raw data.
- Do not silently discard acquisitions because of naming differences.
- Do not modify raw AICOS-HAR files.
- Keep path reconciliation, sampling audit, preprocessing and training as
  separate reproducible stages.
- All scripts must run from repository root.
- Training must use training-only normalization/fit operations.
- Test/official evaluation data remain untouched until final evaluation.

## Intended training branch

Once the audit is complete, the activity branch will benchmark at least:

- a classical handcrafted-feature baseline
- a raw temporal model
- optionally a compact fusion model if the evidence supports it

Primary metrics:

- accuracy
- balanced accuracy
- macro-F1
- per-class precision/recall/F1
- confusion matrix

Secondary robustness slices:

- device position
- sensor availability
- participant/fold
- activity class

No model will be declared better merely because of one metric.

## Target SafeBand activity vocabulary

AICOS-HAR provides:

- walking
- standing
- sitting
- running
- lying down
- stairs
- ramps
- lifts

These can later support the SafeBand vocabulary:

- STATIONARY: sitting / standing / lying
- DYNAMIC: walking / running / stairs / ramps
- OTHER / UNKNOWN
- fall remains a separate safety detector

Lift state should not be merged into ordinary locomotion without a design
decision, because it may be useful context for the fusion engine.

## Current status

- Initial metadata/raw audit: complete
- Acquisition inventory: complete
- AcquisitionID reconciliation: complete
- Raw path discrepancy explanation: pending
- Sampling/device/position audit: pending
- Canonical windowing contract: pending
- Training: intentionally blocked until the above are resolved
