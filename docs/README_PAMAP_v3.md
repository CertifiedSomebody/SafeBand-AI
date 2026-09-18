# SafeBand AI — PAMAP2 Stationary V3

This package is a controlled investigation of Subject 5 from the PAMAP2 stationary benchmark.

## Important design

Subject 5 is treated as a **completely unseen test subject** in a dedicated diagnostic run. The model never uses Subject 5 for hyperparameter selection. A second sensitivity benchmark removes Subject 5 and repeats the same 5-fold grouped protocol on the remaining subjects.

The original all-subject benchmark is also rerun as a control using the **exact 76-feature RBF-SVM representation from V2**. This isolates the effect of subject exclusion.

Subject 5 is NOT deleted from the canonical dataset.

## Repository layout

Extract/copy the files so these scripts are directly under the repository `tools` directory:

```text
SafeBand-AI-main/
├── datasets/processed/pamap2/stationary_accgyro_windows.npz
├── tools/
│   ├── audit_pamap2_stationary_v3.py
│   └── train_pamap2_stationary_v3.py
└── models/
```

## Commands

From the SafeBand repository root:

```powershell
python tools\audit_pamap2_stationary_v3.py
python tools\train_pamap2_stationary_v3.py
```

Optional explicit subject:

```powershell
python tools\train_pamap2_stationary_v3.py --test-subject 5
```

Optional explicit dataset:

```powershell
python tools\train_pamap2_stationary_v3.py --data datasets\processed\pamap2\stationary_accgyro_windows.npz
```

## Outputs

```text
models/pamap2_stationary_v3/audit_report.json
models/pamap2_stationary_v3/subject_class_feature_summary.csv
models/pamap2_stationary_v3/stationary_v3_features.npy
models/pamap2_stationary_v3/pamap2_stationary_v3_subject5_report.json
```

The source NPZ is never modified.
