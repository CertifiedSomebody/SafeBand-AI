# SafeBand AI — PAMAP2 Stationary Model Suite V4

Final comparative benchmark for LYING / SITTING / STANDING.

Models:
- RBF-SVM
- ANN/MLP
- Tiny 1D CNN
- BiLSTM
- BiGRU

## Installation in your repository

Copy the three files under `tools/` directly into the existing SafeBand repository:

`tools/pamap2_stationary_v4_common.py`
`tools/train_pamap2_stationary_v4.py`
`tools/__init__.py`

Do not create nested train directories.

Dataset expected:
`datasets/processed/pamap2/stationary_accgyro_windows.npz`

## Run from repository root

`python tools\train_pamap2_stationary_v4.py`

Optional:
`python tools\train_pamap2_stationary_v4.py --epochs 35`

Output:
`models/pamap2_stationary_v4/model_suite_v4_report.json`

Subject 5 is retained. No source dataset is changed.
