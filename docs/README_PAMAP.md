# SafeBand PAMAP2 Stationary V1

Purpose: establish a clean external benchmark for stationary/postural activity recognition using PAMAP2-derived data.

## Source
Place the verified Kaggle PAMAP2 CSV at:
`datasets/raw/PAMAP2/dataset2.csv`

## Pipeline
1. Run schema/data audit.
2. Build non-crossing 2-second hand/wrist ACC+GYRO windows.
3. Evaluate with subject-independent 5-fold StratifiedGroupKFold.
4. Use only LYING, SITTING, STANDING for the first benchmark.

## Commands

From repository root:

```powershell
python toolsudit_pamap2.py
python toolsuild_pamap2_stationary_windows.py
python tools	rain_pamap2_stationary_v1.py
```

Optional custom paths:

```powershell
python toolsudit_pamap2.py "datasetsaw\PAMAP2\dataset2.csv"
python toolsuild_pamap2_stationary_windows.py --csv "datasetsaw\PAMAP2\dataset2.csv"
```

Important: this V1 does not use chest/ankle sensors and does not merge windows across subject or activity boundaries.
