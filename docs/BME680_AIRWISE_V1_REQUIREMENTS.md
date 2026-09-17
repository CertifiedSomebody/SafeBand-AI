# BME680 AIRWISE V1 — Requirements

Python packages:
- numpy
- pandas
- scikit-learn
- joblib

Recommended environment: the same SafeBand project virtual environment used for the other ML branches.

Expected raw path:
`datasets/raw/AIRWISE/data/indoor/sensor_1min/`

Expected files:
- `office_minute_averages.csv`
- `kitchen_minute_averages.csv`
- `hallway_minute_averages.csv`

All scripts are designed to be launched from the repository root with:
`python tools\script.py`

The scripts explicitly add the repository root to `sys.path` so the common `ModuleNotFoundError: No module named 'ai'` issue from earlier walking experiments is avoided.
