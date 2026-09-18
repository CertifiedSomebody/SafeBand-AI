# SafeBand AI — PAMAP2 Stationary Model Suite V5

Final Phase-1 stationary-state model-family benchmark for SafeBand AI.

## Models
1. RBF-SVM — deterministic handcrafted window features
2. ANN/MLP — same handcrafted feature contract
3. Tiny 1D CNN — raw 6-channel ACC+GYRO
4. BiLSTM — raw 6-channel ACC+GYRO
5. BiGRU — raw 6-channel ACC+GYRO

## Scientific protocol
- PAMAP2 stationary windows: LYING / SITTING / STANDING
- 2 s windows, 50% overlap, wrist/hand ACC+GYRO only
- 5-fold StratifiedGroupKFold by PeopleId
- all model/hyperparameter/epoch selection is confined to a 4-fold grouped inner CV
- outer test folds remain untouched
- normalization is fitted only on outer training data
- seed = 42
- no subject removal
- identical pooled metrics for all models

## Run
From repository root:

```powershell
python tools	rain_pamap2_stationary_v5.py
```

Optional:
```powershell
python tools	rain_pamap2_stationary_v5.py --data datasets\processed\pamap2\stationary_accgyro_windows.npz --out models\pamap2_stationary_v5
```

Report:
`models/pamap2_stationary_v5/model_suite_v5_report.json`

V5 specifically fixes the V4 joblib/generator failure by materializing grouped inner splits before GridSearchCV. It also removes ungrouped neural early-stopping selection and uses grouped inner validation for model/epoch selection.
