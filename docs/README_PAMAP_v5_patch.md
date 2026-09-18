# SafeBand AI — PAMAP2 Stationary Model Suite V5 PATCHED

Corrected model-family benchmark for LYING/SITTING/STANDING using hand/wrist ACC+GYRO.

Models: RBF-SVM, ANN/MLP, Tiny CNN, BiLSTM, BiGRU.

Protocol: 5-fold StratifiedGroupKFold by PeopleId; all selection uses grouped inner CV; outer test untouched; normalization is training-only.

V5 corrections: materialized inner CV splits, Windows-safe SVM `n_jobs=1`, `num_workers=0`, explicit CNN/RNN tensor shapes, grouped neural epoch selection, strict NPZ validation, explicit 109-feature contract.

Run from repository root:

`python tools\train_pamap2_stationary_v5.py`

Report: `models\pamap2_stationary_v5\model_suite_v5_report.json`
