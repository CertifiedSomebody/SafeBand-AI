# SafeBand HR V4 final temporal benchmark

This fixes the prior V4 indexing defect: split-local predictions are never
indexed with global dataset indices.

It also verifies the loaded V2 feature contract before prediction.

Run:

```powershell
python tools\evaluate_ppgdalia_temporal_v4.py `
  --input datasets\processed\ppgdalia_windows.npz `
  --model models\ppgdalia_hr_e4_reference_v2.joblib `
  --report-out models\ppgdalia_hr_e4_temporal_v4_report.json
```

The report always contains regression metrics plus accuracy within ±1/±3/±5/±10 BPM.

Temporal parameters are selected on validation subjects only. Test subjects
are evaluated once using the supplied final V2 artifact.
