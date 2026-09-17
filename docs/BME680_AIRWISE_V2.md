# SafeBand AI — BME680 AIRWISE V2

V2 is the next experiment over the frozen V1 baseline.

V1: 91.40% accuracy, 79.47% balanced accuracy, 78.24% macro-F1 on the untouched chronological test.

V2 changes: causal 5/15/30-minute rolling statistics, first/5-step changes, gas-log and interaction features, cyclic time features, stronger tree ensembles, and validation selection by macro-F1 -> balanced accuracy -> accuracy.

Protocol: chronological 60/20/20 independently within office/kitchen/hallway; test untouched; final model refit on train+validation only; IAQ_class is never an input feature.

Commands:
```powershell
python tools\audit_airwise_v2.py
python tools\train_airwise_bme680_v2.py
python tools\validate_bme680_v2.py
```

Do not overwrite `models/bme680_airwise_v1/`.

For research reporting, 95% raw accuracy alone is not sufficient because AIRWISE has seven imbalanced classes. Report accuracy, balanced accuracy, macro-F1, per-class recall, and confusion matrix.
