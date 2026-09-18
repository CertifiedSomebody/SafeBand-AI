# SafeBand BME680 AIRWISE V2.1 — correction and research-safety patch

## What was fixed

1. The processed AIRWISE filename is discovered automatically. The current prepared file is `datasets/processed/airwise_bme680/airwise_bme680_features.csv`.
2. Temporal rolling features use explicit causal, per-location rolling calculations with stable index alignment.
3. The primary model excludes `IAQ_proxy`, its z-score, and the target `IAQ_class`. This prevents the primary result from simply learning an IAQ-derived labeling rule.
4. The chronological 60/20/20 split remains per location.
5. Model selection uses validation macro-F1, then balanced accuracy, then accuracy.
6. The test set remains untouched until final evaluation.
7. The audit now reports the relationship between `IAQ_proxy` and `IAQ_class` as a leakage diagnostic rather than silently using the proxy as an input.
8. Artifact validation checks the research guardrails.

## Run

```powershell
python tools\audit_airwise_v2.py
python tools\train_airwise_bme680_v2.py
python tools\validate_bme680_v2.py
```

You can also explicitly pass the current prepared CSV:

```powershell
python tools\audit_airwise_v2.py --input datasets\processed\airwise_bme680\airwise_bme680_features.csv
python tools\train_airwise_bme680_v2.py --input datasets\processed\airwise_bme680\airwise_bme680_features.csv
```

## Interpretation rule

Do not target or report a generic “95% accuracy” as a research-grade requirement. This is a multiclass environmental classification task, so accuracy must be accompanied by balanced accuracy, macro precision, macro recall, macro-F1, per-class results, and the confusion matrix. A high score obtained by feeding an IAQ-derived proxy back into its own class labels is not a valid evidence of independent sensor intelligence.
