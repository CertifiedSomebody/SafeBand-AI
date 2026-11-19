# SafeBand Final ML Development Report

- Classification: **DEVELOPMENT EVIDENCE**
- Synthetic sessions: 24
- Synthetic rows: 9698
- Activity test accuracy: 0.8770
- Activity test balanced accuracy: 0.6082
- Activity test macro F1: 0.4768
- Hazard test accuracy: 0.9119
- Hazard test recall: NOT_EVALUABLE (no hazard support in test split)
- Hazard test F1: NOT_EVALUABLE (no hazard support in test split)
- Selected activity ablation: MINUS_BME
- Selected activity model: HistGradientBoosting
- Selected hazard model: ExtraTrees
- Hazard threshold: 0.050

## 2.12 Status

BLOCKED: synthetic sessions are not independent real source sessions.

## Limitations

- Real recordings are replay evidence only.
- No real SafeBand running or hazard ground truth was used.
- HR/SpO2 are telemetry pipelines, not clinically validated predictions.
