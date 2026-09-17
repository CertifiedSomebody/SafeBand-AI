# PAMAP2 Stationary V5 — Final Model-Family Benchmark

## Purpose
Compare classical ML, ANN, CNN, and recurrent neural architectures on the same SafeBand-relevant stationary-state task.

## Task
Classes:
- LYING
- SITTING
- STANDING

Input:
- wrist/hand accelerometer XYZ
- wrist/hand gyroscope XYZ
- 2-second windows
- raw tensor shape `(N, 6, 200)`

The benchmark is a diagnostic model-family comparison. It does not declare a deployment winner until the measured report is reviewed.

## Validation
The outer evaluation is subject-independent:
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by `PeopleId`.

Within every outer training partition:
- RBF-SVM: 4-fold grouped CV over C/gamma.
- ANN: 4-fold grouped CV over hidden-layer configurations and max_iter.
- CNN/BiLSTM/BiGRU: 4-fold grouped CV over training-epoch candidates.

The outer test partition is never used for model selection.

## Metrics
Each model reports:
- Accuracy
- Balanced Accuracy
- Macro-F1
- per-class precision/recall/F1
- pooled confusion matrix
- fold-level selections
- elapsed training time

## Important implementation corrections from V4
1. Inner `StratifiedGroupKFold.split()` results are converted to a list before being passed to joblib-backed `GridSearchCV`; this prevents the Windows `cannot pickle 'generator' object` failure.
2. Neural model selection no longer relies on an ungrouped random validation fraction.
3. Every model uses the same outer subject-independent protocol.
4. Training-only normalization is enforced.
5. The feature contract is asserted rather than silently changing shape.

## Interpretation
The report should be compared with the previously frozen PAMAP2 Stationary V1/V2 results and the SafeBand activity experiments. Do not tune again solely to chase a small metric difference. Use the suite to determine whether additional architecture complexity provides useful evidence for the stationary-state branch.
