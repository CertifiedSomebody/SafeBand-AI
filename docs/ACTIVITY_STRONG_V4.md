# SafeBand Activity Strong V4

## Purpose

V3 tested whether a confusion-guided specialist could improve the activity model. The experiment found a recurring FALL/SITTING confusion region, but the hierarchical system improved macro-F1 by only **0.00080** overall and was not consistently better across folds. Therefore V3 is retained as a **lesson learned**, not as the deployment architecture.

V4 returns to the simpler deployment-relevant activity model and strengthens it through **controlled model selection**, while keeping the fall detector separate.

## What is being strengthened

The ordinary activity task is:

- RESTING
- SITTING
- WALKING
- RUNNING

FALL is intentionally excluded from this classifier. SafeBand retains the separate fall detector and later risk engine.

The sensor contract is unchanged: the existing 38 BITS2 accelerometer-derived features are used. No new sensor information is being assumed.

## Methodology

V4 uses a nested, subject-independent evaluation:

1. Five outer `StratifiedGroupKFold` folds provide the unbiased evaluation.
2. Subjects, not individual windows, define the groups.
3. Inside each outer training partition, three grouped folds select the model configuration.
4. The outer validation subject set is never used for model selection.
5. Selection metric: macro-F1, then balanced accuracy.
6. Candidate family includes ExtraTrees, Random Forest, and RBF-SVM.
7. Tree candidates test no weighting, square-root class balancing, and full balanced weighting; ExtraTrees also tests `min_samples_leaf` values 1–3.
8. After evaluation, a deployment model is selected using grouped CV over the complete activity dataset and fitted on all available activity windows.

This prevents the model from being selected on the same subjects used to report its outer-fold performance.

## Why these candidates

The current feature representation is statistical/spectral rather than raw sequence data. Tree ensembles are therefore natural baselines, while an RBF-SVM provides a nonlinear scaled comparator. Weighted variants explicitly test whether improving minority-class contribution helps the stationary classes without assuming that it will.

Wearable HAR literature supports using accelerometer/gyroscope information for activity/posture recognition, and stationary posture discrimination can be sensitive to the available orientation/gravity information. This is consistent with SafeBand's later plan to validate the model with the BNO055 IMU rather than claiming that accelerometer-only results solve sitting/resting discrimination. citeturn0search1turn0search2turn0search4

## V3 lesson learned retained

V3 should remain in the research record:

> A confusion-guided specialist was methodologically tested using nested subject-grouped validation. FALL/SITTING was selected in four of five outer folds, but the overall macro-F1 gain was only 0.00080 and fold-level gains were inconsistent. The additional hierarchy was therefore rejected for deployment.

This is evidence against unnecessary architectural complexity, not evidence that confusion analysis is useless.

## Acceptance rule for V4

V4 is not accepted merely because one candidate wins one split.

A candidate should replace the existing activity model only if the outer grouped evaluation shows a reproducible improvement in macro-F1 and/or balanced accuracy without a material regression in the safety-relevant class behavior. The fold distribution must also be inspected.

If V4 does not produce a meaningful improvement, keep the existing model and move effort toward the missing information: real hardware-domain data, orientation-aware features, gyro/magnetometer input, and synchronized multimodal fusion.

## Run

From the repository root:

```powershell
python tools\train_activity_strong_v4.py
```

Optional:

```powershell
python tools\train_activity_strong_v4.py --input datasets\processed\bits2\activity_windows.csv --out models\activity_strong_v4
```

Outputs:

```text
models/activity_strong_v4/
├── activity_model.joblib
└── activity_strong_v4_report.json
```

## Important interpretation rule

Do not compare a V4 outer-CV score directly to an old single split as if they were identical experiments. The old result is a historical benchmark. V4's grouped outer folds provide a more robust estimate of subject-independent performance.

The final deployment choice should be based on the complete evaluation record, not on the largest isolated number.
