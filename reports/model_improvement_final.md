# Model Improvement Final Report

## 1. Baseline
The runtime baseline is the BITS-2 Motion V2 ExtraTrees model with 38 ACC features.

## 2. External Independent Benchmark
AICOS-HAR coarse benchmark selected 2.0 s/RandomForest in existing subject-disjoint results: accuracy 96.97%, balanced accuracy 93.30%, macro F1 93.98%, weighted F1 96.95%; STATIC 94.72%, WALKING 96.71%, RUNNING 87.24% in the requested reference summary.

## 3. Real SafeBand Replay
See `real_safeband_replay_baseline.json` and `real_safeband_predictions.csv`.

## 4. Weaknesses
Metrics below 95% are dominated by external running/static weakness and real-domain mismatch; real one-class recordings have insufficient support to measure complete activity classification.

## 5. Improvement Experiments
Existing controlled experiments include BITS-2 nested subject CV (`activity_strong_v4`), External AICOS/SISFall window ablation, tree/SVM candidates, and this replay at 0.5/1/2/3-second row windows.

## 6. Best Model
No real-SafeBand-only model is selected. `activity_strong_v4` is a development candidate selected by subject-grouped macro-F1.

## 7-9. Before/After, Confusion, Error Analysis
Prediction-level rows and baseline confusion matrices are in `real_safeband_predictions.csv` and `reports/confusion_matrices/`.

## 10. Limitations
The 2.12 gate remains blocked. The real CSVs are replay evidence, not independent product validation.

## 11. Future Validation
Collect multiple independent sessions/users for STATIC, WALKING, RUNNING, transitions, high motion, and controlled hazard/fall scenarios under varied placement and timing conditions.
