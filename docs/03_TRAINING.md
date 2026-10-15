# AICOS-HAR Stage 7 — Activity ML Benchmark

After Stage 6 passes:

1. Classical ACC benchmark: ExtraTrees, RandomForest, HistGradientBoosting, RBF-SVM.
2. Official `Fold=test` remains untouched until final evaluation.
3. Classical models train on folds 1–5; no test-driven tuning.
4. Tiny temporal CNN uses acquisition-grouped validation inside folds 1–5 and evaluates the official test fold once.
5. Report accuracy, balanced accuracy, macro-F1, class-wise F1, and confusion matrix.

The first benchmark is ACC-only to establish a clean baseline before adding gyro or other sensors.
