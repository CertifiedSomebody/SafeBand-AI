# AICOS-HAR Training Plan

After the zero-window diagnostic passes, the training sequence is:

1. **Classical ACC benchmark**
   - Official Fold 1–5: training/development.
   - Official `test`: untouched final evaluation.
   - Subject-independent evaluation.
   - 38-feature SafeBand ACC contract.
   - ExtraTrees, RandomForest, HistGradientBoosting, RBF-SVM.
   - Report accuracy, balanced accuracy, macro-F1, per-class F1, confusion matrix.

2. **Tiny CNN**
   - Raw ACC windows `(3, 100)`.
   - Training-only normalization.
   - Same official fold protocol.
   - Compact 1D CNN suitable as a TinyML-oriented baseline.

3. **Comparison**
   - Compare AICOS-HAR results against the frozen SafeBand BITS2 activity baselines.
   - Do not merge datasets or tune against the official AICOS-HAR test fold.
   - If AICOS-HAR substantially improves stationary-class separation, preserve it as external/generalization evidence rather than silently replacing prior benchmarks.
