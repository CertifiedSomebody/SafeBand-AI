# PAMAP2 Stationary Model Suite V4

Controlled final comparison of classical ML, ANN, CNN and recurrent sequence models.

All models share identical outer 5-fold subject-independent splits and seed 42. Outer test subjects are never used for selection. Neural models use inner validation for early stopping; final outer-training models are refit for the selected epoch count. Feature and sensor normalization is training-only.

The benchmark is descriptive. The resulting report should be interpreted using pooled metrics, per-class F1, confusion matrices and fold stability rather than a single score.
