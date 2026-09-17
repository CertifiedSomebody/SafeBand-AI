# Activity DL V2 Experiment

V1 showed raw temporal ACC improves the handcrafted-feature baseline, but RESTING/SITTING remains the bottleneck.

V2 therefore increases temporal modeling capacity without changing the evaluation rules.

## Models
- TCN: residual Conv1D blocks with dilation 1/2/4/8.
- CNN+GRU: local CNN feature extraction followed by temporal GRU.

## Integrity
- Same BITS2 four-class activity task.
- FALL remains separate.
- 5-fold StratifiedGroupKFold by subject.
- Outer folds are untouched for model selection.
- Standardization is fit only on training subjects.
- Early stopping uses a grouped inner validation split.
- Final deployment model is produced only after outer evaluation.

## Shape bug prevention
Accepted input layouts are `(N,3,40)` and `(N,40,3)`, canonicalized to `(N,3,40)`.
The old invalid operation `X.transpose(1,2)` is not used.
