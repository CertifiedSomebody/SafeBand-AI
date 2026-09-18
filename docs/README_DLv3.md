# SafeBand Activity DL V3 — Representation Fusion

## Why V3 exists

DL V1 was better than the handcrafted ACC model, while the stronger TCN and
CNN+GRU models in V2 were worse. That means simply adding model capacity is
not the current bottleneck.

V3 therefore tests a different hypothesis:

> The raw temporal CNN and the handcrafted statistical/spectral representation
> may contain complementary information.

Instead of replacing V1, V3 combines:
- a lightweight raw-window CNN
- the existing locked 38-feature ExtraTrees ACC model

Their class probabilities are blended. The blend weight is selected ONLY on
the inner grouped validation data of each outer fold.

## Integrity

- Same 8277 BITS2 windows.
- Same 41 subjects.
- Same RESTING/SITTING/WALKING/RUNNING task.
- FALL remains separate.
- Same 5-fold subject-independent outer evaluation.
- CNN normalization uses training subjects only.
- Blend weight uses inner validation only.
- Outer test subjects are never used to select the blend.

## Why this is preferable to blindly making the CNN deeper

V2 showed:
- TCN: 81.7204% accuracy, 79.7316% balanced accuracy, 79.7889% macro-F1.
- CNN+GRU: 81.0680% accuracy, 78.7646% balanced accuracy, 78.7058% macro-F1.

The stronger sequence models therefore did not beat the V1 Tiny CNN result.
V3 asks whether representation diversity is more useful than additional
temporal-model capacity.

## Run

From repo root:

    python tools/train_activity_dl_v3.py

The report is written to:

    models/activity_dl_v3/activity_dl_v3_report.json

Do not use the blend result unless the outer pooled metrics and per-class
confusion are actually better.
