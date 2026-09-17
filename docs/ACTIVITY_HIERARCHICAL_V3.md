# SafeBand Activity Recognition V3 — Confusion-Guided Specialist Experiment

## Why this experiment exists

The original idea was to use one model for activities it handled well and introduce another model for an activity/dataset where the first model performed poorly.

The cross-dataset experiment showed why that interpretation is unsafe: poor transfer from FORTH-TRACE to BITS2, or vice versa, demonstrates **domain shift**, not necessarily an activity-specific weakness. A model trained on one wearable/domain cannot be assumed to fail only on one class in another domain.

We therefore changed the research question:

> **Within the same deployment-relevant domain (BITS2), can a specialist model improve a specific, empirically measured confusion of a general activity model?**

This is a data-driven hierarchical / mixture-of-experts experiment. Hierarchical HAR is an established approach, including work that builds activity groups from measured confusion relationships rather than relying only on prior assumptions. citeturn0search2turn0search1

## What we learned

1. Cross-dataset failure is primarily evidence of domain shift.
2. We must not map unrelated labels merely to make datasets look compatible.
3. A specialist should be introduced because the **same-domain error structure** shows a real weakness, not because another dataset happens to disagree.
4. Test data must remain untouched while the specialist group is selected.
5. The specialist must be evaluated against the exact same outer folds as the baseline.
6. If the hierarchical system does not improve held-out macro-F1 / balanced accuracy, the specialist architecture is rejected.

## V3 design

```text
BITS2 sensor features
        |
        v
  General Model A
        |
        v
 predicted activity
        |
        +---- prediction belongs to empirically selected confusion pair?
        |                         |
        | NO                      | YES
        v                         v
  keep Model A             Specialist Model B
                                  |
                                  v
                           final activity
```

### Leakage control

Each outer fold is the evaluation unit.

1. Split subjects with `StratifiedGroupKFold`.
2. Train baseline only on outer-training subjects.
3. Inside outer-training subjects, run a separate 3-fold grouped CV.
4. Use that inner CV confusion matrix to select the most mutually confused pair.
5. Train the specialist only on outer-training samples belonging to that pair.
6. Evaluate both baseline and hierarchical systems on untouched outer-validation subjects.

Thus the outer validation labels are never used to choose the specialist.

## Selection rule

For every outer fold, compute row-normalized confusion from the inner CV. For classes `i` and `j`, use the symmetric score:

`confusion(i,j) = normalized_CM[i,j] + normalized_CM[j,i]`

The highest-scoring off-diagonal pair becomes the specialist pair for that fold.

This is intentionally data-driven. We do **not** predeclare SITTING/RESTING, WALKING/RUNNING, or any other pair.

## Baseline

The baseline is the current BITS2 activity model family: ExtraTrees with balanced class weights.

The purpose of V3 is not to replace the baseline automatically. It is an A/B experiment:

- Baseline: one general model.
- Hierarchical: same general model + confusion-guided specialist.

## Acceptance rule

A specialist architecture is retained only if it provides a reproducible improvement on the outer subject-independent evaluation, especially macro-F1 and balanced accuracy, without relying on test-fold information.

A negative or negligible delta is a valid result and means the additional complexity is not justified.

## Relationship to FORTH-TRACE

FORTH-TRACE remains an external benchmark and cross-domain robustness study. Its cross-domain transfer failure is not used to define the BITS2 specialist.

The BITS2 specialist experiment is deliberately **same-domain** because the research question is now activity-specific error correction rather than cross-device/domain adaptation.

## Run

From repository root:

```powershell
python tools\activity_hierarchical_v3.py
```

Optional:

```powershell
python tools\activity_hierarchical_v3.py --input datasets\processed\bits2\activity_windows.csv --out-dir models\activity_hierarchical_v3
```

Output:

`models/activity_hierarchical_v3/hierarchical_v3_report.json`

## Important

This experiment does not claim that a specialist is useful before the results are observed. The data chooses the specialist pair, and the outer evaluation decides whether the hierarchy is worth keeping.
