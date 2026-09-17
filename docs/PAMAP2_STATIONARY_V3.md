# PAMAP2 Stationary V3 — Subject 5 Diagnostic

## Question

V2 showed one severe cross-subject failure when Subject 5 was held out. V3 tests whether that behavior is specific to Subject 5 and quantifies the effect of excluding that subject without changing the classifier.

## Three results

1. **All-subject control:** exact V2 76-feature RBF-SVM + 5-fold StratifiedGroupKFold.
2. **Subject-5 isolated test:** train on the other subjects, select C/gamma only by grouped inner CV on those training subjects, then evaluate once on Subject 5.
3. **No-Subject-5 sensitivity:** remove Subject 5 and rerun the same 5-fold grouped benchmark on the remaining subjects.

The isolated Subject-5 test is the cleanest answer to “can Subject 5 be the test subject?” because it is never used for model selection.

## Scientific rule

A score improvement after removing Subject 5 does not establish that Subject 5 is bad data. It may simply represent genuine subject/domain variation. Formal exclusion requires independent evidence of corruption, incompatible recording conditions, or another documented data-quality problem.
