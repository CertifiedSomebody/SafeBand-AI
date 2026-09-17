# SafeBand Multi-Dataset V5

## Why V5 exists

V4 already established a strong pooled temporal benchmark, but the held-out
BITS-2 false-positive rate was much higher than SisFall. V5 therefore changes
the training distribution rather than tuning the held-out test set.

## Changes from V4

- Same 60-sample / 20 Hz / step-30 window contract.
- Same frozen 84 fall features.
- Same subject-independent split procedure.
- Equal total training weight for BITS-2 and SisFall.
- Equal FALL/NON_FALL weight within each dataset.
- Optional train-only hard-negative mining with GroupKFold OOF predictions.
- Same validation-only temporal operating-point selection.
- Same final refit on TRAIN + VALIDATION and single TEST evaluation.

## First run

```bash
python tools/train_multidataset_v5_balanced.py \
  --bits2 datasets/processed/bits2/bits2_fall_windows_v6.csv \
  --sisfall datasets/processed/sisfall/sisfall_windows_v6.csv \
  --out models/multidataset_v5_event
```

Then compare the report against:

`models/multidataset_v4_event/multidataset_v4_event_report.json`

## Hard-negative experiment

Run separately so the effect is measurable:

```bash
python tools/train_multidataset_v5_balanced.py \
  --bits2 datasets/processed/bits2/bits2_fall_windows_v6.csv \
  --sisfall datasets/processed/sisfall/sisfall_windows_v6.csv \
  --out models/multidataset_v5_event_hardneg \
  --hard-negative \
  --hard-negative-quantile 0.90 \
  --hard-negative-weight 3.0
```

Do not report a V5 result until the script has completed and the generated
report has been inspected. The existing prototype remains unchanged.
