# Multi-dataset benchmark notes

The benchmark intentionally uses the existing 84-feature V6 representation.
No new feature engineering is introduced here. This isolates whether adding
dataset diversity improves domain robustness.

The primary result is the combined-model performance on held-out subjects from
both datasets. Transfer diagnostics are secondary.

A good result should improve SisFall without destroying BITS-2 performance.
A poor result is still useful: it demonstrates that additional domain
normalization, sensor-aware training, or target-hardware data is required.
