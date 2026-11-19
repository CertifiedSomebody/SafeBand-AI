# SafeBand AI development components

`TemporalFallStateMachine` is a model-agnostic temporal confirmation prototype.
It accepts a probability from a separate model, requires valid motion and
timing quality, confirms only consecutive high-probability samples, latches an
event until recovery, and rejects non-monotonic timestamps. Its thresholds
and sample counts are development defaults, not trained/calibrated
parameters, and have not been validated on wearable hardware.

`FallModelContract` prevents production eligibility unless the artifact has a
valid SHA-256 identity, at least three independent source sessions, passed
independent evaluation, passed hardware validation, and an independently
validated release stage. `CanServeProduction` additionally checks the
artifact bytes against the recorded digest. This repository's current
2.12 gate is blocked, so no model is approved by these changes.
