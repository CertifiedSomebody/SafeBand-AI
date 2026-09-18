# BNO055 ML Benchmark V1 — Experimental Protocol

## Research question

Does adding gyroscope and magnetometer information to accelerometer-only windows
provide measurable information for SafeBand activity/emergency-state recognition?

## Controlled comparison

The same frozen V2.1 windows and the same subject-grouped folds are used for:

```text
ACC
ACC + GYRO
ACC + GYRO + MAG
```

The comparison is therefore a sensor-ablation experiment, not a comparison of
different datasets.

## Leakage prevention

Subject identity is never shared between an outer training fold and its test fold.

For deep learning, channel-wise mean/std normalization is calculated from the
outer training fold only.

The internal validation split used for early stopping is also taken only from
the outer training fold.

## Event analysis

The window builder supplies `is_event_window`. These windows are not removed from
the ordinary activity benchmark. Their metrics are additionally reported so that
transition/fall behavior can be inspected separately.

## Interpretation

Synthetic performance answers whether the implementation can learn the generated
sensor patterns under a controlled subject-independent protocol.

It does **not** establish:

- real BNO055 accuracy,
- real wrist-placement robustness,
- real-world magnetic-environment robustness,
- hardware timing behavior,
- deployment latency or power consumption.

Those require real SafeBand hardware data.

## Frozen artifact

Input artifact:

`bno055_windows_v2_1.npz`

Synthetic generator V2.1 is frozen before this benchmark.
