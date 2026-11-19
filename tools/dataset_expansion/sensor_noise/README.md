# SafeBand AI — Milestone 2.10.4

## Sensor Noise Diversity

This stage generates controlled sensor-noise variants from real SafeBand
sessions while preserving the firmware schema.

## Noise models

### BNO055
- Gaussian measurement noise
- temporally correlated drift
- transient spikes

### BME680
- channel-dependent Gaussian noise
- slower correlated drift
- low-rate transient disturbances

### MAX30102 / PPG
- measurement-domain perturbation
- only applied to valid measurements
- `-999` remains unchanged for unavailable values

### Battery
- low-amplitude measurement noise
- quantization-compatible perturbation

## Guarantees

- Same row count
- Same column order
- Same firmware schema
- Validity flags remain binary
- Invalid measurements retain `-999`
- Deterministic with fixed seed
- Original source is never modified

## Status

Synthetic augmentation only.

This stage does not constitute real-world sensor accuracy validation.
