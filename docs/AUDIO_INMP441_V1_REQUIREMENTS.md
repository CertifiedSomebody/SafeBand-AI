# SafeBand AI — INMP441 Audio V1 Requirements

## Purpose

Build an acoustic-event recognition component that can later contribute evidence to SafeBand sensor fusion.

## Required behavior

The audio branch must distinguish:
- acoustic signal measurement
- acoustic level
- semantic event classification

A high audio level is not automatically a scream or emergency.

## Dataset requirements

Before training, the selected dataset must provide:
- raw audio or lossless waveform access
- explicit event labels
- subject/recording identity
- sufficient samples per class
- a defensible train/validation/test grouping strategy

Prefer subject-disjoint evaluation.

## Current feature set

The V1 extractor uses:
- RMS
- absolute peak
- crest factor
- zero-crossing rate
- DC offset
- dynamic range
- spectral centroid
- spectral bandwidth
- spectral flatness
- spectral rolloff
- normalized spectral-band energies
- dominant frequency

## Runtime contract

The model returns:

```text
event
confidence
probabilities
model_name
model_version
window_samples
```

## Deployment note

The Python implementation is the Phase-1 reference. Actual INMP441 acquisition will occur through ESP32-S3 I2S firmware in Phase 2.

The feature extraction algorithm should be reproduced on-device or replaced with an equivalent embedded implementation before TinyML deployment.

## No final event labels yet

Do not hard-code "SCREAM = EMERGENCY" before a dataset and validation protocol have been selected.

The fusion layer may use a validated audio-event probability as one evidence source.
