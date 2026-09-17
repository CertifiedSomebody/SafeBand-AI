# SafeBand AI — Phase 1 Gap Audit and Integration Plan

## Basis

This audit compares the current repository against the project proposal/presentation and the AI work completed to date.

The final project is intended to continuously monitor activity, physiological and environmental conditions, use synchronized multi-sensor data, perform activity recognition and emergency-event detection, reduce false alarms through sensor fusion, and support lightweight embedded AI plus communication. 

## What was already strong

### Motion / fall
A multi-dataset fall branch has been developed and evaluated using BITS-2 and SisFall. The branch is sufficiently mature for Phase 1 and should be treated as a frozen benchmark until real SafeBand recordings exist.

### PPG
The PPG work established:
- an E4 BVP reference benchmark using PPG-DaLiA
- a MAX30101-domain reference using PTT
- a walking-specific cross-subject stress experiment

These experiments correctly preserve the distinction between benchmark sensor domains and the target MAX30102 hardware.

### BME680
AIRWISE provides the BME680-native indoor environmental benchmark. V1 is frozen with 24 engineered features and chronological 60/20/20 evaluation within location.

## Gaps found in the repository

### 1. Final PPG/BME680 experiment code was not fully integrated
The repository contained documentation and older ML infrastructure, but the latest final PPG V4.3, walking V5.3 and BME680 V1 code packages were not present in the supplied repository snapshot.

This integration pack brings the final reference experiment code into the repository.

### 2. Activity recognition was only partially operational
The repository had an activity training pipeline, but no final activity model artifact was present in the supplied snapshot.

The integration keeps BITS-2 as the motion-first activity reference and makes its training script more robust.

BITS-2 still does not provide a clean STANDING class. STANDING must therefore remain a runtime vocabulary item rather than being fabricated as a BITS-2 training label.

### 3. Audio existed only as an acoustic-level simulation
The INMP441 interface previously exposed a normalized audio level and a loudness category.

That is insufficient for the stated project concept because acoustic intelligence should be learned from waveform-derived features and labeled audio events.

The integration therefore adds:
- deterministic audio feature extraction
- subject-grouped audio model training
- runtime audio model adapter
- audio-event inference contract
- explicit separation between loudness and semantic event recognition

No audio dataset is invented or bundled. The dataset must be selected and audited before training.

### 4. Fusion was not fully connected to risk
The supplied application calculated sensor fusion and then calculated risk independently.

The integration connects the fusion result to the risk engine as a consistency/confirmation signal.

The implementation deliberately avoids adding the complete fusion score again, which would double-count the same sensor evidence.

Explicit SOS, confirmed fall and critical risk conditions retain priority.

### 5. BME680 runtime interface was missing gas-resistance continuity
AIRWISE V1 uses gas resistance, while the prototype BME680 interface exposed only temperature, humidity and pressure.

The updated interface exposes:
- temperature
- humidity
- pressure
- gas resistance
- heat-stability flag

This creates a closer contract between the benchmark and future hardware acquisition layer.

### 6. Tool robustness
The supplied AIRWISE pipeline initially assumed:

`datasets/raw/AIRWISE/data/indoor/sensor_1min`

while the actual extraction contained:

`datasets/raw/AIRWISE/AIRWISE-v1.0.0/data/indoor/sensor_1min`

The updated AIRWISE tools discover the actual minute-data directory recursively and also accept `--data-dir`.

This behavior is now a repository-wide engineering rule.

## What this integration intentionally does NOT claim

It does not claim that:
- public datasets are equivalent to SafeBand hardware
- PTT/MAX30101 is MAX30102
- E4 BVP is MAX30102
- AIRWISE IAQ labels are personal medical/emergency labels
- audio loudness is a scream/distress detector
- the current fusion rules are clinically validated
- the current Python prototype is TinyML-ready

## Phase 1 target

By the end of Phase 1, the software should provide:

1. Reproducible reference datasets and experiments.
2. Stable sensor-specific feature contracts.
3. Activity/fall reference models.
4. PPG reference models and diagnostics.
5. BME680 reference model and diagnostics.
6. Audio feature/training/runtime infrastructure.
7. Hardware-facing sensor interfaces.
8. A fusion/risk API that can consume modality outputs.
9. A preflight tool that detects broken imports and missing artifacts.
10. Clear documentation separating benchmark evidence from hardware validation.

## Phase 2 target

When the physical prototype is available:

1. Implement actual ESP32-S3 sensor acquisition.
2. Synchronize BNO055, MAX30102, BME680 and INMP441 streams.
3. Record timestamped multimodal SafeBand sessions.
4. Validate every reference model on the actual sensor domain.
5. Build a labeled synchronized safety-event dataset.
6. Evaluate multimodal sensor fusion.
7. Measure false-positive and false-negative behavior.
8. Optimize selected models for ESP32-S3/TinyML.
9. Integrate GPS and EC200U emergency communication.
10. Perform end-to-end system validation.

## Definition of done for Phase 1

Phase 1 is not defined as "99% accuracy on every public dataset."

It is defined as having a reproducible, modular and hardware-ready AI software stack whose individual components have documented evidence, known limitations and stable interfaces.

The decisive validation of SafeBand itself begins when synchronized recordings from the actual wearable become available.
