# SafeBand AI — Phase 1 Master Roadmap

## Current objective

Prepare the complete software/AI layer before hardware integration.

The final system is:

```text
Sensors
   ↓
Sensor-specific processing
   ↓
Activity / physiological / environmental / audio intelligence
   ↓
Sensor Fusion
   ↓
Risk Engine
   ↓
SAFE / WARNING / EMERGENCY
   ↓
GPS + EC200U + caregiver/cloud
```

## Work packages

### WP1 — Motion

**Inputs:** BNO055 / accelerometer  
**Outputs:**
- activity
- fall probability
- motion features
- confidence

Status: reference fall branch established; activity pipeline ready.

### WP2 — PPG

**Input:** MAX30102  
**Outputs:**
- HR estimate
- physiological signal quality
- future SpO2 estimate/quality

Status: E4 and MAX30101 references established. Actual MAX30102 validation remains Phase 2.

### WP3 — BME680

**Input:** BME680  
**Outputs:**
- temperature
- humidity
- pressure
- gas resistance
- environmental/IAQ state
- confidence

Status: AIRWISE V1 frozen.

### WP4 — Audio

**Input:** INMP441  
**Outputs:**
- signal quality
- acoustic features
- optional semantic audio event
- confidence

Status: infrastructure added. Dataset selection and training remain to be completed.

### WP5 — Activity recognition

The first learned activity reference uses motion only.

Required runtime vocabulary:

```text
RESTING
SITTING
STANDING
WALKING
RUNNING
UNKNOWN
```

FALL is a separate emergency-event output.

BITS-2 does not provide a clean STANDING class. Do not manufacture one.

### WP6 — Fusion

Fusion receives modality outputs rather than raw sensor-specific business logic.

Conceptual input:

```json
{
  "activity": "...",
  "activity_confidence": 0.0,
  "fall_probability": 0.0,
  "heart_rate": 0.0,
  "ppg_quality": 0.0,
  "spo2": 0.0,
  "iaq_class": "...",
  "audio_event": "...",
  "audio_confidence": 0.0,
  "timestamp": "..."
}
```

The final fusion model must be validated using synchronized SafeBand data.

### WP7 — Risk

The risk engine converts corroborated evidence into a safety state.

Manual SOS is always an explicit input.

### WP8 — Edge deployment

After hardware validation:
- measure latency
- measure memory
- measure flash
- measure power
- quantize where appropriate
- deploy selected models to ESP32-S3

## Immediate next step

The immediate remaining Phase-1 AI branch is **audio**, while the activity benchmark can be trained/validated in parallel if the BITS-2 prepared data is available.

Do not begin final fusion-model training until synchronized multimodal SafeBand data exists.

## Stop conditions

Freeze a branch when:
- the experiment is reproducible
- train/validation/test separation is defensible
- the primary metrics are documented
- limitations are documented
- further benchmark tuning is unlikely to change hardware decisions

Then move toward hardware-domain validation.
