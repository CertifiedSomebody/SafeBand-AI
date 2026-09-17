# Applying SafeBand Phase-1 Integration V1

This package is a **merge/update pack**, not a replacement for the entire repository.

## 1. Backup

Commit or copy the current repository first.

```powershell
git status
git add .
git commit -m "chore: checkpoint before phase1 integration"
```

## 2. Copy the package contents

Extract the folder:

```text
SafeBand_Phase1_Integration_V1/
```

Copy its contents into the SafeBand-AI repository root and allow the listed files to be replaced.

Do not delete existing folders that are not present in this package.

## 3. Install dependencies

From the repository root:

```powershell
pip install -r requirements.txt
```

## 4. Run the mandatory preflight

```powershell
python tools\preflight_phase1.py
```

The preflight should end with:

```text
PHASE-1 PREFLIGHT: PASS
```

Missing datasets/models are expected to show as `--` until their corresponding experiment has been acquired/trained.

## 5. BME680

The AIRWISE scripts now discover the actual extracted directory recursively.

```powershell
python tools\audit_airwise.py
python tools\prepare_airwise_bme680.py
python tools\train_airwise_bme680.py
```

Optional explicit path:

```powershell
python tools\audit_airwise.py --data-dir <actual_sensor_1min_directory>
```

## 6. Activity model

If the prepared BITS-2 windows already exist:

```powershell
python tools\train_activity_models.py --input datasets\processed\bits2\activity_windows.csv
```

The BITS-2 model is a reference model. Do not enable it as the final SafeBand AI until hardware-domain validation exists.

## 7. PPG references

The final PPG V4.3 and walking V5.3 experiment code is now included for reproducibility.

These remain reference/stress experiments:

```text
E4 / PPG-DaLiA     → reference
MAX30101 / PTT     → reference
MAX30102           → actual SafeBand target
```

Do not relabel the first two as MAX30102 validation.

## 8. Audio

The INMP441 audio infrastructure is now ready, but **no audio dataset is bundled**.

Do not create a fake training dataset from the dashboard scenarios.

After selecting and auditing a real labeled audio dataset:

```powershell
python tools\train_audio_model.py --input datasets\raw\audio
```

The trainer requires subject/group-disjoint evaluation.

## 9. Important switches

The following remain disabled by default:

```text
AI_MODEL_ENABLED = False
AI_AUDIO_MODEL_ENABLED = False
AI_BME680_MODEL_ENABLED = False
```

This is intentional.

A model artifact existing on disk does not automatically make it a validated SafeBand model.

## 10. What changed architecturally

The update now provides:

```text
Activity model
      │
Fall detector
      │
PPG reference
      │
BME680 reference
      │
Audio-event model
      │
      └────→ modality outputs
                    │
                    ↓
              Sensor Fusion
                    │
                    ↓
               Risk Engine
```

The fusion result is now passed into the risk engine rather than being calculated and discarded.

The fusion score is used as a **consistency/confirmation signal**, not added wholesale to the raw sensor risk score. This avoids double-counting.

## 11. What remains Phase 2

Do not try to finish these with public benchmarks alone:

- synchronized multimodal SafeBand recordings
- actual MAX30102 validation
- actual INMP441 recordings
- final multimodal fusion model
- ESP32-S3 TinyML deployment
- final false-alarm/false-negative validation
- real GPS/EC200U emergency operation

Those require the physical system and synchronized data.
