# PIFv3 Dataset Audit — SafeBand AI

**Audit date:** 2026-09-16  
**Local archive inspected:** `PIFv3.zip`  
**Intended project location:** `datasets/raw/PIFv3/`  
**Observed internal dataset root:** `PIFv3/PIFv3-main/Pif3_Dataset/`

## 1. Audit conclusion

The downloaded PIFv3 archive is structurally intact and contains data for **32 participant folders (PID1–PID32)** plus a `Participants.xlsx` metadata workbook.

The most important finding for SafeBand AI is:

> **The CSVs do not contain raw optical PPG channels such as RED, IR, BVP, or raw MAX30102 samples.**

The main participant CSV contains ECG-related/processed ECG fields, GSR, EMG, accelerometer, gyroscope and orientation-related fields. The companion `_BPM.csv` contains `hr` and `spo2` values plus activity labels.

Therefore, this archive **must not yet be treated as a raw-MAX30102 waveform dataset for training a PPG-to-SpO2 model**. It may still be useful for a supervised/contextual experiment involving the provided physiological outputs, but that would be a different task from learning SpO2 directly from MAX30102 RED/IR waveforms.

This distinction is now recorded as a project constraint.

---

## 2. Archive structure

```text
PIFv3/
└── PIFv3-main/
    └── Pif3_Dataset/
        ├── Participants.xlsx
        ├── PID1/
        │   ├── PID_1.csv
        │   └── PID_1_BPM.csv
        ├── PID2/
        │   ├── PID_2.csv
        │   └── PID_2_BPM.csv
        ├── ...
        └── PID32/
            ├── PID_32.csv
            └── PID_32_BPM.csv
```

Observed archive contents:

- 32 participant directories: PID1–PID32
- 64 CSV files: 2 files per participant
- 1 Excel workbook: `Participants.xlsx`
- No separate raw RED/IR/BVP file was found in the uploaded archive.
- No README or dataset-methodology document was found inside the uploaded archive.

---

## 3. Participant metadata

`Participants.xlsx` contains **88 metadata rows** and 13 columns:

```text
PID
Weight in(kg)
Height
Age
Gender
Medical History
State
Smoking
Drinking
Diabetes
Heart Disease
Obese
Thyroid
```

Only **32 PIDs (1–32)** have corresponding data folders in this archive. The workbook therefore contains **0 additional metadata PIDs without corresponding signal folders in this copy**.

For the 32 participants with signal data:

- Gender: {'Female': 20, 'Male': 12}
- Age range: 8–73 years
- Mean age: 26.09 years
- Median age: 22 years

The workbook also contains participant-specific medical-history and health-status fields. Those fields should not be used as model inputs unless there is a clear scientific justification and an explicit privacy/ethics decision.

Source terminology/spelling is preserved as supplied by the dataset, including entries such as `Himachal Pardesh`, `Occasionaly`, and both `Yes`/`yes` variants.

---

## 4. Main signal CSV

Example: `PID_1.csv`

Observed dimensions:

- 4,575 rows
- 31 columns

Columns:

```text
Datetime
ECGR
AvgECG
ampECG
MAX
GSRR
Resistance
GSR(kohm)
SCL
GSRRMS
GSRMAV
EMGR
EMGRMS
EMGMAV
EMGVAR
EMGSSI
IEMG
AccX
AccY
AccZ
GyroX
GyroY
GyroZ
Aroll
Apitch
Groll
Gpitch
Gy
Combroll
Combpitch
Features
```

Signal groups:

- ECG-related: `ECGR`, `AvgECG`, `ampECG`, `MAX`
- GSR-related: `GSRR`, `Resistance`, `GSR(kohm)`, `SCL`, `GSRRMS`, `GSRMAV`
- EMG-related: `EMGR`, `EMGRMS`, `EMGMAV`, `EMGVAR`, `EMGSSI`, `IEMG`
- Accelerometer: `AccX`, `AccY`, `AccZ`
- Gyroscope: `GyroX`, `GyroY`, `GyroZ`
- Orientation/derived motion: `Aroll`, `Apitch`, `Groll`, `Gpitch`, `Gy`, `Combroll`, `Combpitch`
- Activity label: `Features`

### Raw PPG check

The inspected main CSV has no column explicitly corresponding to:

```text
RED
IR
PPG
BVP
raw optical SpO2 samples
```

The archive-level scan therefore gives us no raw optical waveform input for a MAX30102 waveform model.

---

## 5. Companion `_BPM.csv`

Example: `PID_1_BPM.csv`

Observed dimensions:

- 6,601 rows
- 6 columns

Schema:

```text
Datetime
sys
dia
hr
spo2
features
```

It explicitly provides:

- `sys` — systolic blood pressure
- `dia` — diastolic blood pressure
- `hr` — heart rate
- `spo2` — oxygen saturation value
- `features` — activity/state label

For PID1:

- SpO2 range: 96.9–98.7
- HR range: 67–102 BPM

Across PID1–PID32:

- Total BPM rows: **181,370**
- Total main-signal rows: **151,431**

Some participant files contain physiologically unusual HR values at face value (for example, PID11 reaches 225 BPM and PID6 reaches 10 BPM). These are **data-quality observations**, not automatic grounds for deletion. They need source/methodology validation before any filtering rule is defined.

---

## 6. Signal volume and timing

Across PID1–PID32:

- Main signal rows: **151,431**
- BPM rows: **181,370**
- Participant recordings: **32**

For PID1, the median timestamp spacing is approximately **0.1 s** in both the main and BPM files, with some 0.2 s intervals and occasional larger gaps.

This suggests an approximately 10 Hz released CSV representation, **but the uploaded archive does not contain enough methodology documentation to certify the acquisition sampling rate**.

Do not hard-code a sampling rate into training code until the original dataset documentation confirms it.

---

## 7. Activity labels

Observed activity/state labels include:

```text
Anxiety
Fist
Forward Fall while Sitting
Forward Fall while Standing
Funny
Hand at Rest
Left Fall while Sitting
Left Fall while Standing
Motivate
Relax
Right Fall while Sitting
Right Fall while Standing
Sad
Sitting
Standing
Stress Ball
Walking
```

The BPM files additionally contain `Rest`.

There are also whitespace variants such as:

- `Funny` vs `Funny `
- `Walking` vs `Walking `

The loader should strip leading/trailing whitespace from categorical labels while retaining an audit of the original values.

The dataset therefore covers:

- affective/context states (`Relax`, `Sad`, `Anxiety`, `Motivate`, `Funny`)
- physical activities (`Standing`, `Sitting`, `Walking`, `Fist`, `Stress Ball`)
- fall scenarios (forward/left/right, while sitting/standing)
- resting/hand-rest states.

---

## 8. Missing-data findings

The archive is not perfectly uniform.

PID2 contains substantial missingness:

- 11,354 missing cells in the main CSV
- 3,949 missing cells in the BPM CSV
- 658 missing SpO2 values
- 658 missing HR values

Other participants should also be checked individually before training.

Required preprocessing audit:

1. per-column missingness
2. per-participant missingness
3. timestamp continuity
4. label alignment
5. duplicate timestamps
6. explicit missing-value handling

Missing values must not be silently imputed.

---

## 9. Schema inconsistencies

One concrete column-name inconsistency was found:

- Most `_BPM.csv` files use `Datetime`
- `PID_3_BPM.csv` uses lowercase `datetime`

Label whitespace inconsistencies were also observed:

- `Funny` / `Funny `
- `Walking` / `Walking `

The data loader should normalize column names and categorical whitespace rather than assuming every file is byte-for-byte schema-identical.

---

## 10. What this means for SafeBand AI

### Potentially useful

PIFv3 can support:

- SpO2 distribution analysis across activities
- HR/SpO2 behavior analysis
- contextual multimodal experiments using ECG/GSR/EMG/IMU fields
- validation of data handling around the supplied `spo2` target
- secondary physiological-state experiments

### Not suitable yet for

PIFv3 should **not** currently be designated as the primary dataset for:

> **MAX30102 raw RED/IR waveform → SpO2**

because the uploaded archive does not expose the raw optical waveform needed for that learning problem.

Also, the existence of an `spo2` column **does not by itself prove that it is an independent clinical/reference ground truth**. Its provenance must be confirmed from the original dataset methodology.

---

## 11. Critical project distinction

For SafeBand, the intended physical sensor is MAX30102.

The desired training evidence chain is:

```text
MAX30102 RED + IR raw samples
            ↓
sensor-specific preprocessing
            ↓
PPG representation/features
            ↓
SpO2 target/reference
            ↓
subject-independent evaluation
            ↓
validation on SafeBand's own MAX30102 recordings
```

PIFv3 currently gives us the **target-side `spo2` field**, but the uploaded archive does not give us the **raw optical input side**.

Therefore:

> **Sensor match ≠ raw-signal availability ≠ ground-truth validity.**

All three must be established independently.

---

## 12. Required next step before training

**Do not write the PIFv3 SpO2 training pipeline yet.**

First obtain/verify the original methodology/source documentation and answer:

1. What exact device generated `spo2`?
2. Are raw MAX30102 RED/IR samples available elsewhere?
3. Is `spo2` from an independent reference device or from the sensor/device processing chain?
4. What are the true sampling rates?
5. How are the main CSV and `_BPM.csv` timestamps aligned?
6. What was the recording protocol for each activity?
7. Were all 32 signal participants recorded using the same hardware/configuration?
8. What preprocessing was already applied before CSV release?

Only then should PIFv3 be classified as:

- primary raw-PPG training data,
- reference/target data,
- contextual multimodal data, or
- unsuitable for the intended SpO2 model.

**Current classification from the uploaded archive alone:**

> **PIFv3 = candidate dataset with explicit SpO2 outputs, but raw optical PPG is not present in this archive; sensor provenance and SpO2 ground-truth provenance still require verification.**

---

## 13. Per-participant audit

A machine-readable audit file is provided separately:

`PIFv3_per_participant_audit.csv`

It records:

- main/BPM row counts
- column counts
- missing-cell counts
- missing HR/SpO2 counts
- activity counts
- per-participant HR/SpO2 ranges
- observed median timestamp spacing

---

## 14. Project placement

Keep the dataset locally as:

```text
datasets/
└── raw/
    └── PIFv3/
        └── PIFv3-main/
            └── Pif3_Dataset/
                ├── Participants.xlsx
                ├── PID1/
                ├── ...
                └── PID32/
```

Do not rename internal dataset files yet.

Do not commit the complete raw dataset to GitHub unless its license, size and project policy explicitly allow it.

---

## 15. Status

| Item | Status |
|---|---|
| PIFv3 acquisition | COMPLETE |
| Archive inspection | COMPLETE |
| Participant structure audit | COMPLETE |
| Schema audit | COMPLETE |
| Initial missingness audit | COMPLETE |
| Raw optical PPG availability | **NOT FOUND IN ARCHIVE** |
| SpO2 ground-truth provenance | **NOT YET VERIFIED** |
| SpO2 model training | **BLOCKED PENDING METHODOLOGY AUDIT** |

This audit prevents us from repeating the failure mode of downloading a dataset with a promising sensor/target description and discovering only later that the actual training signal is unavailable.
