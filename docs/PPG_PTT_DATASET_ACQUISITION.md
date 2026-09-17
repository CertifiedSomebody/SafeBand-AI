# Pulse Transit Time PPG Dataset — Acquisition Log

## Dataset
- **Name:** Pulse Transit Time PPG Dataset
- **PhysioNet version:** 1.1.0
- **Role:** SafeBand AI second-stage PPG reference dataset.

This dataset is being added because it provides raw multi-wavelength PPG from Maxim MAX30101-family hardware, which is substantially closer to the planned MAX30102 hardware than the Empatica E4 BVP used by PPG-DaLiA.

**Important:** it is not treated as a MAX30102 dataset. Any MAX30101 → MAX30102 transfer is documented as cross-device/domain transfer.

## Acquisition

The complete dataset is being downloaded locally rather than selecting only a few recordings. This preserves the original coverage and allows later SafeBand preprocessing to select subjects/activities.

**Destination:**
`D:\Admin\Downloads\SafeBand-AI-main\SafeBand-AI-main\datasets\raw\PTT`

AWS CLI was installed and verified:

`aws-cli/2.36.47 Python/3.14.6 Windows/11 exe/AMD64`

The public PhysioNet S3 listing was successfully accessed with:

```powershell
aws s3 ls --no-sign-request s3://physionet-open/pulse-transit-time-ppg/1.1.0/
```

The listing confirmed 22 subjects (`s1`–`s22`) and `run`, `sit`, and `walk` recordings, with `.dat`, `.hea`, and `.atr` WFDB files.

## Full download command

Run this from PowerShell:

```powershell
aws s3 sync --no-sign-request `
s3://physionet-open/pulse-transit-time-ppg/1.1.0/ `
"D:\Admin\Downloads\SafeBand-AI-main\SafeBand-AI-main\datasets\raw\PTT"
```

This preserves the repository metadata, checksums, headers, signal files, annotations, and schematic.

## Expected raw structure

```text
datasets/
└── raw/
    └── PTT/
        ├── ANNOTATORS
        ├── LICENSE.txt
        ├── README.txt
        ├── RECORDS
        ├── SHA256SUMS.txt
        ├── device_schematic.png
        ├── s1_run.dat
        ├── s1_run.hea
        ├── s1_run.atr
        ├── s1_sit.dat
        ├── s1_sit.hea
        ├── s1_sit.atr
        ├── s1_walk.dat
        ├── s1_walk.hea
        ├── s1_walk.atr
        └── ... through s22
```

## Initial remote audit

The S3 listing showed:
- 22 subject IDs
- 3 activities: sit, walk, run
- WFDB `.dat` signal files
- WFDB `.hea` headers
- `.atr` annotation files
- `README.txt`, `RECORDS`, `SHA256SUMS.txt`, `LICENSE.txt`, `ANNOTATORS`
- `device_schematic.png`

Individual `.dat` files observed were approximately 6.5–7.4 MB.

## Why this dataset

The PTT dataset is intended to fill the hardware-domain gap left by PPG-DaLiA:

```text
PPG-DaLiA
    ↓
General PPG + motion HR methodology
    ↓
PTT / MAX30101
    ↓
Raw red + infrared + green PPG
    ↓
Pulse-oximeter reference measurements
    ↓
Own MAX30102 recordings
    ↓
Final SafeBand hardware validation
```

The actual SafeBand MAX30102 recordings remain necessary for final hardware-domain validation.

## Audit-before-training rule

Do **not** train immediately after downloading. First audit:

1. Exact WFDB channel names/order.
2. Sampling frequencies and units.
3. MAX30101 red/IR/green channel mapping.
4. IMU channels.
5. ECG/reference channels.
6. SpO2 representation and timing.
7. Whether SpO2 is continuous or sparse.
8. Subject/session/activity boundaries.
9. Missing or unusable recordings.
10. Cross-device implications for MAX30102.

Only after this audit should data be converted into `datasets/processed/ptt_*`.

## Reproducibility

Do not rename or manually reorganize files inside `datasets/raw/PTT`.

Preprocessing, filtering, windowing, feature extraction, and label construction belong under `datasets/processed/` and should be implemented through reproducible scripts.

The exact PhysioNet version (`1.1.0`) and supplied SHA256 checksums should be retained for reproducibility.

## Status

**Acquisition:** in progress  
**Remote access:** verified  
**AWS CLI:** verified  
**Training:** not started  
**Dataset audit:** pending local acquisition

---
SafeBand AI project continuity record.
