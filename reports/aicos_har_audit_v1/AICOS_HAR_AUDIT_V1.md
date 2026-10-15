# AICOS-HAR Forensic Audit V1

- Metadata rows: **5231**
- Metadata columns: `AcquisitionID, Age, Gender, Position_FullName, Fold`
- Raw sensor files found: **18587**
- Sensor files inspected: **18587**

## Metadata unique-value summary
- **AcquisitionID**: 5231 unique non-empty values
- **Age**: 5 unique non-empty values
- **Gender**: 2 unique non-empty values
- **Position_FullName**: 25 unique non-empty values
- **Fold**: 6 unique non-empty values

## Activities

```text
{
  "Running": 287,
  "Walking": 892,
  "Downstairs": 1012,
  "LyingDown": 93,
  "Sitting": 285,
  "Standing": 270,
  "Upstairs": 1032,
  "LiftDown": 360,
  "LiftUp": 351,
  "RampDown": 314,
  "RampUp": 335
}
```

## Fold distribution

```text
{
  "5": 251,
  "test": 1230,
  "2": 890,
  "4": 1833,
  "1": 261,
  "3": 766
}
```

## Raw sensor summary

```text
{
  "acc": {
    "files": 5231,
    "estimated_hz_median_across_files": 397.0371499720486,
    "estimated_hz_min": 49.927450421792095,
    "estimated_hz_max": 838.6475633933693,
    "duration_s_median": 20.160547362,
    "duration_s_min": 3.051971085,
    "duration_s_max": 6051.926817066,
    "non_monotonic_timestamp_files": 0
  },
  "gyro": {
    "files": 4899,
    "estimated_hz_median_across_files": 397.1986374497941,
    "estimated_hz_min": 49.927450421792095,
    "estimated_hz_max": 838.6475633933693,
    "duration_s_median": 19.559463557,
    "duration_s_min": 3.051971085,
    "duration_s_max": 6051.680357105,
    "non_monotonic_timestamp_files": 0
  },
  "mag": {
    "files": 4744,
    "estimated_hz_median_across_files": 99.93304485994383,
    "estimated_hz_min": 10.0,
    "estimated_hz_max": 108.33062506770663,
    "duration_s_median": 18.759968923499997,
    "duration_s_min": 3.051971085,
    "duration_s_max": 6051.594511158,
    "non_monotonic_timestamp_files": 0
  },
  "baro": {
    "files": 3713,
    "estimated_hz_median_across_files": 10.001186840842403,
    "estimated_hz_min": 0.49757658867717974,
    "estimated_hz_max": 33.42344852112103,
    "duration_s_median": 19.599480939,
    "duration_s_min": 2.87981046,
    "duration_s_max": 6051.376310474,
    "non_monotonic_timestamp_files": 0
  }
}
```

## Leakage warnings
- Some metadata acquisitions were not found in the inspected raw hierarchy.
- Some raw acquisitions were not found in metadata.
