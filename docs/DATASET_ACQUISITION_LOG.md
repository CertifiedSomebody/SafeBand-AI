# SafeBand AI — Dataset Acquisition & Decision Log

**Purpose:** Preserve exactly which external datasets/papers were selected, what was actually acquired, why it was selected, and what it must *not* be used to claim.

**Project:** SafeBand AI — An Intelligent Safety Band Using Sensor Fusion and Artificial Intelligence

---

## 1. Current acquisition status

| Sensor / branch | Resource | Status | Intended role |
|---|---|---|---|
| PPG / MAX30102 | Sensors 2026 paper, DOI `10.3390/s26082487` | **Paper acquired; dataset NOT acquired** | Reference for actual MAX30102 PPG acquisition and HRV processing |
| BME680 | **AIRWISE v1.0.0** | **Dataset acquired** | Primary BME680-native environmental AI reference dataset |
| IMU / fall | BITS2 | Already part of project | Existing fall-detection branch |
| PPG reference | PPG-DaLiA / E4 | Already part of project | Cross-sensor PPG benchmark only; not MAX30102-equivalent |

---

# 2. PPG / MAX30102 — important correction

## Resource inspected

**Paper:** *Comparative Assessment of PPG-Derived HRV Using MAX30102 Sensor and Analog Circuitry with ADS1115 ADC*

**Journal:** Sensors, 2026, 26, 2487

**DOI:** `10.3390/s26082487`

The paper explicitly describes a MAX30102-based PPG acquisition system. It uses an ESP32-S3 and MAX30102, with simultaneous RED and IR acquisition. The experimental MAX30102 sampling rate was **200 samples/s** for both IR and RED channels.

The paper's dataset description says the recordings contain synchronized MAX30102 RED (660 nm), IR (880 nm), and ADS1115 IR data in CSV format.

## Critical dataset availability finding

The paper **does not provide a public downloadable dataset**.

Its Data Availability Statement says:

> "The data presented in this study are available upon request from the corresponding author."

Therefore:

**We did NOT download or acquire the paper's experimental dataset.**

This is why the previously supplied link led to the paper rather than a dataset download.

### Why we keep the paper

Even without the raw dataset, it is highly useful as a **hardware-domain reference** because it documents:

- actual MAX30102 acquisition;
- RED = 660 nm;
- IR = 880 nm;
- 200 sps acquisition in the reported experiments;
- IIR smoothing;
- DC removal using a 0.1 Hz high-pass filter;
- 0.1–6 Hz cardiac band-pass filtering;
- peak/beat detection;
- IBI extraction;
- HR and HRV calculation;
- motion/contact-related waveform distortion;
- signal-quality considerations.

The experimental setup used the MAX30102 on an **index finger**, not the final SafeBand wrist placement.

## What we must NOT claim

This paper is **not a SafeBand training dataset**.

It does not give us a large public MAX30102 training corpus, and its experiments are controlled finger measurements. Therefore:

- Do not train a SafeBand HR model from this paper's tables.
- Do not describe the paper as a downloadable dataset.
- Do not claim that the final wrist-worn MAX30102 model is validated by this paper.
- Use it to guide preprocessing, acquisition parameters, and methodological comparison.

## PPG next step

Before searching for another PPG dataset, inspect the existing **BITS2 MAX30102 channels** already present in the SafeBand project.

The desired future hierarchy is:

1. Actual MAX30102 data already available in BITS2.
2. Another genuinely public MAX30102 raw PPG dataset with usable HR/HRV ground truth, if one can be verified.
3. Real SafeBand MAX30102 recordings once the hardware is available.
4. PPG-DaLiA/E4 remains a cross-sensor benchmark/reference, not a MAX30102 substitute.

---

# 3. BME680 — AIRWISE v1.0.0

## Dataset identity

**Title:** AIRWISE: Environmental Sensor and LLM Query Dataset for Edge AI-Based Air-Quality and Thermal-Comfort Analytics

**Version:** 1.0.0

**Release date:** 2026-07-08

**Dataset DOI:** `10.5281/zenodo.20783874`

**License:** CC BY 4.0 for the authors' original contributions; third-party observations have additional provider terms described by the dataset.

**Associated publication:** *Enabling Cloud-Level Accuracy in Edge AI through IoT Data Preprocessing*

## Acquisition

**Downloaded file:** `AIRWISE-v1.0.0.zip`

**Local acquisition filename:** `AIRWISE-v1.0.0.zip`

**Archive size:** 47,457,908 bytes

**SHA-256:**
`7f4caad33c2d3474f38d07885c767fa3314dec6516de54abffb240762b5c2b40`

The archive contains **56 files** including documentation, metadata, checksums, validation code, dataset-generation scripts, indoor BME680 data, outdoor reference data, standards, and LLM query datasets.

---

# 4. Why AIRWISE was selected

AIRWISE is a substantially better SafeBand BME680 starting point than a scent-classification-only dataset.

The indoor component was collected using **Raspberry Pi nodes with Bosch BME680 sensors** in three real indoor micro-environments:

- office
- kitchen
- hallway

Location: **Tampere, Finland**

Collection period: **2025-11-14 to 2025-12-18**

The indoor sensor data are provided at **1-minute resolution**.

This is a direct sensor match at the BME680 hardware level and gives environmental time-series data rather than only a toy classification task.

---

# 5. AIRWISE indoor BME680 data

The package README reports:

- more than **146,000** minute-averaged indoor readings across the three environments;
- approximately **48.8k rows per environment**;
- BME680 temperature;
- relative humidity;
- pressure;
- gas resistance;
- an IAQ proxy;
- z-score anomaly detectors;
- IAQ classes;
- heat-stability information;
- anomaly/limit annotations.

The primary 1-minute files are:

```text
data/indoor/sensor_1min/
├── office_minute_averages.csv
├── kitchen_minute_averages.csv
└── hallway_minute_averages.csv
```

Reported row counts:

```text
office   48,860
kitchen  48,826
hallway  48,852
----------------
total   146,538
```

---

# 6. AIRWISE BME680 columns

The 1-minute indoor files use a wide format with 16 columns:

```text
Date
location
temperature_c
temperature_c_zscore
pressure_hpa
pressure_hpa_zscore
humidity_rh
humidity_rh_zscore
gas_resistance_ohms
gas_resistance_ohms_zscore
heat_stable
IAQ_proxy
IAQ_proxy_zscore
IAQ_class
samples_per_hour
Z-score
```

### Raw/primary sensor quantities

| Column | Meaning |
|---|---|
| `temperature_c` | Air temperature in °C |
| `pressure_hpa` | Barometric pressure in hPa |
| `humidity_rh` | Relative humidity in % |
| `gas_resistance_ohms` | BME680 gas-sensor resistance in ohms |
| `heat_stable` | BME680 gas-heater stability/validity flag |

### Derived quantities

| Column | Meaning |
|---|---|
| `IAQ_proxy` | Indoor air-quality proxy derived from gas resistance and humidity |
| `IAQ_class` | Categorical IAQ state |
| `*_zscore` | Standardized anomaly-detection features |
| `Z-score` | Largest absolute monitored z-score for the row |
| `samples_per_hour` | Sample-rate information projected from the aggregation window |

---

# 7. Important distinction: raw sensor vs derived labels

AIRWISE is **BME680-native**, but its main indoor files are **minute-averaged and processed**, rather than raw high-frequency BME680 register streams.

Therefore we should treat the dataset as:

```text
Actual BME680 measurements
        ↓
1-minute aggregation
        ↓
derived IAQ / anomaly annotations
```

This makes it excellent for environmental-context modeling, anomaly detection, and threshold-aware reasoning.

It does **not** automatically provide a SafeBand-specific physiological or emergency label.

---

# 8. How AIRWISE fits SafeBand

The intended SafeBand BME680 branch should not be forced into the dataset's exact original problem.

The useful conceptual mapping is:

```text
BME680
 ├── Temperature
 ├── Humidity
 ├── Pressure
 └── Gas resistance
          ↓
 Environmental state / anomaly features
          ↓
 Sensor-fusion feature vector
          ↓
 Risk Engine
```

Possible SafeBand uses to investigate:

- environmental-condition classification;
- abnormal environmental-state detection;
- context features for the final risk engine;
- temperature/humidity stress context;
- air-quality context;
- sensor-quality/anomaly indicators.

The exact final task must be defined **after inspecting the data distribution and the actual SafeBand use case**.

---

# 9. AIRWISE also contains non-BME680 data

The archive contains an outdoor component for:

- Helsinki
- Katowice
- Warsaw

with hourly 2023 air-quality and meteorological data.

It also contains:

- reference standards;
- processed long-format tables;
- LLM query datasets;
- source-provider files;
- dataset-generation scripts.

For the first SafeBand BME680 experiment, we should **not automatically mix all outdoor datasets with the BME680 indoor measurements**.

The first experiment should stay focused on the actual BME680 indoor component.

---

# 10. Reproducibility assets retained

AIRWISE v1.0.0 includes:

```text
README.md
MANIFEST.md
CITATION.cff
CHANGELOG.md
LICENSE.txt
THIRD_PARTY_NOTICES.md
checksums.sha256
Airwise.json
validate_package.py
requirements.txt
scripts/
```

This is useful for SafeBand because the dataset does not just provide CSVs; it also provides documentation, provenance, validation and preprocessing/reconstruction scripts.

---

# 11. Recommended SafeBand folder placement

After extracting the downloaded archive:

```text
SafeBand-AI/
└── datasets/
    └── raw/
        └── AIRWISE/
            ├── README.md
            ├── Airwise.json
            ├── data/
            ├── scripts/
            └── ...
```

**Do not rename individual files inside the dataset yet.**

Preserve the original AIRWISE structure for reproducibility.

When we create SafeBand-specific processed data, use a separate path:

```text
datasets/
├── raw/
│   └── AIRWISE/
└── processed/
    └── airwise_bme680/
```

---

# 12. Current dataset decision

## KEEP / USE

### BME680
**AIRWISE v1.0.0**

Reason:
- actual Bosch BME680;
- real environmental deployment;
- three indoor micro-environments;
- long time span;
- minute-level measurements;
- temperature + humidity + pressure + gas resistance;
- anomaly/IAQ annotations;
- reproducible processing assets.

### PPG
**MAX30102 paper: retain as methodological reference**

Reason:
- confirms real MAX30102 acquisition methodology;
- documents RED/IR channels and sampling;
- documents practical PPG filtering and HRV extraction;
- but **dataset is not publicly downloadable**.

### PPG
**BITS2: retain as an existing actual-MAX30102 dataset**

Reason:
- already part of the SafeBand project;
- actual MAX30102 hardware is part of the BITS2 wearable platform;
- especially relevant because BITS2 also includes motion sensing.

---

# 13. Current status after this acquisition

```text
IMU / Fall
    └── BITS2 + SisFall
        └── Mature trained branch

PPG / HR
    ├── PPG-DaLiA / E4
    │   └── Reference benchmark
    ├── BITS2 / MAX30102
    │   └── Existing hardware-domain data to inspect
    └── MAX30102 2026 paper
        └── Methodology reference; dataset unavailable publicly

BME680
    └── AIRWISE v1.0.0
        └── Newly acquired primary BME680 dataset

Final Sensor Fusion
    └── Not trained/finalized yet
```

---

# 14. Next experiment — DO NOT train yet

Before building the BME680 model:

1. Verify the extracted AIRWISE directory.
2. Inspect all three indoor BME680 CSV schemas.
3. Check missing values.
4. Check sampling/aggregation consistency.
5. Examine class distribution of `IAQ_class`.
6. Examine gas-resistance, temperature and humidity distributions.
7. Determine whether the anomaly labels are suitable for a supervised SafeBand task.
8. Check whether room/location leakage could occur.
9. Define a leakage-safe train/validation/test split.
10. Only then build the first BME680 baseline.

**Decision rule:** sensor compatibility is confirmed, but task compatibility must still be experimentally established.

---

## Source notes

The MAX30102 paper was inspected directly from the supplied PDF.

The AIRWISE information in this document was inspected directly from the supplied `AIRWISE-v1.0.0.zip`, including its README, metadata JSON, dataset manifest, and actual CSV schema/sample.

This document intentionally records the PPG dataset-access limitation so that a future chat does not mistakenly assume that the MAX30102 paper supplied a downloadable dataset.
