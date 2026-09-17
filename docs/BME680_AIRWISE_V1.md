# SafeBand AI — BME680 / AIRWISE V1

## Purpose
First supervised environmental-state model for the SafeBand BME680 branch.

AIRWISE v1.0.0 is BME680-native and contains 1-minute indoor measurements from office, kitchen and hallway environments. The primary quantities are temperature, pressure, relative humidity and gas resistance, with an `IAQ_class` categorical state and related derived fields.

The source dataset is processed/minute-averaged rather than a raw high-frequency BME680 register stream. Therefore this experiment is an environmental-context benchmark, not yet proof of real-time ESP32/BME680 performance.

## First task
Predict `IAQ_class` from BME680 sensor measurements and SafeBand-derived temporal features.

Important: `IAQ_proxy` and source z-score/annotation columns are deliberately excluded from model inputs because they are derived quantities and could create target leakage.

## Split
60/20/20 chronological split independently inside each location:
- first 60%: train
- next 20%: validation
- final 20%: untouched test

This avoids random mixing of adjacent time points and gives a meaningful future-time test.

## Models
- Logistic Regression
- Random Forest
- Extra Trees
- HistGradientBoosting

Model selection uses validation balanced accuracy. The selected model is refit on train+validation, then evaluated once on the held-out test.

## Features
Raw sensor values:
- temperature_c
- pressure_hpa
- humidity_rh
- gas_resistance_ohms

Derived:
- log gas resistance
- temperature/humidity interaction
- first difference
- percentage change
- 5-minute rolling mean/std

Temporal rolling features are computed separately per location.

## What this model does NOT claim
- It does not establish clinical safety.
- It does not establish a universal air-quality index.
- It does not prove generalization to outdoor environments.
- It does not prove performance on a physical ESP32/BME680 until target-hardware data are collected.

## Commands
Run from SafeBand repository root.

```powershell
python tools\audit_airwise.py
python tools\prepare_airwise_bme680.py
python tools\train_airwise_bme680.py
```

Prediction:

```powershell
python tools\predict_airwise_bme680.py --csv path\to\new_data.csv
```
