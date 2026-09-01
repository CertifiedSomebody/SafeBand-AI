# SAFEBAND AI 🛡️

## Intelligent AI-Based Safety Monitoring System

SAFEBAND AI is a prototype wearable safety-monitoring platform designed to continuously monitor a user's activity, physiological condition, environmental parameters, and location, then use AI-based processing and multi-sensor fusion to identify abnormal or emergency situations.

> **Prototype status:** Software demonstration using simulated sensor data.  
> **Purpose:** Demonstrate the proposed system architecture and capabilities before real hardware and real-world datasets are integrated.

---

## 1. What This Prototype Demonstrates

The prototype demonstrates the complete intended processing flow:

```text
Simulated Sensor Data
        ↓
Activity Recognition
        ↓
Sensor Fusion
        ↓
Risk Assessment
        ↓
Emergency / Safety Alert
        ↓
Cellular + Cloud Communication
        ↓
Real-Time Dashboard
```

The software is therefore more than a static dashboard: the modules form an end-to-end demonstrator of how the final SAFEBAND AI system is intended to operate.

---

## 2. Main Capabilities

- ❤️ Heart-rate monitoring
- 🫁 SpO₂ monitoring
- 🧭 Motion and orientation monitoring
- 🌡️ Environmental monitoring
- 🎙️ Acoustic activity monitoring
- 📍 GPS location tracking
- 🤖 Activity recognition
- 🔗 Multi-sensor fusion
- ⚠️ Risk scoring
- 🚨 Fall/emergency detection
- 🆘 Manual SOS scenario
- 📡 Cellular communication simulation
- ☁️ Cloud synchronization simulation
- 📊 Real-time monitoring dashboard
- 📜 Alert/event history
- 📈 Sensor and risk trends

---

## 3. Sensors Represented

| Sensor | Parameters / Purpose |
|---|---|
| **MAX30102** | Heart rate and SpO₂ |
| **BNO055** | Acceleration, motion and orientation |
| **BME680** | Temperature, humidity and pressure |
| **INMP441** | Audio/acoustic level |
| **GPS** | Latitude and longitude |
| **EC200U** | Cellular communication interface |

At the current stage, these sensors are represented through software simulation. The sensor modules are separated from the AI and dashboard layers so that real hardware drivers can be integrated later.

---

## 4. AI Processing

### Activity Recognition

The activity-recognition layer classifies the user's current state, including:

```text
SITTING
STANDING
WALKING
RUNNING
FALL
UNKNOWN
```

It also provides a confidence value for the detected activity.

### Sensor Fusion

Sensor fusion combines evidence from multiple sources rather than depending on one sensor alone:

```text
Motion
   +
Physiological Data
   +
Environmental Data
   +
Audio
   +
Activity Recognition
        ↓
  Fused Condition
```

### Risk Engine

The risk engine converts the available evidence into a **0–100 safety risk score**.

| Score | Risk Level | System Status |
|---:|---|---|
| 0–29 | LOW | SAFE |
| 30–59 | MODERATE | WARNING |
| 60–79 | HIGH | WARNING |
| 80–100 | CRITICAL | EMERGENCY |

---

## 5. Demonstration Scenarios

The dashboard includes predefined scenarios so the system can be demonstrated reliably without physical hardware.

### `NORMAL`

Normal user condition.

```text
Normal sensors
      ↓
LOW RISK
      ↓
SAFE
```

### `WALKING`

Normal walking activity with corresponding movement and physiological changes.

### `RUNNING`

Higher-intensity activity with increased motion and heart rate.

### `FALL`

Simulates a sudden abnormal motion/orientation event.

```text
Sudden Motion
      +
Abnormal Orientation
      +
Sensor Evidence
      ↓
Fall Detection
      ↓
Emergency Processing
```

### `HIGH_RISK`

Simulates multiple abnormal sensor conditions occurring together.

### `SOS`

Simulates a manual emergency activation by the user.

```text
SOS
 ↓
Emergency Alert
 ↓
Location
 ↓
Cellular Transmission
 ↓
Cloud Synchronization
```

---

## 6. Project Structure

```text
SAFEBAND_AI/
│
├── app.py
├── README.md
│
├── ai/
│   ├── __init__.py
│   ├── activity_recognition.py
│   ├── sensor_fusion.py
│   └── risk_engine.py
│
├── communication/
│   ├── __init__.py
│   ├── cellular.py
│   └── cloud.py
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── dashboard/
│   ├── __init__.py
│   ├── alerts.py
│   ├── charts.py
│   └── ui.py
│
├── data/
│   ├── __init__.py
│   ├── demo_scenarios.py
│   └── simulated_data.py
│
├── sensors/
│   ├── __init__.py
│   ├── bme680.py
│   ├── bno055.py
│   ├── gps.py
│   ├── inmp441.py
│   └── max30102.py
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   └── loggers.py
│
├── assets/
│   └── logo/
│       └── safeband_logo.png
│
└── logs/
    └── safeband_ai.log
```

---

## 7. Module Responsibilities

### `ai/`

Contains the intelligence layer.

- `activity_recognition.py` — activity classification
- `sensor_fusion.py` — combines multi-sensor evidence
- `risk_engine.py` — calculates overall safety risk

### `communication/`

Handles external communication interfaces.

- `cellular.py` — EC200U communication simulation
- `cloud.py` — cloud synchronization simulation

### `config/`

Central project configuration.

- `settings.py` — thresholds, sensor configuration, dashboard settings, demo settings and system configuration

### `dashboard/`

Contains the Streamlit presentation layer.

- `ui.py` — dashboard components
- `charts.py` — Plotly visualizations
- `alerts.py` — safety and emergency alert management

### `data/`

Provides demonstration data.

- `demo_scenarios.py` — predefined scenarios
- `simulated_data.py` — continuously varying simulated sensor readings

### `sensors/`

Provides sensor interfaces.

- `max30102.py`
- `bno055.py`
- `bme680.py`
- `inmp441.py`
- `gps.py`

### `utils/`

Shared utilities.

- `helpers.py`
- `loggers.py`

---

## 8. Software Stack

### Core

- Python 3
- Streamlit
- Plotly
- Pandas

### Intended Embedded Platform

- ESP32-S3

### Intended Hardware

- MAX30102
- BNO055
- BME680
- INMP441
- GPS/GNSS receiver
- Quectel EC200U

---

## 9. Installation

Open a terminal in the project root.

### Optional: create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install streamlit plotly pandas
```

---

## 10. Run the Prototype

From the project root:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in the browser.

---

## 11. Recommended Demo Flow

For a project/proposal presentation, use this sequence.

### 1. Start with `NORMAL`

Show:

- Live sensor readings
- SAFE status
- Low risk score
- Normal activity

### 2. Select `WALKING`

Show:

- Activity recognition changing to walking
- Sensor values changing
- Dashboard updating in real time

### 3. Select `RUNNING`

Show:

- Increased motion
- Increased heart rate
- Activity recognition
- Sensor-fusion response

### 4. Select `FALL`

This is the main emergency demonstration.

Show:

```text
FALL
 ↓
Abnormal Motion
 ↓
Abnormal Orientation
 ↓
Sensor Fusion
 ↓
Risk Engine
 ↓
EMERGENCY
```

Then show the alert, location, cellular status and cloud synchronization.

### 5. Select `SOS`

Demonstrate the independent manual emergency path:

```text
Manual SOS
 ↓
Emergency Alert
 ↓
Location Information
 ↓
EC200U Transmission
 ↓
Cloud Event
```

### 6. Show Charts and History

Scroll down to demonstrate:

- Heart-rate trend
- SpO₂ trend
- Temperature trend
- Motion trend
- Risk-score trend
- Alert/event history

---

## 12. Prototype Architecture vs Final System

### Current prototype

```text
Simulated Sensors
       ↓
Rule-Based Processing
       ↓
Sensor Fusion
       ↓
Risk Engine
       ↓
Simulated Communication
       ↓
Streamlit Dashboard
```

### Intended final system

```text
Real Sensors
       ↓
ESP32-S3
       ↓
AI / TinyML Models
       ↓
Multi-Sensor Fusion
       ↓
Risk Assessment
       ↓
EC200U / Network
       ↓
Cloud Backend
       ↓
Caregiver / Emergency Response
```

The prototype intentionally keeps these layers modular so that simulation components can be replaced progressively with real sensor drivers, trained AI models and production communication services.

---

## 13. Development Roadmap

### Phase 1 — Software Prototype

- [x] Project architecture
- [x] Simulated sensor interfaces
- [x] Demonstration scenarios
- [x] Activity recognition
- [x] Sensor fusion
- [x] Risk engine
- [x] Alert manager
- [x] GPS simulation
- [x] Cellular simulation
- [x] Cloud simulation
- [x] Real-time dashboard
- [x] Charts and event history

### Phase 2 — Hardware Integration

- [ ] ESP32-S3
- [ ] MAX30102
- [ ] BNO055
- [ ] BME680
- [ ] INMP441
- [ ] GPS/GNSS
- [ ] EC200U
- [ ] Battery/power subsystem

### Phase 3 — Real Data & AI

- [ ] Collect real sensor data
- [ ] Build labeled activity dataset
- [ ] Collect fall/emergency-event data
- [ ] Train activity-recognition model
- [ ] Train/validate fall-detection model
- [ ] Evaluate sensor-fusion strategy
- [ ] Optimize models for TinyML/edge execution

### Phase 4 — Communication & Cloud

- [ ] Real cellular communication
- [ ] Production cloud backend
- [ ] Real-time event synchronization
- [ ] Caregiver notification service
- [ ] Emergency location sharing
- [ ] Persistent event database

### Phase 5 — Validation

- [ ] Sensor calibration
- [ ] Hardware testing
- [ ] Battery testing
- [ ] Wearability testing
- [ ] False-positive evaluation
- [ ] False-negative evaluation
- [ ] Field testing
- [ ] End-to-end validation

---

## 14. Prototype Limitations

This version is a **demonstration prototype**, not a production safety or medical device.

The current system uses:

- Simulated sensor readings
- Predefined demonstration scenarios
- Rule-based AI/risk logic
- Simulated cellular communication
- Simulated cloud storage

Before real-world deployment, the system will require real sensor data, hardware validation, model training, calibration, reliability testing, false-alert analysis, power optimization and field validation.

---

## 15. Future Integration Concept

The important design principle is that the application is **hardware-agnostic at the processing layer**.

For example:

```text
Current:
MAX30102 Interface
        ↓
Simulated Reading
        ↓
AI Pipeline
```

can later become:

```text
Real MAX30102
        ↓
ESP32-S3
        ↓
Real Sensor Reading
        ↓
Same AI Pipeline
```

The same principle applies to the BNO055, BME680, INMP441, GPS and EC200U modules.

---

## 16. Project Vision

SAFEBAND AI is intended to evolve from this prototype into an intelligent wearable safety platform capable of:

```text
SENSE
  ↓
UNDERSTAND
  ↓
FUSE
  ↓
ASSESS
  ↓
ALERT
  ↓
RESPOND
```

The long-term goal is to move from simple threshold-based monitoring toward an adaptive AI-driven safety system using real-world data and edge intelligence.

---

## 17. Prototype Status

```text
┌─────────────────────────────────────────┐
│          SAFEBAND AI PROTOTYPE          │
├─────────────────────────────────────────┤
│ Sensor Simulation          ✓            │
│ Activity Recognition       ✓            │
│ Sensor Fusion              ✓            │
│ Risk Assessment            ✓            │
│ Emergency Detection        ✓            │
│ GPS Simulation             ✓            │
│ Cellular Simulation        ✓            │
│ Cloud Simulation           ✓            │
│ Dashboard                  ✓            │
│ Real Hardware              → Next Phase │
│ Real Dataset               → Next Phase │
│ Trained TinyML Model       → Next Phase │
└─────────────────────────────────────────┘
```

**SAFEBAND AI — Intelligent Safety Monitoring, Anywhere.**
