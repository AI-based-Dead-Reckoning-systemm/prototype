# Smartphone-Based Intelligent Dead Reckoning (IDR) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SIH Problem Statement](https://img.shields.io/badge/SIH%202024-PS%2026168%20(ISRO)-orange.svg)](https://www.sih.gov.in/)

> **Smart India Hackathon (SIH) Problem Statement 26168 (ISRO)**:  
> *AI/ML-Powered Smartphone Inertial Navigation & Error-State Kalman Filter (ES-EKF) Fusion for Seamless Urban & GNSS-Denied Vehicle Localization.*

---

## 1. System Architecture & End-to-End Pipeline

The system is organized into a modular monorepo ensuring interoperability across the three pipeline stages:

```
                                  SMARTPHONE SENSORS (10 Hz)
                    [ Tri-Axial Accel, Gyro, Mag, Gravity, GNSS Doppler ]
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │      1. Sensing & Preprocessing           │
                        │      (preprocessing/)                     │
                        │  - Sensor Standardization & Calibration   │
                        │  - Zero Velocity Detection (ZUPT)         │
                        │  - Fixed Boresight Vehicle Alignment      │
                        │  - 5-Class Maneuver Taxonomy Classifier   │
                        └─────────────────────┬─────────────────────┘
                                              │
                           AlignedSample Stream (10 Hz)
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │        2. AI Speed Estimator              │
                        │        (speed_estimation/)                │
                        │  - 2.0s Sliding Window Feature Extraction │
                        │  - Orientation-Invariant Road Baseline    │
                        │  - Random Forest Ensemble Uncertainty     │
                        │  - Physics Bounds & Fallback Dispatcher   │
                        └─────────────────────┬─────────────────────┘
                                              │
                         SpeedEstimate with Variance (10 Hz)
                                              │
                                              ▼
                        ┌───────────────────────────────────────────┐
                        │        3. EKF Fusion Core & Output        │
                        │        (ekf_fusion/)                      │
                        │  - 15-State Error-State Kalman Filter     │
                        │  - Continuous R_k Measurement Covariance  │
                        │  - Soft-GNSS Continuous Trust Weighting   │
                        │  - ZUPT & NHC Kinematic Velocity Updates  │
                        └─────────────────────┬─────────────────────┘
                                              │
                                              ▼
                                 IDRFusedEstimate Stream
                        [ Lat, Lon, Alt, Velocity, Heading, 1σ Uncertainty ]
```

---

## 2. Monorepo Directory Layout

```
sih-idr-26168/
├── schemas.py                 # Single source of truth data contracts (RawSample, AlignedSample, SpeedEstimate, IDRFusedEstimate)
├── validate_contracts.py      # Monorepo CI verification script validating all contracts against real outputs
├── requirements.txt           # Unified dependency specifications
├── README.md                  # Project overview & architecture guide
│
├── preprocessing/             # Track 1: Sensing, calibration, alignment & maneuver classification
│   ├── src/                   # Pipeline implementations (alignment, zupt, maneuver, loader)
│   ├── output/                # Standardized RawSample_*.csv and AlignedSample_*.csv outputs
│   ├── plots/                 # Alignment and ZUPT validation figures
│   ├── run_demo.py            # Quickstart runner for raw sensor preprocessing
│   └── README.md              # Detailed preprocessing documentation & calibration constants
│
├── speed_estimation/          # Track 2: ML speed estimation, ensemble variance & physics bounds
│   ├── src/                   # Feature extraction, RF model, physics bounds, fallback dispatcher
│   ├── models/                # Production model artifact (speed_estimator_rf.pkl, joblib compressed)
│   ├── output/                # SpeedEstimates_*.csv outputs & multi-trip benchmark evaluations
│   ├── plots/                 # Diagnostic performance plots across trips
│   ├── train_and_evaluate.py  # Full multi-trip training & evaluation runner
│   ├── run_demo.py            # Standalone inference demo runner
│   └── README.md              # Model evolution, multi-trip benchmarks & EKF integration spec
│
└── ekf_fusion/                # Track 3: Error-State Kalman Filter Fusion Core (Person C)
    └── README.md              # EKF implementation specifications, R_k noise formulas & interfaces
```

---

## 3. Data Contracts & Module Handoff Matrix

All data crossing module boundaries is governed by canonical Python dataclasses in [`schemas.py`](./schemas.py):

| Contract | Producing Module | Consuming Module(s) | Validation Status | Key Architectural Guarantees |
| :--- | :--- | :--- | :---: | :--- |
| **`RawSample`** | `preprocessing/` | Preprocessing, Logs | **Validated** (5 Trips) | Separate named tri-axial IMU/Mag/Gravity fields; `gps_` prefix matching hardware logs. |
| **`AlignedSample`** | `preprocessing/` | `speed_estimation/`, `ekf_fusion/` | **Validated** (4 Trips) | 5-class maneuver taxonomy (`gentle_curve` flagged as lower confidence $F_1 \approx 0.47-0.50$); vehicle-frame named accelerations (`accel_vehicle_fwd`, `lat`, `up`). |
| **`SpeedEstimate`** | `speed_estimation/` | `ekf_fusion/` | **Validated** (5 Trips) | Continuous `speed_variance` $\sigma^2$ for filter measurement noise $R_k = \max(\sigma^2, 0.25) + 0.50$; continuous `speed_confidence`; `active_source` tag. |
| **`IDRFusedEstimate`**| `ekf_fusion/` | UI, Trajectory Engine | **Schema self-test only**<br>*(no EKF implementation exists yet to validate against)* | Soft continuous GNSS trust weight ($\gamma_k \in [0.0, 1.0]$); $1\sigma$ state uncertainties (`pos_uncertainty_m`, `vel_uncertainty_mps`, `heading_uncertainty_deg`). |
| **`GroundTruthSample`**| Dataset Loader | Evaluation Suite | **Validated** | Reference CAN-bus vehicle speed, steering, yaw rate, and longitudinal/lateral accelerations. |

---

## 4. Benchmark Highlights Across Real-World Driving

Validated against **194,903 synchronized CAN-bus ground truth samples (5.4 hours)** from the IO-VNBD dataset:

* **Sensing & Preprocessing**:
  * Vehicle Yaw Rate Correlation ($r$): **$+0.9348$** (Trip S1), **$+0.9490$** (Trip S3c)
  * Stationary Detection (ZUPT): **$90.93\%$ Precision**, **$0.8005$ F1 Score**
* **AI Speed Estimator**:
  * In-Distribution Training (Trip S1 / Trip M): **$\text{MAE} = 1.086 - 1.244\text{ m/s}$**, $r = 0.924 - 0.964$
  * Clean Unseen Validation (Trip S2): **$\text{MAE} = 3.897\text{ m/s}$** (Urban $5-15\text{ m/s}$: **$2.927\text{ m/s}$**)
  * Swivel-Corrupted Validation (Trip S3c): **$\text{MAE} = 5.517\text{ m/s}$** (**$89.3\%$ ML usage**, eliminating dead-reckoning drift)
  * Completely Untouched Blind Trips (S3b, S3a, S4, Vfa01): **$\text{MAE} = 2.167 - 4.640\text{ m/s}$** (**$94.9\% - 97.7\%$ ML usage**)

---

## 5. Quickstart & Verification

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Validate All Module Contracts
```bash
python validate_contracts.py
```

### 3. Run Preprocessing Demo
```bash
cd preprocessing
python run_demo.py
cd ..
```

### 4. Run AI Speed Estimator Inference Demo
```bash
cd speed_estimation
python run_demo.py --input ../preprocessing/output/AlignedSample_S1.csv --output output/SpeedEstimates_demo.csv
cd ..
```
