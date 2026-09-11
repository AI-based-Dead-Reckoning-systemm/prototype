# Sensing & Preprocessing Module — Intelligent Dead Reckoning (IDR)
> **Smart India Hackathon (SIH) — Problem Statement 26168 (ISRO)**  
> **Module Owner:** Sensing & Preprocessing Subsystem  
> **Status:** Fully Validated & Ground-Truth Benchmarked against IO-VNBD CAN-Bus Dataset  
> **Target Audience:** Downstream Teammates (AI Speed Estimator, ES-EKF Fusion, Map-Matching)

---

## 1. Module Overview & Responsibilities

This repository contains the **Sensing & Preprocessing** stage of the smartphone-based Intelligent Dead Reckoning (IDR) system. It transforms raw smartphone IMU and GNSS sensor streams into calibrated, coordinate-aligned, vehicle-frame signals with stationary detection (ZUPT) and deterministic maneuver states for downstream consumption.

```
RAW PHONE IMU/GNSS (Android Body Frame)
               │
               ▼
┌────────────────────────────────────────────────────────┐
│  1. Ingestion & SI Normalization (src/loader.py)       │
├────────────────────────────────────────────────────────┤
│  2. Stationary / ZUPT Engine (src/zupt.py)             │
├────────────────────────────────────────────────────────┤
│  3. Fixed Phone-to-Vehicle Alignment (src/alignment.py)│
├────────────────────────────────────────────────────────┤
│  4. Rule-Based Maneuver Classifier (src/maneuver.py)   │
└────────────────────────────────────────────────────────┘
               │
               ▼
STANDARDIZED VEHICLE-FRAME CONTRACT (output/AlignedSample_*.csv)
       ├── AI Speed Estimator
       ├── Error-State Kalman Filter (ES-EKF Fusion)
       └── Map-Matching & Track Reconstruction
```

---

## 2. Coordinate Conventions & Fixed Alignment Matrix

```
PHONE BODY FRAME (Android Native)             VEHICLE BODY FRAME (ISO 8855 / Robotics)
        +Y (top of phone)                            +Z_v (Up, Gravity Removed)
         ▲                                            ▲
         │                                            │
         │  ┌─────────┐                               │   ▲ +X_v (Forward / Long.)
         │  │ Screen  │                               │  /
         └──┼────────► +X (right)                     └──┼────────► +Y_v (Left / Lat.)
           /                                            /
          ▼ +Z (out of screen)                         /
```

### Fixed Mount Assumption & Verification:
> **Assumes dashboard-mounted phone in this axis convention; verified against IO-VNBD CAN ground truth.**

In the vehicle mount setup (smartphone held in a rigid landscape dashboard cradle):
- **Yaw Rate ($+Z_v$):** Directly routed from **Phone `gyro_y`** (Column index 16, $r = \mathbf{+0.9348}$ on Trip S1, $r = \mathbf{+0.9490}$ on Trip S3c).
- **Pitch Rate ($+Y_v$):** Directly routed from **Phone `gyro_x`** (Column index 15).
- **Roll Rate ($+X_v$):** Directly routed from **Phone `gyro_z`** (Column index 17).
- **Vertical Acceleration ($+Z_v$):** Measured on **Phone `accel_z`** minus standard gravity ($9.80665\text{ m/s}^2$).
- **Horizontal Accelerations ($+X_v, +Y_v$):** Rotated from phone $(a_x, a_y)$ by the fixed cradle boresight angle $\psi = 316.0^\circ$ ($\approx -44.0^\circ$):

$$\mathbf{R}_{p \to v} = \begin{bmatrix} +0.7193 & -0.6947 & 0.0000 \\ +0.6947 & +0.7193 & 0.0000 \\ 0.0000 & 0.0000 & 1.0000 \end{bmatrix}$$

---

## 3. Output Data Contracts

Standardized CSV files are generated into `output/` for downstream integration:

### Contract 1: `RawSample` (`output/RawSample_*.csv`)
| Column Name | Type | Unit | Description |
| :--- | :---: | :---: | :--- |
| `timestamp_s` | `float64` | $\text{s}$ | Elapsed monotonic time from trip start |
| `accel_x`, `accel_y`, `accel_z` | `float64` | $\text{m/s}^2$ | Tri-axial raw phone acceleration (including gravity) |
| `gyro_x`, `gyro_y`, `gyro_z` | `float64` | $\text{rad/s}$ | Tri-axial raw phone angular velocity |
| `mag_x`, `mag_y`, `mag_z` | `float64` | $\mu\text{T}$ | Tri-axial magnetic field strength |
| `gravity_x`, `gravity_y`, `gravity_z` | `float64` | $\text{m/s}^2$ | Android OS estimated gravity vector |
| `gps_lat`, `gps_lon`, `gps_alt` | `float64` | $\text{deg}, \text{m}$ | GNSS coordinates (WGS84) |
| `gps_speed_mps` | `float64` | $\text{m/s}$ | GPS ground speed |
| `gps_accuracy_m` | `float64` | $\text{m}$ | GPS horizontal error radius ($1\sigma$) |
| `gps_heading_deg` | `float64` | $\text{deg}$ | GPS Course-over-Ground |
| `gps_satellites` | `int64` | $\text{count}$ | Satellites in view/solution |

### Contract 2: `AlignedSample` (`output/AlignedSample_*.csv`)
*Primary input for AI Speed Estimator and Error-State EKF Fusion Core.*
| Column Name | Type | Unit | Description |
| :--- | :---: | :---: | :--- |
| `timestamp_s` | `float64` | $\text{s}$ | Monotonic timestamp sampled at 10 Hz |
| `accel_vehicle_fwd` | `float64` | $\text{m/s}^2$ | **Vehicle forward acceleration** (+ = acceleration, - = braking) |
| `accel_vehicle_lat` | `float64` | $\text{m/s}^2$ | **Vehicle lateral acceleration** (+ = left turn centripetal accel) |
| `accel_vehicle_up` | `float64` | $\text{m/s}^2$ | **Vehicle vertical acceleration** (gravity subtracted, $\approx 0$) |
| `gyro_vehicle_yaw` | `float64` | $\text{rad/s}$ | **Vehicle yaw rate** (+ = turning left, right-hand rule) |
| `gyro_vehicle_pitch` | `float64` | $\text{rad/s}$ | Vehicle pitch rate |
| `gyro_vehicle_roll` | `float64` | $\text{rad/s}$ | Vehicle roll rate |
| `zupt_flag` | `bool` | `True/False` | **Zero Velocity Update flag** (True = vehicle at complete stop) |
| `maneuver_state` | `string` | categorical | `{stationary, straight, gentle_curve, sharp_turn, reversal}` |
| `gps_lat`, `gps_lon`, `gps_alt` | `float64` | $\text{deg}, \text{m}$ | GNSS coordinates (NaN during tunnel / blackout outage) |
| `gps_speed_mps` | `float64` | $\text{m/s}$ | GNSS ground speed |
| `gps_accuracy_m` | `float64` | $\text{m}$ | GNSS accuracy estimate |
| `gps_heading_deg` | `float64` | $\text{deg}$ | GNSS course heading |
| `gps_satellites` | `int64` | $\text{count}$ | GNSS satellite count |

---

## 4. Final Locked Calibration & Classification Thresholds

### ZUPT Stationary Detector (`src/zupt.py`):
- **Sliding Window Size:** $5\text{ samples}$ ($0.5\text{s}$ moving window at $10\text{Hz}$)
- **`accel_var_thresh`:** $0.015\text{ m}^2/\text{s}^4$ (max variance of acceleration magnitude)
- **`gyro_var_thresh`:** $0.0010\text{ rad}^2/\text{s}^2$ (max variance of gyro magnitude)
- **`gyro_mag_thresh`:** $0.028\text{ rad/s}$ ($\approx 1.6^\circ/\text{s}$, max mean gyro rate during stop)

### Maneuver Classifier (`src/maneuver.py`):
- **`stationary`:** Triggered whenever `zupt_flag == True`.
- **`reversal`:** Triggered when `speed_mps < -0.4 m/s`.
- **`straight` vs `gentle_curve` Boundary:**
  - `yaw_rate_gentle_thresh = 0.045 rad/s` ($2.58^\circ/\text{s}$, set strictly above straight-line lane-keeping noise floor).
  - `lat_acc_gentle_thresh  = 1.00 m/s²` (set above standard road camber / suspension roll induced offsets).
- **`gentle_curve` vs `sharp_turn` Boundary:**
  - `yaw_rate_sharp_thresh  = 0.120 rad/s` ($6.88^\circ/\text{s}$, e.g. 90° turns, roundabouts, U-turns).
  - `lat_acc_sharp_thresh   = 1.30 m/s²`.
- **`straight`:** Default state when turning conditions are inactive.
- **Post-Filter:** 3-sample median filter to eliminate single-sample boundary transitions.

---

## 5. Multi-Trip Benchmark Summary Table

Validated against synchronized CAN-bus ground truth across **194,903 samples (5.4 hours)** of real-world driving:

| Trip | Description | Samples | Duration | Yaw $r$ | Long Accel $r$ (smooth) | Lat Accel $r$ (smooth) | ZUPT Prec. | ZUPT Recall | ZUPT F1 | Maneuver Acc. | Straight F1 | Gentle Curve F1 | Sharp Turn F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | Driver A (City + Highway) | 51,746 | 86.2 min | **+0.9348** | **+0.5436** | **+0.4854** | **90.93%** | 71.50% | **0.8005** | **64.02%** | **0.6536** | **0.4702** | **0.7429** |
| **S3c** | Driver A (Roundabouts + Turns) | 37,183 | 62.0 min | **+0.9490** | +0.0140 | +0.0940 | **60.89%** | **92.37%** | **0.7340** | **72.03%** | **0.7993** | **0.4781** | **0.7540** |
| **M** | Driver B (Urban, Uncorrected) | 105,974 | 176.6 min | **+0.6363** | +0.1794 | **+0.4978** | **76.65%** | 63.28% | **0.6933** | **59.16%** | **0.6746** | **0.3522** | **0.6038** |
| **M (Lag-Corr)** | Driver B (Fair Physical Ground Truth) | 105,912 | 176.6 min | **+0.8812** | +0.1794 | **+0.4978** | **83.48%** | 68.77% | **0.7541** | **70.64%** | **0.7562** | **0.5030** | **0.7823** |

### Per-Class Detailed Metrics:
- **Trip S1 (Driver A, 86.2 min):**
  - `stationary`   -> Precision: **90.7%**, Recall: **72.0%**, F1: **0.803** (FPR: **0.76%**)
  - `straight`     -> Precision: **71.9%**, Recall: **59.9%**, F1: **0.654**
  - `gentle_curve` -> Precision: **43.3%**, Recall: **51.5%**, F1: **0.470**
  - `sharp_turn`   -> Precision: **67.4%**, Recall: **82.8%**, F1: **0.743**
- **Trip S3c (Driver A, 62.0 min):**
  - `stationary`   -> Precision: **60.5%**, Recall: **92.9%**, F1: **0.733** (FPR: **4.14%**)
  - `straight`     -> Precision: **79.7%**, Recall: **80.2%**, F1: **0.799**
  - `gentle_curve` -> Precision: **55.8%**, Recall: **41.8%**, F1: **0.478**
  - `sharp_turn`   -> Precision: **72.2%**, Recall: **78.9%**, F1: **0.754**
- **Trip M (Driver B, Lag-Corrected, 176.6 min):**
  - `stationary`   -> Precision: **83.4%**, Recall: **70.4%**, F1: **0.763** (FPR: **1.38%**)
  - `straight`     -> Precision: **81.0%**, Recall: **70.9%**, F1: **0.756**
  - `gentle_curve` -> Precision: **44.7%**, Recall: **57.5%**, F1: **0.503**
  - `sharp_turn`   -> Precision: **73.8%**, Recall: **83.3%**, F1: **0.782**

---

## 6. Known Limitations & Downstream Consumer Guidance

Downstream modules (**AI Speed Estimator**, **Error-State EKF Fusion**, **Map-Matching**) should account for the following design boundaries:

1. **`gentle_curve` Confidence Level:**
   - While `sharp_turn` (F1 $\approx 0.74-0.78$) and `straight` (F1 $\approx 0.65-0.80$) are well-separated, `gentle_curve` remains the weakest class with F1 $\approx 0.47-0.50$.
   - Highway lane changes and wide-radius gentle curves generate low angular rates ($1.5-3^\circ/\text{s}$) that partially overlap with road bank angles and lane-keeping micro-adjustments.
   - **Guidance for EKF / AI Model:** Downstream consumers should treat `gentle_curve` predictions with **lower confidence / broader covariance weights** than `sharp_turn` or `straight`.
2. **Fixed Boresight Angle ($\psi = 316.0^\circ$):**
   - The transformation matrix $\mathbf{R}_{p \to v}$ uses a fixed boresight rotation angle $\psi = 316.0^\circ$ ($\approx -44.0^\circ$), calibrated specifically for the IO-VNBD landscape dashboard cradle mount.
   - **Guidance for Hackathon Demo Day:** If the physical phone orientation or cradle mount differs during the live hackathon demonstration (e.g. portrait mount or flat on console), $\psi$ and axis mappings **must be re-estimated during initial calibration, not blindly re-used**.
3. **ZUPT Recall & False Non-Detections:**
   - ZUPT recall ranges from **$68\%$ to $93\%$** across trips. Engine idling vibrations, passenger movement, or wind buffeting during very brief traffic stops may cause variance to exceed thresholds momentarily.
   - **Guidance for EKF:** The EKF should **not** interpret `zupt_flag == False` as conclusive proof the vehicle is in motion. If GNSS speed is $0\text{ m/s}$ or AI speed estimation indicates near-zero velocity, the EKF should still constrain velocity drift.

---

## 7. How to Run & Verify

### Environment Setup:
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (numpy, pandas, scipy, matplotlib)
pip install -r requirements.txt
```

### Run End-to-End Benchmark Suite:
```bash
python run_demo.py
```

### Python API Integration:
```python
from src.pipeline import process_trip

results = process_trip(
    s_csv_path="data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv",
    v_csv_path="data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv",
    output_raw_path="output/RawSample_S1.csv",
    output_aligned_path="output/AlignedSample_S1.csv"
)

# Extract aligned dataframe for EKF / AI speed model
aligned_df = results["aligned_sample"]
R_p_to_v   = results["rotation_matrix"]
```
