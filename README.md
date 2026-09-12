# AI Speed Estimator for Smartphone-Based Intelligent Dead Reckoning (IDR)
### SIH Problem Statement 26168 (ISRO) — Track 2: AI Speed & Pseudo-Measurement Estimation

---

## 1. Executive Summary & Purpose

In smartphone-based vehicle navigation, standard GNSS positioning frequently fails in tunnels, underpasses, and urban canyons. Without access to OBD-II or vehicle CAN-bus wheel encoders, the downstream **Error-State Kalman Filter (ES-EKF)** cannot directly measure distance traveled or velocity.

The **AI Speed Estimator** replaces hardware wheel sensors by continuously estimating forward vehicle speed ($v \ge 0\text{ m/s}$) along with an empirical **uncertainty metric ($\sigma^2$)** and **confidence score ($C \in (0, 1]$)** directly from windowed IMU acceleration and gyroscope data at **10 Hz**.

```
+---------------------------+        +---------------------------+        +---------------------------+
| Sensing & Preprocessing   |  --->  |    AI Speed Estimator     |  --->  |    EKF Fusion Core        |
| (Upstream Module)         |        |   (This Module)           |        |   (Downstream Track)      |
| - Fixed Boresight Align   |        | - 2.0s Window FFT & Jerk  |        | - ES-EKF Filter Engine    |
| - ZUPT Stationary Detect  |        | - Regularized RF Ensemble |        | - Pseudo-measurement V    |
| - Maneuver Classification |        | - Physics Slew & Bounds   |        | - Measurement Covariance  |
+---------------------------+        +---------------------------+        +---------------------------+
```

---

## 2. Core Architecture & Production Model

The module is designed as a **single unified regressor** (not brittle per-maneuver submodels) paired with real-time physics bounds and a defensive fallback dispatcher:

```
                  +----------------------------------------------+
                  |  AlignedSample Stream (10 Hz)                |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Sliding Window Feature Extractor (2.0s, N=20|
                  |  - Time-Domain: fwd/lat/up accel & yaw rates |
                  |  - Orientation-Invariant: ||a||, jerk        |
                  |  - 20s Road-Roughness Rolling Min Baseline   |
                  |  - Spectral Bands: 0.5-2.0Hz, 2.0-4.5Hz      |
                  |  - Kinematics: Centripetal |alat|*|wyaw|     |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Optimized Random Forest (100 Trees)         |
                  |  - max_depth=16, min_samples_leaf=2          |
                  |  - max_features=0.5 (feature subsampling)    |
                  |  - Out-of-Bag (OOB) R²: 0.8727               |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Analytical Physics Post-Processor           |
                  |  - Non-negativity & Max speed: [0, 45 m/s]   |
                  |  - ZUPT Hard Lock: snap to 0.0 m/s           |
                  |  - Kinematic Slew Limit: <= 5.0 m/s² (0.5m/s)|
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  Defensive Fallback Dispatcher               |
                  |  - Confidence >= 0.25, Var <= 16.0 -> "ml"   |
                  |  - Low Confidence / High Var -> "fallback"   |
                  +----------------------------------------------+
                                         |
                                         v
                  +----------------------------------------------+
                  |  SpeedEstimates_*.csv Contract               |
                  +----------------------------------------------+
```

### Production Feature Importance Distribution
Using `max_features=0.5` combined with orientation-invariant 20s road-roughness baseline normalization eliminates single-feature dominance (`aup_std` reduced from 64.2% to 28.35%), balancing predictive load across multi-axis dynamics:
1. `aup_std` (Vertical vibration std): **28.35%**
2. `amag_var` (Total acceleration variance): **12.38%** (Orientation-invariant)
3. `amag_std` (Total acceleration magnitude std): **7.60%** (Orientation-invariant)
4. `wmag_mean` (Total angular velocity mean): **5.24%** (Orientation-invariant)
5. `amag_std_ratio_road` (20s Road-roughness relative ratio): **4.92%** (Orientation-invariant)
6. `wmag_std` (Total angular velocity std): **3.06%**
7. `afwd_mean` (Mean forward acceleration): **2.77%**
8. `amag_range` (Total acceleration range): **2.33%**
9. `wyaw_max` (Peak vehicle yaw rate): **2.19%**
10. `zupt_ratio` (Stationary window fraction): **2.12%**

---

## 3. Comparative Benchmark Evolution & Validation

The model was trained on **157,577 samples (4.38 hours)** across two distinct drivers (Trip S1 + Trip M) and validated across an unseen clean trip (**Trip S2**) and a known upstream cradle-swivel edge case (**Trip S3c**).

### Model Evolution Across Iterations

| Iteration / Checkpoint | Model Configuration | Train MAE | S1 MAE | M MAE | S2 Val MAE | S2 Gap | S2 Urban (5-15 m/s) | S3c Val MAE | S3c ML Usage % | S3c ML MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Original Baseline (Turn 8)** | `D=18, L=2, F=1.0` (Raw, Old Dispatcher) | **1.001** | 1.164 | 0.921 | **4.227** | +3.226 | 3.233 | **6.851** | 23.4% | 3.955 |
| **2. Aggressive Reg (Turn 9)** | `D=12, L=8, F='sqrt'` (Raw, Old Dispatcher) | **2.064** | 2.123 | 2.035 | **4.149** | +2.085 | 3.008 | **6.327** | 41.1% | 3.355 |
| **3. Balanced Model (Turn 10)** | `D=16, L=4, F='sqrt'` (60s Dual-Axis, Old Dispatcher) | **1.459** | 1.593 | 1.393 | **4.114** | +2.655 | 3.163 | **6.574** | 22.4% | 2.996 |
| **4. Final Production Model** | `D=16, L=2, F=0.5` (20s `\|a\|` Invariant, Tuned Dispatcher) | **1.138** | **1.244** | **1.086** | **3.897** | **+2.759** | **2.927** | **5.517** | **89.3%** | **5.153** |

> [!NOTE]
> **Methodological Note on ML-Active MAE Comparability**:
> ML-Active MAE is **not directly comparable across configurations with drastically different ML-usage percentages** because each metric is averaged over a fundamentally different (and differently difficult) subset of windows:
> - **Selective Gating (22.4% ML Usage, Row 3)**: The model only predicted on the easiest 22.4% of windows (primarily stationary and low-speed cruising where tree variance was small), achieving an artificially low subset MAE ($2.996\text{ m/s}$), while shunting all difficult high-speed and dynamic segments onto the corrupted fallback path (causing overall trip MAE to balloon to $6.574\text{ m/s}$).
> - **Production Gating (89.3% ML Usage, Row 4)**: The model predicted across almost the entire trip, including aggressive maneuvers, high-speed highway regimes ($>25\text{ m/s}$), and swivel transients. While the subset MAE on this broader, much harder distribution rose to $5.153\text{ m/s}$, keeping the ML model active prevented catastrophic dead-reckoning drift ($8.548\text{ m/s}$ error), driving overall trip MAE down from $6.574\text{ m/s} \to 5.517\text{ m/s}$ (a **$1.057\text{ m/s}$ net improvement**).

### Consolidated Multi-Trip Speed Estimation Benchmarks

| Trip Name | Role | Samples (10Hz) | Overall MAE (m/s) | Overall RMSE (m/s) | Pearson $r$ | ML-Active MAE (m/s) | ML Usage % | Fallback-Active MAE (m/s) | Bias (m/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip S1** | **TRAIN (Driver A)** | 51,684 | **1.244** | **1.761** | **0.924** | **1.227** | 99.4% | 4.185 | +0.174 |
| **Trip M** | **TRAIN (Driver B)** | 105,893 | **1.086** | **1.558** | **0.964** | **1.070** | 99.4% | 3.879 | -0.119 |
| **Trip S2** | **VAL (Clean Unseen)** | 93,816 | **3.897** | **5.101** | **0.520** | **3.865** | 96.1% | 4.687 | +0.094 |
| **Trip S3c** | **VAL (Swivel Corrupted)** | 37,122 | **5.517** | **7.639** | **0.689** | **5.153** | 89.3% | 8.548 | -4.421 |

---

### Diagnostic Deep-Dives

#### 1. Root Cause of S3c Fallback Collapse & Resolution
- **Pre-fix State (Row 3)**: In Turn 10, rigid dispatcher gating (`max_variance_threshold = 4.0`) dropped S3c ML usage to $22.4\%$, forcing $77.6\%$ of frames into physics dead reckoning. Because S3c contains 6 mid-trip azimuthal phone swivels, integrating the rotated forward accelerometer channel accumulated massive drift ($\text{MAE} = 7.603\text{ m/s}$ on fallback segments), pushing total trip MAE to $6.574\text{ m/s}$.
- **Resolution (Row 4)**: Tuning dispatcher thresholds (`max_variance_threshold = 16.0`, `confidence_threshold = 0.25`) and using 20s orientation-invariant `amag` rolling baselines restored ML usage to **$89.3\%$**, reducing overall trip MAE from **$6.574\text{ m/s} \to 5.517\text{ m/s}$** (a **$1.057\text{ m/s}$ improvement** over the pre-fix balanced state, and **$1.334\text{ m/s}$ improvement** over the initial unregularized baseline at $6.851\text{ m/s}$).

#### 2. Root Cause of S2 Stationary Anomaly
- **When ZUPT fired on S2 ($N=4,626$)**: $\text{MAE} = \mathbf{0.070\text{ m/s}}$ (Exact zero snap).
- **When ZUPT missed on S2 ($N=7,885$)**: $\text{MAE} = \mathbf{6.000\text{ m/s}}$ (Model predicted motion due to high engine idling vibrations of $0.15 - 0.25\text{ m/s}^2$).
- **Conclusion**: The stationary error on S2 is an engine idling vibration confound exceeding the upstream $0.08\text{ m/s}^2$ ZUPT threshold during stops, not an inference failure.

---

### 3.1. Reconciling Baseline Evolution & Exact Checkpoint Numbers
- **Initial Baseline (Turn 8)**: Evaluated at `lag_steps = 0` (no time-shift lag alignment), yielding Trip S2 MAE = **$3.801\text{ m/s}$** ($N=93,857$).
- **Calibrated Time Alignment (Turns 9–11)**: Following upstream Preprocessing Module standards, verified hardware time-lag alignment was applied across all trips (`lag_steps = -41` on S2, `+43` on S1, `+42` on S3c).
- **Exact Checkpoint Metrics on S2 Under Calibrated Alignment**:
  - **Original Unregularized Model (Turn 8, D=18, L=2, F=1.0)**: Pure-ML MAE = **$4.1307\text{ m/s}$** ($4.227\text{ m/s}$ with old fallback dispatcher).
  - **Turn 9 Historical Checkpoint (D=12, L=8, F='sqrt')**: Pure-ML MAE = **$4.1448\text{ m/s}$** ($4.1488\text{ m/s}$ with old fallback dispatcher).
  - **Turn 9 Retroactive Sweep Variant (D=12, L=8, F=1.0)**: Pure-ML MAE = **$4.1361\text{ m/s}$** ($4.131\text{ m/s}$ with old fallback dispatcher).
  - **Turn 10 Balanced Model (D=16, L=4, F='sqrt', 60s Dual Baseline)**: Pure-ML MAE = **$4.016\text{ m/s}$** ($4.114\text{ m/s}$ with old fallback dispatcher).
  - **Final Production Model (Turn 11, D=16, L=2, F=0.5, 20s `\|a\|` Baseline)**: Pure-ML MAE = **$3.8974\text{ m/s}$** ($3.8974\text{ m/s}$ full pipeline, $2.927\text{ m/s}$ in-distribution urban), achieving a genuine **$0.233\text{ m/s}$ Pure-ML improvement** and **$0.330\text{ m/s}$ full-pipeline improvement** over the unregularized baseline.

---

### 3.2. Out-of-Sample Validation on Completely Untouched Blind Trips

To strictly verify that the dispatcher thresholds (`confidence_threshold = 0.25`, `max_variance_threshold = 16.0`) and 20s `|a|` rolling normalization are not overfit to Trip S3c, the locked production model was evaluated across **four completely held-out, untouched trips**:

| Trip Name | Description / Role | Samples (10Hz) | Overall MAE (m/s) | Overall RMSE (m/s) | Pearson $r$ | ML Usage % | ML-Active MAE (m/s) | In-Dist Urban ($5-15\text{ m/s}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Trip S3b** | Held-out Clean (Driver A) | 6,794 | **2.167** | **2.893** | **0.570** | **95.1%** | **2.151** | **1.731** |
| **Trip S3a** | Held-out Clean (Driver A) | 24,602 | **3.670** | **4.671** | **0.668** | **94.9%** | **3.589** | **3.005** |
| **Trip S4** | Held-out Long Trip (Driver A) | 94,581 | **4.640** | **6.471** | **0.434** | **97.6%** | **4.619** | **2.772** ($N=50,256$) |
| **Trip Vfa01** | Held-out Driver E & Vehicle | 11,467 | **7.904** | **9.259** | **0.627** | **97.7%** | **7.900** | **2.710** ($N=3,156$) |

> [!IMPORTANT]
> **Key Takeaway**: Across all 4 untouched blind trips, ML usage holds remarkably steady at **$94.9\% - 97.7\%$**, confirming that the calibrated dispatcher thresholds generalize across diverse drivers, routes, and vehicle dynamics without premature fallback degradation.

---

## 4. Downstream EKF Hand-Off Specification & Continuous Uncertainty Semantics

### Critical Architecture Notice for the EKF Fusion Track
> [!WARNING]
> **ML-Active Semantic Shift & Continuous Variance Ingestion**:
> Because the dispatcher threshold was relaxed to prevent premature fallback to corrupted dead reckoning, the label `active_source == "ml_model"` no longer represents a binary guarantee of ultra-low error.
> 
> **Downstream EKF Implementation Rule**:
> - **DO NOT** treat `active_source` as a binary switch with fixed measurement noise.
> - **DO** ingest the continuous `speed_variance` column directly into the EKF measurement covariance update:
>   $$R_k = \max(\text{speed\_variance}_k, R_{\min}) + \sigma_{\text{floor}}^2$$
>   where $R_{\min} = 0.25\text{ m}^2/\text{s}^2$ and $\sigma_{\text{floor}}^2 = 0.50\text{ m}^2/\text{s}^2$.
> - Use the continuous `speed_confidence` score ($C_k \in (0.0, 1.0]$) for dynamic innovation gating (e.g. scaling the Mahalanobis gating threshold).

### Output CSV Contract Schema (`SpeedEstimates_<Trip>.csv`)

| Column Name | Type | Description | Downstream EKF Usage Guidance |
| :--- | :---: | :--- | :--- |
| `timestamp_s` | `float` | Sensor sample timestamp (seconds, 10 Hz) | Time synchronization with IMU mechanization step |
| `estimated_speed_mps` | `float` | Estimated forward velocity ($v \ge 0\text{ m/s}$) | Pseudo-measurement $z_k = v_k$ along vehicle body x-axis |
| `speed_variance` | `float` | Tree ensemble prediction variance $\sigma^2$ ($\text{m}^2/\text{s}^2$) | Measurement noise covariance $R_k = \max(\sigma^2, R_{\min}) + \sigma_0^2$ |
| `speed_confidence` | `float` | Normalized confidence score $[0.0, 1.0]$ | Dynamic thresholding for measurement gating / innovation checks |
| `active_source` | `str` | `"ml_model"` or `"physics_fallback"` | Diagnostic health tag (logs whether ML model or kinematic integrator is active) |
| `zupt_flag` | `bool` | True if vehicle is confirmed stationary | Triggers direct Zero Velocity Update ($v=0, P \to \text{reset}$) |
| `maneuver_state` | `str` | `"stationary"`, `"straight"`, `"gentle_curve"`, `"sharp_turn"` | Context tag for maneuver-dependent process noise scaling |

### Downstream EKF Measurement Update Formulation:
$$z_k = \begin{bmatrix} v_{\text{fwd}} \end{bmatrix}, \quad H_k = \begin{bmatrix} 1 & 0 & 0 & \dots \end{bmatrix}$$
$$R_k = \text{speed\_variance}_k + \sigma_{\text{floor}}^2$$
$$\text{Innovation: } y_k = z_k - \hat{v}_k, \quad S_k = H_k P_k^- H_k^T + R_k$$
$$\text{Kalman Gain: } K_k = P_k^- H_k^T S_k^{-1}$$

---

## 5. Quickstart & Usage

### 1. Run Complete Multi-Trip Training & Benchmarks:
```bash
python train_and_evaluate.py
```
Outputs trained model `models/speed_estimator_rf.pkl`, benchmark tables, output CSV contracts in `output/`, and high-resolution diagnostic plots in `plots/`.

### 2. Run Standalone Inference Demo on Any Trip:
```bash
python run_demo.py --input ../sih_idr_preprocessing/output/AlignedSample_S1.csv --output output/SpeedEstimates_demo.csv
```
