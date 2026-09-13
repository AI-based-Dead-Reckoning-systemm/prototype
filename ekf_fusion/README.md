# EKF Fusion Core & Output Module (Track 3)

> **SIH Problem Statement 26168 (ISRO)**: Smartphone-based Intelligent Dead Reckoning (IDR) System  
> **Module Owner:** Person C  
> **Status:** Track 3 Interface Specification & Implementation Target

---

## 1. Module Responsibilities & System Context

The **EKF Fusion Core** ingests upstream preprocessed inertial/GNSS streams and AI speed estimates, mechanizes vehicle kinematics in the navigation frame (NED), and produces the fused, continuous navigation solution:

```
┌───────────────────────────────┐        ┌──────────────────────────────┐
│  preprocessing/AlignedSample  │        │ speed_estimation/SpeedEst    │
│  - IMU (afwd, alat, aup)      │        │ - v_est (forward speed)      │
│  - Gyro (wyaw, wpitch, wroll) │        │ - speed_variance (sigma^2)   │
│  - zupt_flag, maneuver_state  │        │ - speed_confidence (score)   │
│  - GNSS coordinates & dop     │        │ - active_source (tag)        │
└──────────────┬────────────────┘        └──────────────┬───────────────┘
               │                                        │
               └───────────────────┬────────────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │    ekf_fusion/ (Track 3)      │
                   │    - 15-State Error-State EKF │
                   │    - Soft-GNSS Trust Weighting│
                   │    - Continuous Uncertainty Rk│
                   └───────────────┬───────────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │       IDRFusedEstimate        │
                   │  (schemas.IDRFusedEstimate)   │
                   └───────────────────────────────┘
```

---

## 2. Inbound Data Contracts & Integration Rules

### 1. Ingesting AI Speed Estimates (`SpeedEstimate`)
From top-level [`schemas.py`](../schemas.py):

* **Pseudo-Measurement Vector**:
  $$z_k = \begin{bmatrix} v_{\text{fwd}} \end{bmatrix}, \quad H_k = \begin{bmatrix} 0 & 0 & 0 & 1 & 0 & 0 & \dots \end{bmatrix}$$
* **Continuous Measurement Covariance Update ($R_k$)**:
  Do **NOT** treat `active_source` as a binary confidence switch. The filter must ingest the continuous `speed_variance` column directly:
  $$R_k = \max(\text{speed\_variance}_k, R_{\min}) + \sigma_{\text{floor}}^2$$
  where $R_{\min} = 0.25\text{ m}^2/\text{s}^2$ and $\sigma_{\text{floor}}^2 = 0.50\text{ m}^2/\text{s}^2$.
* **Dynamic Innovation Gating**:
  Scale the Mahalanobis gating threshold using `speed_confidence` ($C_k \in (0.0, 1.0]$) to protect against severe transient outliers.

### 2. Soft-GNSS Continuous Trust Weighting
* Do **NOT** toggle between binary `"GNSS"` and `"DEAD_RECKONING"` modes.
* Compute continuous soft trust weight $\gamma_k \in [0.0, 1.0]$ based on `gps_accuracy_m`, `gps_satellites`, and innovation residuals:
  $$R_{\text{GNSS}, k} = \frac{R_{\text{GNSS, nominal}}}{\max(\gamma_k, 10^{-4})}$$
  - Open sky ($\text{accuracy} < 3\text{m}, \text{sats} > 15$): $\gamma_k \to 1.0$ (tight GNSS tracking).
  - Degraded / Multipath: $\gamma_k \in (0.1, 0.5)$ (soft trust scaling).
  - Tunnel Outage / Blackout: $\gamma_k = 0.0, \text{gnss\_active} = \text{False}$ (pure IDR mechanization).

---

## 3. Outbound Data Contract: `IDRFusedEstimate`

Every epoch output must serialize to [`IDRFusedEstimate`](../schemas.py) with the following fields:
* `timestamp_s`: Epoch timestamp (10 Hz)
* `pos_lat`, `pos_lon`, `pos_alt`: Fused WGS84 coordinates
* `vel_north_mps`, `vel_east_mps`, `vel_down_mps`, `vel_forward_mps`: Velocities in NED and vehicle frame
* `heading_deg`, `pitch_deg`, `roll_deg`: Vehicle attitude angles
* `pos_uncertainty_m`, `vel_uncertainty_mps`, `heading_uncertainty_deg`: Continuous $1\sigma$ filter covariance diagonals
* `gnss_trust_weight`: Float in $[0.0, 1.0]$
* `gnss_active`, `active_speed_source`, `zupt_active`, `maneuver_state`: Diagnostic flags
