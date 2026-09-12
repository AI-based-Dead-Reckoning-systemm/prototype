"""
schemas.py - Single Source of Truth for SIH 26168 (ISRO) IDR Data Contracts.

This file defines the canonical Python dataclasses for all data streams crossing
module boundaries between the three independent sub-systems:
  1. Sensing & Preprocessing (Person A: preprocessing/)
  2. AI Speed Estimator (Person B: speed_estimation/)
  3. EKF Fusion Core & Output (Person C: ekf_fusion/ Navigation Filter)

Derived directly from audited, committed production code and CSV outputs
(RawSample_*.csv, AlignedSample_*.csv, SpeedEstimates_*.csv).
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Union, Optional, List


def _parse_float(val: Any, default: float = 0.0) -> float:
    """Helper to safely parse float values from CSV strings or numbers."""
    if val is None:
        return default
    if isinstance(val, (float, int)):
        return float(val)
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ("nan", "null", "none"):
        return float("nan")
    try:
        return float(val_str)
    except ValueError:
        return default


def _parse_int(val: Any, default: int = 0) -> int:
    """Helper to safely parse int values from CSV strings or numbers."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if not math.isnan(val) else default
    val_str = str(val).strip()
    if val_str == "" or val_str.lower() in ("nan", "null", "none"):
        return default
    try:
        return int(float(val_str))
    except ValueError:
        return default


def _parse_bool(val: Any, default: bool = False) -> bool:
    """Helper to safely parse bool values from CSV representations."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    val_str = str(val).strip().lower()
    if val_str in ("true", "1", "t", "yes", "y"):
        return True
    if val_str in ("false", "0", "f", "no", "n"):
        return False
    return default


def _parse_str(val: Any, default: str = "") -> str:
    """Helper to safely parse string values."""
    if val is None:
        return default
    return str(val).strip()


# ==============================================================================
# 1. RawSample Contract (Sensing & Preprocessing Raw Smartphone Stream)
# ==============================================================================
@dataclass
class RawSample:
    """
    Contract 1: Raw Normalized Smartphone Sensor Stream (10 Hz).

    PRODUCER: Sensing & Preprocessing Module (`preprocessing.src.loader.load_raw_smartphone`)
    CONSUMERS: Preprocessing Pipeline, Noise Floor Characterization, Diagnostic Logging

    Notes:
    - Acceleration includes Earth gravity (~9.81 m/s^2 along the gravity vector).
    - Gyroscope is in radians/second (converted from deg/s if source was deg/s).
    - GNSS fields are prefixed with 'gps_' matching committed CSV outputs.
    - GNSS coordinates/speed will be NaN during tunnel/blackout outages.
    """
    timestamp_s: float          # Elapsed monotonic time from trip start [s]
    accel_x: float              # Phone body X acceleration (incl. gravity) [m/s^2]
    accel_y: float              # Phone body Y acceleration (incl. gravity) [m/s^2]
    accel_z: float              # Phone body Z acceleration (incl. gravity) [m/s^2]
    gyro_x: float               # Phone body X angular velocity [rad/s]
    gyro_y: float               # Phone body Y angular velocity [rad/s]
    gyro_z: float               # Phone body Z angular velocity [rad/s]
    mag_x: float                # Phone body X magnetic field [uT]
    mag_y: float                # Phone body Y magnetic field [uT]
    mag_z: float                # Phone body Z magnetic field [uT]
    gravity_x: float            # Android OS estimated gravity vector X [m/s^2]
    gravity_y: float            # Android OS estimated gravity vector Y [m/s^2]
    gravity_z: float            # Android OS estimated gravity vector Z [m/s^2]
    gps_lat: float              # GNSS WGS84 Latitude [deg] (NaN during outage)
    gps_lon: float              # GNSS WGS84 Longitude [deg] (NaN during outage)
    gps_alt: float              # GNSS Altitude above sea level [m]
    gps_speed_mps: float        # GNSS ground speed [m/s]
    gps_accuracy_m: float       # GNSS horizontal 1-sigma accuracy radius [m]
    gps_heading_deg: float      # GNSS Course-over-Ground [deg] (0-360)
    gps_satellites: int         # Satellites used in GNSS fix [count]

    @classmethod
    def from_csv_row(cls, row: Union[Dict[str, Any], Any]) -> "RawSample":
        """Constructs a RawSample instance from a dictionary or pandas Series."""
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        return cls(
            timestamp_s=_parse_float(get("timestamp_s")),
            accel_x=_parse_float(get("accel_x")),
            accel_y=_parse_float(get("accel_y")),
            accel_z=_parse_float(get("accel_z")),
            gyro_x=_parse_float(get("gyro_x")),
            gyro_y=_parse_float(get("gyro_y")),
            gyro_z=_parse_float(get("gyro_z")),
            mag_x=_parse_float(get("mag_x")),
            mag_y=_parse_float(get("mag_y")),
            mag_z=_parse_float(get("mag_z")),
            gravity_x=_parse_float(get("gravity_x")),
            gravity_y=_parse_float(get("gravity_y")),
            gravity_z=_parse_float(get("gravity_z")),
            gps_lat=_parse_float(get("gps_lat")),
            gps_lon=_parse_float(get("gps_lon")),
            gps_alt=_parse_float(get("gps_alt")),
            gps_speed_mps=_parse_float(get("gps_speed_mps")),
            gps_accuracy_m=_parse_float(get("gps_accuracy_m")),
            gps_heading_deg=_parse_float(get("gps_heading_deg")),
            gps_satellites=_parse_int(get("gps_satellites")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. AlignedSample Contract (Preprocessing -> Speed Estimator & EKF Fusion)
# ==============================================================================
@dataclass
class AlignedSample:
    """
    Contract 2: Vehicle-Frame Aligned Inertial & Kinematic Sample (10 Hz).

    PRODUCER: Sensing & Preprocessing Module (`preprocessing.src.pipeline.process_trip`)
    CONSUMERS:
      - AI Speed Estimator (`speed_estimation/`)
      - EKF Fusion Core (`ekf_fusion/`)
      - Map-Matching / Trajectory Engine

    Axes and Conventions:
    - Vehicle Body Frame:
        +X_v (fwd): Forward longitudinal axis (+ = acceleration, - = braking)
        +Y_v (lat): Lateral axis (+ = left turn centripetal accel)
        +Z_v (up) : Vertical axis (gravity subtracted, nominal ~0 m/s^2)
    - Gyroscope rates:
        gyro_vehicle_yaw: Vehicle yaw rate (+ = left turn, right-hand rule)
        gyro_vehicle_pitch: Pitch rate (+ = nose up)
        gyro_vehicle_roll: Roll rate (+ = right side down)

    Maneuver Taxonomy:
    - Uses a 5-class taxonomy:
        {'stationary', 'straight', 'gentle_curve', 'sharp_turn', 'reversal'}
    - CRITICAL CONSUMER GUIDANCE: 'gentle_curve' is documented as the weakest-performing
      class (F1 ~0.47-0.50 across multi-trip benchmarks due to road banking/camber noise).
      Downstream modules (AI Speed Estimator & EKF) MUST treat 'gentle_curve' with
      lower confidence / broader covariance weights than 'sharp_turn' (F1 ~0.74-0.82)
      or 'straight' (F1 ~0.65-0.83).

    Stationary Detection (ZUPT):
    - zupt_flag: Zero Velocity Update flag. True when 0.5s IMU variance < calibrated thresholds.
    - CRITICAL EKF GUIDANCE: ZUPT recall is ~68-93%. zupt_flag == False should NOT be
      interpreted as conclusive proof the vehicle is in motion (e.g. engine idle vibration).
    """
    timestamp_s: float          # Monotonic timestamp sampled at 10 Hz [s]
    accel_vehicle_fwd: float    # Forward longitudinal acceleration [m/s^2]
    accel_vehicle_lat: float    # Lateral acceleration (+ = left turn) [m/s^2]
    accel_vehicle_up: float     # Vertical acceleration (gravity removed) [m/s^2]
    gyro_vehicle_yaw: float     # Vehicle yaw angular velocity [rad/s]
    gyro_vehicle_pitch: float   # Vehicle pitch angular velocity [rad/s]
    gyro_vehicle_roll: float    # Vehicle roll angular velocity [rad/s]
    zupt_flag: bool             # Zero Velocity Update detection (True = at rest)
    maneuver_state: str         # 5-class taxonomy: stationary, straight, gentle_curve, sharp_turn, reversal
    gps_lat: float              # GNSS Latitude [deg] (NaN during outage)
    gps_lon: float              # GNSS Longitude [deg] (NaN during outage)
    gps_alt: float              # GNSS Altitude [m]
    gps_speed_mps: float        # GNSS ground speed [m/s]
    gps_accuracy_m: float       # GNSS horizontal 1-sigma accuracy [m]
    gps_heading_deg: float      # GNSS course heading [deg]
    gps_satellites: int         # GNSS satellite count [count]

    @classmethod
    def from_csv_row(cls, row: Union[Dict[str, Any], Any]) -> "AlignedSample":
        """Constructs an AlignedSample instance from a dictionary or pandas Series."""
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        return cls(
            timestamp_s=_parse_float(get("timestamp_s")),
            accel_vehicle_fwd=_parse_float(get("accel_vehicle_fwd")),
            accel_vehicle_lat=_parse_float(get("accel_vehicle_lat")),
            accel_vehicle_up=_parse_float(get("accel_vehicle_up")),
            gyro_vehicle_yaw=_parse_float(get("gyro_vehicle_yaw")),
            gyro_vehicle_pitch=_parse_float(get("gyro_vehicle_pitch")),
            gyro_vehicle_roll=_parse_float(get("gyro_vehicle_roll")),
            zupt_flag=_parse_bool(get("zupt_flag")),
            maneuver_state=_parse_str(get("maneuver_state"), default="straight"),
            gps_lat=_parse_float(get("gps_lat")),
            gps_lon=_parse_float(get("gps_lon")),
            gps_alt=_parse_float(get("gps_alt")),
            gps_speed_mps=_parse_float(get("gps_speed_mps")),
            gps_accuracy_m=_parse_float(get("gps_accuracy_m")),
            gps_heading_deg=_parse_float(get("gps_heading_deg")),
            gps_satellites=_parse_int(get("gps_satellites")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 3. SpeedEstimate Contract (AI Speed Estimator -> EKF Fusion Core)
# ==============================================================================
@dataclass
class SpeedEstimate:
    """
    Contract 3: AI Speed Estimator Output with Continuous Uncertainty (10 Hz).

    PRODUCER: AI Speed Estimator Module (`sih_idr_speed_estimation.src.pipeline.process_speed_estimation`)
    CONSUMER: EKF Fusion Core (`sih_idr_ekf`)

    CRITICAL EKF INTEGRATION ARCHITECTURE SPECIFICATION:
    1. Continuous Measurement Noise (R_k):
       Downstream EKF MUST ingest the continuous `speed_variance` directly into the measurement
       noise covariance update:
           R_k = max(speed_variance_k, R_min) + sigma_floor^2
       where:
           R_min = 0.25 m^2/s^2
           sigma_floor^2 = 0.50 m^2/s^2 (minimum noise floor)
       DO NOT assign a fixed, hardcoded measurement noise based on `active_source`.

    2. Dynamic Innovation Gating:
       Use `speed_confidence` (C_k in (0.0, 1.0], computed as 1.0 / (1.0 + sqrt(speed_variance)/1.5))
       to scale Mahalanobis gating thresholds during high-uncertainty / maneuver epochs.

    3. Semantics of `active_source`:
       The field `active_source` ('ml_model' vs 'physics_fallback') is a diagnostic health tag.
       Following dispatcher threshold tuning, `active_source == 'ml_model'` alone does NOT
       reliably guarantee low error in out-of-distribution conditions; `speed_variance` is
       the primary, continuous trust signal.
    """
    timestamp_s: float          # Sensor sample timestamp (10 Hz) [s]
    estimated_speed_mps: float  # Estimated forward velocity (v >= 0) [m/s]
    speed_variance: float       # Tree ensemble prediction variance sigma^2 [m^2/s^2]
    speed_confidence: float     # Continuous confidence score in (0.0, 1.0] [unitless]
    active_source: str          # Diagnostic source tag: 'ml_model' or 'physics_fallback'
    zupt_flag: bool             # Stationary flag (True = zero velocity lock)
    maneuver_state: str         # Maneuver classification context tag

    @classmethod
    def from_csv_row(cls, row: Union[Dict[str, Any], Any]) -> "SpeedEstimate":
        """Constructs a SpeedEstimate instance from a dictionary or pandas Series."""
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        return cls(
            timestamp_s=_parse_float(get("timestamp_s")),
            estimated_speed_mps=_parse_float(get("estimated_speed_mps")),
            speed_variance=_parse_float(get("speed_variance")),
            speed_confidence=_parse_float(get("speed_confidence"), default=1.0),
            active_source=_parse_str(get("active_source"), default="ml_model"),
            zupt_flag=_parse_bool(get("zupt_flag")),
            maneuver_state=_parse_str(get("maneuver_state"), default="straight"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 4. IDRFusedEstimate Contract (EKF Fusion Core -> Downstream Navigation / UI)
# ==============================================================================
@dataclass
class IDRFusedEstimate:
    """
    Contract 4: EKF Multi-Sensor Fused State & Uncertainty Output (10 Hz).

    PRODUCER: EKF Fusion Core Module (`sih_idr_ekf` / Person C)
    CONSUMERS: Navigation UI / Dashboard, Trajectory Exporter, Evaluation Engine, Map-Matching

    STATUS: Proposed Canonical Contract Specification (Drafted for Person C / EKF Module)

    CRITICAL SOFT-GNSS FUSION ARCHITECTURE SPECIFICATION:
    - GNSS fallback is NOT a hard binary toggle between GNSS and Dead-Reckoning.
    - Uses continuous soft trust weighting (`gnss_trust_weight` in [0.0, 1.0]):
        gnss_trust_weight = f(gps_accuracy_m, gps_satellites, innovation_residual)
        R_GNSS_k = R_GNSS_nominal / max(gnss_trust_weight, 1e-4)
      In open-sky with high accuracy, gnss_trust_weight -> 1.0 (tight GNSS tracking).
      In degraded GNSS (urban canyon/multipath), gnss_trust_weight smoothly drops towards 0.0,
      allowing the AI speed estimator and IMU mechanization to carry position smoothly.
      During total tunnel outages, gnss_active = False, gnss_trust_weight = 0.0 (pure IDR).
    """
    timestamp_s: float              # Epoch timestamp [s]
    pos_lat: float                  # Fused WGS84 Latitude [deg]
    pos_lon: float                  # Fused WGS84 Longitude [deg]
    pos_alt: float                  # Fused Altitude above sea level [m]
    vel_north_mps: float            # North velocity in NED navigation frame [m/s]
    vel_east_mps: float             # East velocity in NED navigation frame [m/s]
    vel_down_mps: float             # Down velocity in NED navigation frame [m/s]
    vel_forward_mps: float          # Vehicle body forward velocity [m/s]
    heading_deg: float              # Vehicle course/yaw heading [deg] (0-360 clockwise from North)
    pitch_deg: float                # Vehicle pitch attitude [deg] (+ = nose up)
    roll_deg: float                 # Vehicle roll attitude [deg] (+ = right side down)
    pos_uncertainty_m: float        # Horizontal position 1-sigma uncertainty sqrt(P_nn + P_ee) [m]
    vel_uncertainty_mps: float      # Forward/horizontal velocity 1-sigma uncertainty [m/s]
    heading_uncertainty_deg: float  # Heading attitude 1-sigma uncertainty [deg]
    gnss_trust_weight: float        # Soft-GNSS continuous trust weight in [0.0, 1.0] (0 = pure DR, 1 = full GNSS)
    gnss_active: bool               # True if valid GNSS measurement ingested in this epoch
    active_speed_source: str        # 'ml_model', 'physics_fallback', or 'gnss_doppler'
    zupt_active: bool               # True if zero-velocity constraint active in filter update
    maneuver_state: str             # Current vehicle maneuver state context

    @classmethod
    def from_csv_row(cls, row: Union[Dict[str, Any], Any]) -> "IDRFusedEstimate":
        """Constructs an IDRFusedEstimate instance from a dictionary or pandas Series."""
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        return cls(
            timestamp_s=_parse_float(get("timestamp_s")),
            pos_lat=_parse_float(get("pos_lat")),
            pos_lon=_parse_float(get("pos_lon")),
            pos_alt=_parse_float(get("pos_alt")),
            vel_north_mps=_parse_float(get("vel_north_mps")),
            vel_east_mps=_parse_float(get("vel_east_mps")),
            vel_down_mps=_parse_float(get("vel_down_mps")),
            vel_forward_mps=_parse_float(get("vel_forward_mps")),
            heading_deg=_parse_float(get("heading_deg")),
            pitch_deg=_parse_float(get("pitch_deg")),
            roll_deg=_parse_float(get("roll_deg")),
            pos_uncertainty_m=_parse_float(get("pos_uncertainty_m")),
            vel_uncertainty_mps=_parse_float(get("vel_uncertainty_mps")),
            heading_uncertainty_deg=_parse_float(get("heading_uncertainty_deg")),
            gnss_trust_weight=_parse_float(get("gnss_trust_weight"), default=1.0),
            gnss_active=_parse_bool(get("gnss_active"), default=True),
            active_speed_source=_parse_str(get("active_speed_source"), default="ml_model"),
            zupt_active=_parse_bool(get("zupt_active"), default=False),
            maneuver_state=_parse_str(get("maneuver_state"), default="straight"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 5. GroundTruthSample Contract (CAN Bus Reference Stream)
# ==============================================================================
@dataclass
class GroundTruthSample:
    """
    Contract 5: Synchronized CAN Bus Ground Truth Stream (10 Hz).

    PRODUCER: Dataset Loader (`preprocessing.src.loader.load_ground_truth_can`)
    CONSUMERS: Evaluation Scripts, Benchmark Validation, Error Analysis
    """
    timestamp_s: float          # Timestamp [s]
    gt_speed_mps: float         # CAN Bus true vehicle speed [m/s]
    gt_lat: float               # High-precision RTK/GPS Latitude [deg]
    gt_lon: float               # High-precision RTK/GPS Longitude [deg]
    gt_heading_deg: float       # Reference vehicle heading [deg]
    gt_yaw_rate_rads: float     # CAN Bus yaw rate [rad/s]
    gt_yaw_rate_degs: float     # CAN Bus yaw rate [deg/s]
    gt_steering_deg: float      # CAN Bus steering wheel angle [deg]
    gt_long_accel_mps2: float   # CAN Bus longitudinal acceleration [m/s^2]
    gt_lat_accel_mps2: float    # CAN Bus lateral acceleration [m/s^2]
    gt_is_stationary: bool      # True if gt_speed_mps < 0.1 m/s

    @classmethod
    def from_csv_row(cls, row: Union[Dict[str, Any], Any]) -> "GroundTruthSample":
        """Constructs a GroundTruthSample instance from a dictionary or pandas Series."""
        get = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        return cls(
            timestamp_s=_parse_float(get("timestamp_s")),
            gt_speed_mps=_parse_float(get("gt_speed_mps")),
            gt_lat=_parse_float(get("gt_lat")),
            gt_lon=_parse_float(get("gt_lon")),
            gt_heading_deg=_parse_float(get("gt_heading_deg")),
            gt_yaw_rate_rads=_parse_float(get("gt_yaw_rate_rads")),
            gt_yaw_rate_degs=_parse_float(get("gt_yaw_rate_degs")),
            gt_steering_deg=_parse_float(get("gt_steering_deg")),
            gt_long_accel_mps2=_parse_float(get("gt_long_accel_mps2")),
            gt_lat_accel_mps2=_parse_float(get("gt_lat_accel_mps2")),
            gt_is_stationary=_parse_bool(get("gt_is_stationary")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
