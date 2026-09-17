import numpy as np
import math

class ReducedEKF:
    def __init__(self, initial_lat, initial_lon, initial_heading_rad):
        # State: [lat, lon, v_forward, heading_rad, gyro_bias_yaw, accel_bias_fwd]
        self.x = np.array([
            initial_lat,
            initial_lon,
            0.0,
            initial_heading_rad,
            0.0,
            0.0
        ], dtype=float)
        
        # Covariance
        self.P = np.diag([
            1e-8,  # lat (deg^2)
            1e-8,  # lon (deg^2)
            1.0,   # v_forward (m/s)^2
            0.1,   # heading_rad (rad^2)
            0.01,  # gyro_bias_yaw (rad/s)^2
            0.1    # accel_bias_fwd (m/s^2)^2
        ])
        
        self.last_gnss_trust_weight = 1.0
        self.gnss_active = True
        self.R_earth = 111320.0 # roughly meters per degree at equator

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, dt, accel_vehicle_fwd, gyro_vehicle_yaw):
        if dt <= 0:
            return
            
        lat, lon, v, heading, b_g, b_a = self.x
        
        c_lat = self.R_earth
        c_lon = self.R_earth * math.cos(math.radians(lat)) if math.cos(math.radians(lat)) != 0 else self.R_earth
        
        # Jacobian F using prior state
        F = np.eye(6)
        F[0, 2] = (math.cos(heading) * dt) / c_lat
        F[0, 3] = (-v * math.sin(heading) * dt) / c_lat
        F[1, 2] = (math.sin(heading) * dt) / c_lon
        F[1, 3] = (v * math.cos(heading) * dt) / c_lon
        F[2, 5] = -dt
        # d(heading_new)/d(b_g) = +dt because heading_new = heading - (gyro - b_g)*dt
        F[3, 4] = dt
        
        # Process noise Q (tunable placeholders)
        Q = np.diag([
            1e-12, # lat
            1e-12, # lon
            1e-2,  # v
            1e-4,  # heading
            1e-6,  # b_g
            1e-4   # b_a
        ])
        
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        
        # Propagate states: vehicle frame yaw rate is positive for left turns (CCW),
        # while geographic heading is clockwise from North. Turning left decreases heading.
        v_new = v + (accel_vehicle_fwd - b_a) * dt
        heading_new = heading - (gyro_vehicle_yaw - b_g) * dt
        heading_new = self._wrap_angle(heading_new)
        
        lat_new = lat + (v_new * math.cos(heading_new) * dt) / c_lat
        lon_new = lon + (v_new * math.sin(heading_new) * dt) / c_lon
        
        self.x[0] = lat_new
        self.x[1] = lon_new
        self.x[2] = v_new
        self.x[3] = heading_new

    def _update(self, z, H, R, gating_threshold=None):
        y = z - H @ self.x
        
        if len(y) > 0 and H.shape[0] == 1 and H[0, 3] == 1.0:
            y[0] = self._wrap_angle(y[0])
            
        S = H @ self.P @ H.T + R
        
        if gating_threshold is not None:
            # Mahalanobis distance D^2 = y^T * S^-1 * y
            if np.isscalar(y):
                D2 = (y ** 2) / S
            else:
                D2 = y.T @ np.linalg.inv(S) @ y
            if D2 > gating_threshold:
                return # Reject outlier
                
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = self._wrap_angle(self.x[3])
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_speed(self, estimated_speed_mps, speed_variance, speed_confidence):
        if math.isnan(estimated_speed_mps) or math.isnan(speed_variance):
            return
            
        R_k = max(speed_variance, 0.25) + 0.50
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        z = np.array([estimated_speed_mps])
        R = np.array([[R_k]])
        
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # DECOUPLING: A 1D speed measurement gives no positional or directional information.
        # Spurious cross-covariances during long GNSS outages cause speed residuals
        # to violently rotate the heading. Force speed to ONLY update velocity and accel_bias.
        K[0] = 0.0  # lat
        K[1] = 0.0  # lon
        K[3] = 0.0  # heading
        K[4] = 0.0  # gyro_bias
        
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def update_gnss(self, gps_lat, gps_lon, gps_speed_mps, gps_accuracy_m, gps_satellites):
        if math.isnan(gps_lat) or math.isnan(gps_lon):
            self.gnss_active = False
            self.last_gnss_trust_weight = 0.0
            return
            
        if gps_satellites <= 3:
            gamma = 0.0
        elif gps_accuracy_m < 3.0 and gps_satellites > 15:
            gamma = 1.0
        elif (4 <= gps_satellites <= 15) or (3.0 <= gps_accuracy_m <= 10.0):
            gamma = 0.3
        else:
            gamma = 0.3 # default degraded
            
        self.last_gnss_trust_weight = gamma
        
        if gamma == 0.0:
            self.gnss_active = False
            return
            
        self.gnss_active = True
        
        # R_nominal = 5.0 (tunable)
        R_nominal = 5.0
        R_gnss = R_nominal / max(gamma, 1e-4)
        
        c_lat = self.R_earth
        c_lon = self.R_earth * math.cos(math.radians(self.x[0]))
        if c_lon == 0: c_lon = self.R_earth
        
        R_lat = R_gnss / (c_lat ** 2)
        R_lon = R_gnss / (c_lon ** 2)
        R_v = R_gnss 
        
        H = np.zeros((3, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        if not math.isnan(gps_speed_mps):
            H[2, 2] = 1.0
            
        z = np.array([gps_lat, gps_lon, 0.0 if math.isnan(gps_speed_mps) else gps_speed_mps])
        R = np.diag([R_lat, R_lon, R_v])
        
        if math.isnan(gps_speed_mps):
            H = H[:2, :]
            z = z[:2]
            R = R[:2, :2]
            
        self._update(z, H, R)

    def update_heading(self, heading_rad, var=0.05):
        """
        Fuses a direct heading measurement (e.g. GNSS course-over-ground when moving).
        State index 3 is heading_rad.
        """
        if math.isnan(heading_rad):
            return
        H = np.zeros((1, 6))
        H[0, 3] = 1.0
        z = np.array([heading_rad])
        R = np.array([[var]])
        self._update(z, H, R)

    def update_bias_prior(self, known_accel_bias_fwd, prior_confidence=0.05):
        """
        Applies a weak pseudo-measurement pulling accel_bias_fwd (state index 5)
        toward known_accel_bias_fwd.
        """
        z = np.array([known_accel_bias_fwd])
        H = np.zeros((1, 6))
        H[0, 5] = 1.0
        R = np.array([[prior_confidence]])
        self._update(z, H, R)

    def update_nhc(self, constraint_variance=1e-4):
        """
        Applies the non-holonomic constraint (NHC). Since lateral velocity is
        structurally zero in this reduced state, we apply this as a tight constraint
        on gyro_bias_yaw (state 4) to prevent it from wandering to absorb NHC violations.
        """
        z = np.array([0.0])
        H = np.zeros((1, 6))
        H[0, 4] = 1.0
        R = np.array([[constraint_variance]])
        self._update(z, H, R)

    def update_zupt(self, zupt_flag):
        if zupt_flag:
            H = np.zeros((1, 6))
            H[0, 2] = 1.0
            z = np.array([0.0])
            R = np.array([[1e-4]]) # high confidence
            
            y = z - H @ self.x
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Decouple: zero velocity tells us nothing about heading or gyro bias
            K[0] = 0.0 # lat
            K[1] = 0.0 # lon
            K[3] = 0.0 # heading
            K[4] = 0.0 # gyro_bias
            
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ H) @ self.P

    def get_estimate(self, timestamp_s, active_speed_source, maneuver_state, zupt_flag, gps_alt=0.0):
        pos_uncertainty_m = math.sqrt(self.P[0,0]*(self.R_earth**2) + self.P[1,1]*((self.R_earth * math.cos(math.radians(self.x[0])))**2))
        vel_uncertainty_mps = math.sqrt(self.P[2, 2])
        heading_uncertainty_deg = math.degrees(math.sqrt(self.P[3, 3]))
        
        return {
            "timestamp_s": timestamp_s,
            "pos_lat": self.x[0],
            "pos_lon": self.x[1],
            "pos_alt": gps_alt,
            "vel_north_mps": self.x[2] * math.cos(self.x[3]),
            "vel_east_mps": self.x[2] * math.sin(self.x[3]),
            "vel_down_mps": 0.0,
            "vel_forward_mps": self.x[2],
            "heading_deg": math.degrees(self.x[3]),
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "pos_uncertainty_m": pos_uncertainty_m,
            "vel_uncertainty_mps": vel_uncertainty_mps,
            "heading_uncertainty_deg": heading_uncertainty_deg,
            "gnss_trust_weight": self.last_gnss_trust_weight,
            "gnss_active": self.gnss_active,
            "active_speed_source": active_speed_source,
            "zupt_active": zupt_flag,
            "maneuver_state": maneuver_state
        }
