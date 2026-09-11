"""
fallback.py - Fallback Dispatcher & Physics-Based Kinematic Estimator.
When the AI model confidence drops below an empirical threshold (or high variance),
falls back to forward-integrated acceleration anchored by ZUPT, and logs active_source.
"""

import numpy as np
import pandas as pd

class PhysicsFallbackDispatcher:
    """
    Dispatches between AI Speed Estimator predictions and Kinematic ZUPT Integration.
    Logs active_source at every 10Hz step for debugging, observability, and UI timeline strips.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.35, # Trigger fallback if confidence is lower than this
        max_variance_threshold: float = 4.0  # m²/s² max allowed variance
    ):
        self.confidence_threshold = confidence_threshold
        self.max_variance_threshold = max_variance_threshold

    def apply_fallback(
        self,
        model_pred_df: pd.DataFrame,
        aligned_df: pd.DataFrame,
        window_end_indices: np.ndarray,
        dt: float = 0.1
    ) -> pd.DataFrame:
        """
        Applies fallback logic across time:
        - When model confidence >= threshold: uses 'ml_model'
        - When model confidence < threshold: uses 'physics_fallback'
        """
        n = len(model_pred_df)
        
        ml_speed = model_pred_df["predicted_speed_mps"].values
        ml_var = model_pred_df["speed_variance"].values
        ml_conf = model_pred_df["speed_confidence"].values
        
        afwd = aligned_df["accel_vehicle_fwd"].values[window_end_indices]
        zupt = aligned_df["zupt_flag"].values[window_end_indices]
        maneuver = aligned_df["maneuver_state"].values[window_end_indices]
        timestamps = aligned_df["timestamp_s"].values[window_end_indices]

        final_speed = np.zeros(n, dtype=np.float64)
        final_var = np.zeros(n, dtype=np.float64)
        final_conf = np.zeros(n, dtype=np.float64)
        active_source = []

        kinematic_v = 0.0

        for i in range(n):
            is_low_conf = (ml_conf[i] < self.confidence_threshold) or (ml_var[i] > self.max_variance_threshold)
            
            # Update running kinematic forward integration
            if zupt[i] or maneuver[i] == "stationary":
                kinematic_v = 0.0
            else:
                kinematic_v = max(0.0, kinematic_v + afwd[i] * dt)

            if is_low_conf and not zupt[i]:
                # Dispatch to physics fallback
                final_speed[i] = kinematic_v
                # Physics fallback has higher variance / lower confidence
                final_var[i] = max(ml_var[i], 2.5)
                final_conf[i] = 0.30
                active_source.append("physics_fallback")
            else:
                # Dispatch to ML model
                final_speed[i] = ml_speed[i]
                final_var[i] = ml_var[i]
                final_conf[i] = ml_conf[i]
                active_source.append("ml_model")
                # Synchronize kinematic integrator with trusted ML prediction
                kinematic_v = ml_speed[i]

        output_df = pd.DataFrame({
            "timestamp_s": np.round(timestamps, 4),
            "estimated_speed_mps": np.round(final_speed, 3),
            "speed_variance": np.round(final_var, 4),
            "speed_confidence": np.round(final_conf, 3),
            "active_source": active_source,
            "zupt_flag": zupt,
            "maneuver_state": maneuver
        })

        return output_df
