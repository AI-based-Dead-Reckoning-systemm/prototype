"""
model.py - Unified AI Speed Estimator Model & Inference Core.
Uses an ensemble Random Forest regressor with tree-level prediction variance
as an uncertainty/confidence metric, combined with physics-informed bounds.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

class SpeedEstimatorModel:
    """
    Unified AI Speed Estimator for Smartphone-Based Dead Reckoning.
    Predicts forward speed (m/s) + uncertainty/confidence from windowed IMU features.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 14,
        min_samples_leaf: int = 4,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_names = None
        self.is_trained = False

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Trains the random forest regressor on feature matrix X and ground truth speed y (m/s).
        """
        self.feature_names = list(X.columns)
        self.model.fit(X.values, y)
        self.is_trained = True
        return self

    def predict_raw(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts mean speed and ensemble variance across individual decision trees.
        Returns:
          - pred_mean (m/s)
          - pred_var (m²/s²)
          - confidence_score (0.0 to 1.0)
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call fit() or load() first.")

        X_mat = X[self.feature_names].values
        # Predict with all individual trees in parallel
        # Shape: (n_estimators, n_samples)
        all_tree_preds = np.array([tree.predict(X_mat) for tree in self.model.estimators_])
        
        pred_mean = np.mean(all_tree_preds, axis=0)
        pred_var = np.var(all_tree_preds, axis=0) # Ensemble variance
        pred_std = np.sqrt(pred_var)
        
        # Confidence score: 1.0 for zero variance, decaying with uncertainty
        # Confidence = 1 / (1 + std / 1.5)
        confidence = 1.0 / (1.0 + (pred_std / 1.5))
        
        return pred_mean, pred_var, confidence

    def predict_with_physics(
        self,
        features_df: pd.DataFrame,
        aligned_df: pd.DataFrame,
        window_end_indices: np.ndarray,
        max_accel_slew: float = 5.0, # m/s² max plausible slew
        dt: float = 0.1
    ) -> pd.DataFrame:
        """
        Inference with physics-informed constraints:
          1. Clip speed: [0.0, 45.0] m/s (0 to 162 km/h)
          2. ZUPT Stationary lock: If zupt_flag is True, snap to 0.0 m/s with high confidence
          3. Kinematic slew rate limiting between consecutive 10Hz steps
        """
        pred_mean, pred_var, confidence = self.predict_raw(features_df)
        n = len(pred_mean)
        
        zupt_flags = aligned_df["zupt_flag"].values[window_end_indices]
        maneuver_states = aligned_df["maneuver_state"].values[window_end_indices]
        timestamps = aligned_df["timestamp_s"].values[window_end_indices]
        
        constrained_speed = np.zeros(n, dtype=np.float64)
        constrained_var = pred_var.copy()
        constrained_conf = confidence.copy()

        prev_v = 0.0
        max_dv = max_accel_slew * dt # Max allowed change per 0.1s step (e.g. 0.5 m/s)

        for i in range(n):
            v_raw = pred_mean[i]
            
            # Rule 1: Non-negative and highway upper ceiling
            v_bounded = np.clip(v_raw, 0.0, 45.0)
            
            # Rule 2: ZUPT / Stationary override
            if zupt_flags[i] or maneuver_states[i] == "stationary":
                v_bounded = 0.0
                constrained_var[i] = 0.005 # Minimal variance during stationary
                constrained_conf[i] = 0.99
            
            # Rule 3: Kinematic acceleration slew rate limiting
            if i > 0:
                dv = v_bounded - prev_v
                if abs(dv) > max_dv:
                    v_bounded = prev_v + np.sign(dv) * max_dv
            
            constrained_speed[i] = v_bounded
            prev_v = v_bounded

        results_df = pd.DataFrame({
            "timestamp_s": np.round(timestamps, 4),
            "predicted_speed_mps": np.round(constrained_speed, 3),
            "speed_variance": np.round(constrained_var, 4),
            "speed_confidence": np.round(constrained_conf, 3),
            "raw_model_speed": np.round(pred_mean, 3)
        })

        return results_df

    def save(self, filepath: str):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "params": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "min_samples_leaf": self.min_samples_leaf,
                    "random_state": self.random_state
                }
            }, f)

    @classmethod
    def load(cls, filepath: str) -> "SpeedEstimatorModel":
        """Loads model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.is_trained = True
        return instance
