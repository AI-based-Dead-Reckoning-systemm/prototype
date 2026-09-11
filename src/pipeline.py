"""
pipeline.py - End-to-end AI Speed Estimation & Fallback Pipeline.
Integrates feature extraction, model inference, physics bounding, and fallback dispatching.
"""

import os
import pandas as pd
import numpy as np

from src.features import extract_window_features
from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import evaluate_predictions, plot_speed_evaluation

def process_speed_estimation(
    aligned_df: pd.DataFrame,
    model: SpeedEstimatorModel,
    gt_df: pd.DataFrame = None,
    output_csv_path: str = None,
    output_plot_path: str = None,
    trip_name: str = "Trip",
    window_size: int = 20, # 2.0 seconds at 10 Hz
    step_size: int = 1     # 0.1s hop
) -> dict:
    """
    Runs full inference pipeline:
      1. Feature extraction over 2.0s sliding windows
      2. AI Speed prediction + ensemble variance estimation
      3. Physics-informed bounding (slew rate, non-negativity, ZUPT lock)
      4. Fallback dispatching (ML model vs Kinematic integration)
      5. Optional Ground Truth CAN evaluation & plot generation
    """
    # 1. Extract multi-domain features
    features_df, window_end_indices = extract_window_features(
        aligned_df,
        window_size=window_size,
        step_size=step_size
    )

    # 2. Predict with Physics Constraints
    model_preds_df = model.predict_with_physics(
        features_df=features_df,
        aligned_df=aligned_df,
        window_end_indices=window_end_indices
    )

    # 3. Fallback Dispatching
    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
    final_estimates_df = dispatcher.apply_fallback(
        model_pred_df=model_preds_df,
        aligned_df=aligned_df,
        window_end_indices=window_end_indices
    )

    # 4. Save CSV Output Contract
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        final_estimates_df.to_csv(output_csv_path, index=False)

    # 5. Ground Truth Evaluation
    eval_results = {}
    if gt_df is not None:
        gt_speed = gt_df["gt_speed_mps"].values[window_end_indices]
        eval_results = evaluate_predictions(final_estimates_df, gt_speed, trip_name=trip_name)
        
        if output_plot_path:
            plot_speed_evaluation(
                final_estimates_df,
                gt_speed,
                trip_name=trip_name,
                output_png_path=output_plot_path
            )

    return {
        "estimates": final_estimates_df,
        "features": features_df,
        "window_end_indices": window_end_indices,
        "evaluation_results": eval_results
    }
