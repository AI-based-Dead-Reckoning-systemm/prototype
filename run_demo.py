"""
run_demo.py - Quickstart Demonstration for AI Speed Estimator Module.
Loads the trained Random Forest model with ensemble uncertainty, extracts sliding window features,
enforces physics constraints & ZUPT anchoring, applies fallback dispatching, and produces SpeedEstimates_*.csv.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Setup paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(1, os.path.join(CURRENT_DIR, "src"))

from src.model import SpeedEstimatorModel
from src.pipeline import process_speed_estimation

def run_demo(input_aligned_csv: str, output_csv: str = "output/SpeedEstimates_demo.csv", model_path: str = "models/speed_estimator_rf.pkl"):
    print("=" * 75)
    print("AI SPEED ESTIMATOR — DEMO INFERENCE RUNNER")
    print("=" * 75)
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Run train_and_evaluate.py first.")
        sys.exit(1)
        
    print(f"Loading trained AI Speed Estimator from: {model_path}")
    model = SpeedEstimatorModel.load(model_path)
    
    print(f"Loading aligned input data from: {input_aligned_csv}")
    aligned_df = pd.read_csv(input_aligned_csv)
    print(f"Input records: {len(aligned_df)} rows at 10 Hz ({len(aligned_df)*0.1:.1f} seconds)")
    
    print("\nRunning inference pipeline (Features -> ML Regressor -> Physics Bounds -> Fallback)...")
    res = process_speed_estimation(
        aligned_df=aligned_df,
        model=model,
        output_csv_path=output_csv,
        trip_name="Demo_Trip"
    )
    
    estimates_df = res["estimates"]
    print(f"\n[OK] Speed estimation completed successfully!")
    print(f"     Output saved to: {output_csv}")
    print(f"     Output columns: {list(estimates_df.columns)}")
    print(f"     Mean predicted speed: {estimates_df['estimated_speed_mps'].mean():.2f} m/s")
    print(f"     Max predicted speed:  {estimates_df['estimated_speed_mps'].max():.2f} m/s")
    print(f"     ML Model Usage:       {(estimates_df['active_source'] == 'ml_model').mean()*100:.1f}%")
    print(f"     Physics Fallback:     {(estimates_df['active_source'] == 'physics_fallback').mean()*100:.1f}%")
    print("\nFirst 5 predicted timesteps:")
    print(estimates_df.head(5).to_string(index=False))
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Speed Estimator on an AlignedSample CSV file.")
    parser.add_argument(
        "--input",
        type=str,
        default="../sih_idr_preprocessing/output/AlignedSample_S1.csv",
        help="Path to input AlignedSample CSV (from Preprocessing module)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/SpeedEstimates_demo.csv",
        help="Path to save output SpeedEstimates CSV"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/speed_estimator_rf.pkl",
        help="Path to trained speed estimator model pickle"
    )
    args = parser.parse_args()
    run_demo(args.input, args.output, args.model)
