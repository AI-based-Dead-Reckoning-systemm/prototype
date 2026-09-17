import os
import sys
import pandas as pd
import numpy as np

# Add sensing-preprocessing root to path to import schemas
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from schemas import AlignedSample, SpeedEstimate

def load_trip_for_fusion(trip_id: str, base_path: str = "..") -> pd.DataFrame:
    """
    Loads AlignedSample and SpeedEstimates CSV files for a given trip ID,
    validates them using schemas.py, and merges them on timestamp_s.
    """
    aligned_path = os.path.join(base_path, "preprocessing", "output", f"AlignedSample_{trip_id}.csv")
    
    speed_path = os.path.join(base_path, "speed_estimation", "output", f"SpeedEstimates_{trip_id}.csv")
    if not os.path.exists(speed_path):
        speed_path_alt = os.path.join(base_path, "speed_estimation", "output", f"SpeedEstimates_Trip_{trip_id}.csv")
        if os.path.exists(speed_path_alt):
            speed_path = speed_path_alt

    # 1. Load AlignedSample
    aligned_raw = pd.read_csv(aligned_path)
    aligned_dicts = [AlignedSample.from_csv_row(row).to_dict() for row in aligned_raw.to_dict('records')]
    aligned_df = pd.DataFrame(aligned_dicts)
    
    # 2. Load SpeedEstimates
    speed_raw = pd.read_csv(speed_path)
    speed_dicts = [SpeedEstimate.from_csv_row(row).to_dict() for row in speed_raw.to_dict('records')]
    speed_df = pd.DataFrame(speed_dicts)
    
    # 3. Sort DataFrames and merge with merge_asof
    aligned_df = aligned_df.sort_values("timestamp_s")
    speed_df = speed_df.sort_values("timestamp_s")
    
    merged_df = pd.merge_asof(
        aligned_df, 
        speed_df, 
        on="timestamp_s", 
        direction="nearest", 
        tolerance=0.05
    )
    
    unmatched_rows = merged_df['estimated_speed_mps'].isna().sum()
    print(f"Rows from aligned_df with NO match within tolerance: {unmatched_rows}")
    
    # 4. Return the merged DataFrame
    return merged_df

if __name__ == "__main__":
    # Adjust base_path depending on where the script is executed
    base_path = os.path.abspath(os.path.join(current_dir, "../.."))
    
    print(f"Loading trip 'S1' from base path: {base_path}...")
    merged_df = load_trip_for_fusion("S1", base_path=base_path)
    
    print("\nMerged DataFrame shape:", merged_df.shape)
    print("Merged DataFrame columns:", list(merged_df.columns))
    print("\nFirst 10 rows:")
    print(merged_df.head(10))
    
    blackout_rows = merged_df[merged_df['gps_lat'].isna()]
    print(f"\nRows with GNSS blackout (gps_lat is NaN): {len(blackout_rows)}")
    
    if len(blackout_rows) > 0:
        is_na = merged_df['gps_lat'].isna().values
        na_indices = np.where(is_na)[0]
        
        start_idx = na_indices[0]
        end_idx = 0
        while end_idx + 1 < len(na_indices) and na_indices[end_idx + 1] == na_indices[end_idx] + 1:
            end_idx += 1
            
        real_start_idx = na_indices[0]
        real_end_idx = na_indices[end_idx]
        
        start_time = merged_df.iloc[real_start_idx]['timestamp_s']
        end_time = merged_df.iloc[real_end_idx]['timestamp_s']
        print(f"First contiguous GNSS blackout segment: from {start_time:.2f}s to {end_time:.2f}s")
