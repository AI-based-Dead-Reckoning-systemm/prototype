import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from loader import load_trip_for_fusion

def inject_synthetic_blackout(df: pd.DataFrame, start_time: float, duration_s: float) -> pd.DataFrame:
    df_mod = df.copy()
    end_time = start_time + duration_s
    mask = (df_mod['timestamp_s'] >= start_time) & (df_mod['timestamp_s'] <= end_time)
    df_mod.loc[mask, 'gps_satellites'] = 0
    df_mod.loc[mask, 'gps_accuracy_m'] = 999.0
    return df_mod

if __name__ == "__main__":
    base_path = os.path.abspath(os.path.join(current_dir, "../.."))
    print("Loading trip 'S1'...")
    df = load_trip_for_fusion("S1", base_path=base_path)
    
    start_time = 200.0
    duration_s = 40.0
    
    print(f"\nInjecting blackout from {start_time}s to {start_time + duration_s}s...")
    df_mod = inject_synthetic_blackout(df, start_time, duration_s)
    
    mask = (df['timestamp_s'] >= start_time) & (df['timestamp_s'] <= start_time + duration_s)
    modified_rows = mask.sum()
    
    if 'maneuver_state_x' in df.columns:
        maneuvers = df.loc[mask, 'maneuver_state_x'].unique()
    elif 'maneuver_state' in df.columns:
        maneuvers = df.loc[mask, 'maneuver_state'].unique()
    else:
        maneuvers = []
        
    print(f"Modified {modified_rows} rows.")
    print(f"Maneuver states present during the blackout window: {list(maneuvers)}")
