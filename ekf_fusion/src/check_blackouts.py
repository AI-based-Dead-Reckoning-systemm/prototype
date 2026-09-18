import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from loader import load_trip_for_fusion

def analyze_blackouts():
    trips = ["S1", "S2", "S3c", "M", "Vfa01"]
    base_path = os.path.abspath(os.path.join(current_dir, "../.."))

    results = []

    print("Starting blackout analysis...\n")

    for trip in trips:
        print(f"--- Analyzing Trip: {trip} ---")
        try:
            df = load_trip_for_fusion(trip, base_path=base_path)
        except Exception as e:
            print(f"Skipping Trip {trip}: Error loading - {e}\n")
            continue
            
        total_rows = len(df)
        is_na = (df['gps_satellites'] <= 3).values
        blackout_rows = is_na.sum()
        
        longest_duration = 0.0
        longest_range = "N/A"
        max_acc = 0.0
        
        if blackout_rows > 0:
            na_indices = np.where(is_na)[0]
            
            # Group into contiguous segments
            segments = []
            start_idx = na_indices[0]
            prev_idx = na_indices[0]
            
            for i in range(1, len(na_indices)):
                curr_idx = na_indices[i]
                if curr_idx != prev_idx + 1:
                    # End of current segment
                    segments.append((start_idx, prev_idx))
                    start_idx = curr_idx
                prev_idx = curr_idx
            # Add last segment
            segments.append((start_idx, prev_idx))
            
            # Find longest segment
            for s_idx, e_idx in segments:
                s_time = df.iloc[s_idx]['timestamp_s']
                e_time = df.iloc[e_idx]['timestamp_s']
                duration = e_time - s_time
                if duration >= longest_duration:
                    longest_duration = duration
                    longest_range = f"{s_time:.2f}s to {e_time:.2f}s"
                    max_acc = df.iloc[s_idx:e_idx+1]['gps_accuracy_m'].max()
                    
            print(f"Total Rows: {total_rows}")
            print(f"GNSS Degraded Rows: {blackout_rows}")
            print(f"Longest Degraded Segment: {longest_range} (Duration: {longest_duration:.2f}s, Peak Acc: {max_acc}m)\n")
        else:
            print(f"Total Rows: {total_rows}")
            print(f"GNSS Blackout Rows: 0\n")
            
        results.append({
            "trip_id": trip,
            "total_rows": total_rows,
            "blackout_rows": blackout_rows,
            "longest_blackout_duration_s": longest_duration
        })

    print("================ FINAL SUMMARY TABLE ================")
    print(f"{'trip_id':<10} | {'total_rows':<12} | {'blackout_rows':<15} | {'longest_blackout_duration_s'}")
    print("-" * 75)
    for res in results:
        print(f"{res['trip_id']:<10} | {res['total_rows']:<12} | {res['blackout_rows']:<15} | {res['longest_blackout_duration_s']:.2f}")
    print("=====================================================")

if __name__ == "__main__":
    analyze_blackouts()
