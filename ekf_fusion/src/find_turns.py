import pandas as pd
import numpy as np
from loader import load_trip_for_fusion

base_path = "."
trip_id = "M"
df = load_trip_for_fusion(trip_id, base_path=base_path)

# calculate heading changes
w = df['gyro_vehicle_yaw'].values
dt = np.diff(df['timestamp_s'].values, prepend=df['timestamp_s'].values[0])
dt[0] = 0.1
heading_change = w * dt * 180.0 / np.pi

# find windows of 30s with large absolute heading change
window = 300 # 30s at 10Hz
max_change = 0
best_start = 0

for i in range(len(df) - window):
    change = np.sum(heading_change[i:i+window])
    if abs(change) > abs(max_change):
        max_change = change
        best_start = df['timestamp_s'].values[i]

print(f"Largest turn in {trip_id} is {max_change:.2f} deg over 30s starting at {best_start:.2f}s")
