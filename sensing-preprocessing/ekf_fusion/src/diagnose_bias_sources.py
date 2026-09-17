import os
import math
import numpy as np
import pandas as pd

from loader import load_trip_for_fusion

def calculate_stats(subset_df, name):
    subset_df = subset_df.dropna(subset=['gps_speed_mps', 'estimated_speed_mps'])
    if subset_df.empty:
        print(f"\n--- {name} ---")
        print("No data rows match this criteria.")
        return None
        
    err = subset_df['estimated_speed_mps'] - subset_df['gps_speed_mps']
    mse = err.mean()
    mae = err.abs().mean()
    
    # Linear regression forced through 0: k = sum(x*y) / sum(x^2)
    x = subset_df['gps_speed_mps'].values
    y = subset_df['estimated_speed_mps'].values
    
    # filter out zeros in x to avoid division by zero
    valid_mask = x > 0.1
    if valid_mask.sum() > 0:
        x_v = x[valid_mask]
        y_v = y[valid_mask]
        k = np.sum(x_v * y_v) / np.sum(x_v ** 2)
    else:
        k = float('nan')
        
    print(f"\n--- {name} ---")
    print(f"Data points: {len(subset_df)}")
    print(f"Mean Signed Error (est - gps): {mse:.4f} m/s")
    print(f"Mean Absolute Error: {mae:.4f} m/s")
    print(f"Scale Factor (slope through 0): {k:.4f}")
    
    return mse, mae, k

def main():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print(f"Loading trip 'S1'...")
    df = load_trip_for_fusion("S1", base_path=base_path)
    
    if 'zupt_flag_x' in df.columns:
        zupt_col = 'zupt_flag_x'
    elif 'zupt_flag' in df.columns:
        zupt_col = 'zupt_flag'
    else:
        zupt_col = None
        
    if 'maneuver_state_x' in df.columns:
        man_col = 'maneuver_state_x'
    elif 'maneuver_state' in df.columns:
        man_col = 'maneuver_state'
    else:
        man_col = None

    # GOOD GNSS mask
    good_gnss = (df['gps_satellites'] > 15) & (df['gps_accuracy_m'] < 3.0)
    
    # 1. SPEED ESTIMATOR BIAS/SCALE CHECK (Straight/Gentle)
    mask_straight = good_gnss & (df[man_col].isin(['straight', 'gentle_curve']))
    straight_subset = df[mask_straight].dropna(subset=['gps_speed_mps', 'estimated_speed_mps'])
    straight_stats = calculate_stats(straight_subset, "1. SPEED ESTIMATOR (Straight/Gentle Curve)")

    if straight_stats:
        mse1, mae1, k1 = straight_stats
        
        # 1b. TEST HYPOTHESES
        print(f"\n--- 1b. SPEED ESTIMATOR HYPOTHESES TESTS ---")
        
        # Hypothesis A: km/h conversion
        speed_conv = straight_subset['estimated_speed_mps'] / 3.6
        err_conv = speed_conv - straight_subset['gps_speed_mps']
        mse_conv = err_conv.mean()
        mae_conv = err_conv.abs().mean()
        
        x = straight_subset['gps_speed_mps'].values
        y_conv = speed_conv.values
        valid_mask = x > 0.1
        if valid_mask.sum() > 0:
            x_v = x[valid_mask]
            y_c_v = y_conv[valid_mask]
            k_conv = np.sum(x_v * y_c_v) / np.sum(x_v ** 2)
        else:
            k_conv = float('nan')
            
        print("Hypothesis A: km/h Conversion (Divide by 3.6)")
        print(f"  Mean Signed Error  : {mse_conv:.4f} m/s")
        print(f"  Mean Absolute Error: {mae_conv:.4f} m/s")
        print(f"  Scale Factor       : {k_conv:.4f}")
        
        if abs(k_conv - 1.0) <= 0.05 and mae_conv < mae1 * 0.5:
            print("  -> CONFIRMED: The data strongly fits the km/h hypothesis.")
        else:
            print("  -> NOT CONFIRMED: The ~3.4x error needs another explanation.")
            
        # Hypothesis B: Additive offset
        speed_add = straight_subset['estimated_speed_mps'] - mse1
        err_add = speed_add - straight_subset['gps_speed_mps']
        mae_add = err_add.abs().mean()
        
        print("\nHypothesis B: Additive Constant Offset")
        print(f"  Mean Absolute Error: {mae_add:.4f} m/s (subtracting {mse1:.4f} m/s)")
        
        print("\nComparison of MAE:")
        print(f"  Original MAE : {mae1:.4f} m/s")
        print(f"  km/h MAE     : {mae_conv:.4f} m/s")
        print(f"  Additive MAE : {mae_add:.4f} m/s")

    # 2. ACCELEROMETER BIAS CHECK
    print(f"\n--- 2. ACCELEROMETER BIAS CHECK (ZUPT=True) ---")
    if zupt_col:
        zupt_mask = df[zupt_col] == True
        zupt_df = df[zupt_mask].dropna(subset=['accel_vehicle_fwd'])
        if not zupt_df.empty:
            mean_accel = zupt_df['accel_vehicle_fwd'].mean()
            std_accel = zupt_df['accel_vehicle_fwd'].std()
            print(f"Data points: {len(zupt_df)}")
            print(f"Mean accel_vehicle_fwd: {mean_accel:.6f} m/s^2")
            print(f"Std accel_vehicle_fwd:  {std_accel:.6f} m/s^2")
        else:
            print("No ZUPT=True rows found.")
    else:
        print("No ZUPT column found.")

    # 3. GYRO BIAS CHECK
    print(f"\n--- 3. GYRO BIAS CHECK (ZUPT=True) ---")
    if zupt_col and not zupt_df.empty:
        mean_gyro = zupt_df['gyro_vehicle_yaw'].mean()
        std_gyro = zupt_df['gyro_vehicle_yaw'].std()
        print(f"Data points: {len(zupt_df)}")
        print(f"Mean gyro_vehicle_yaw: {mean_gyro:.6f} rad/s ({math.degrees(mean_gyro):.6f} deg/s)")
        print(f"Std gyro_vehicle_yaw:  {std_gyro:.6f} rad/s")
    else:
        print("No ZUPT data available.")

    # 4. SPEED ESTIMATOR BEHAVIOR (Sharp Turns)
    mask_sharp = good_gnss & (df[man_col] == 'sharp_turn')
    sharp_stats = calculate_stats(df[mask_sharp], "4. SPEED ESTIMATOR (Sharp Turns)")

    # 5. SUMMARY COMPARISON
    print(f"\n--- 5. SUMMARY COMPARISON ---")
    if straight_stats and sharp_stats:
        mse1, mae1, k1 = straight_stats
        mse4, mae4, k4 = sharp_stats
        
        print("Error Metric          | Straight/Gentle | Sharp Turn")
        print("----------------------|-----------------|-----------")
        print(f"Mean Signed Error     | {mse1:14.4f}  | {mse4:9.4f}")
        print(f"Mean Absolute Error   | {mae1:14.4f}  | {mae4:9.4f}")
        print(f"Scale Factor (slope)  | {k1:14.4f}  | {k4:9.4f}")
        
        diff_mae = mae4 - mae1
        pct_diff_mae = (diff_mae / mae1) * 100.0 if mae1 != 0 else float('inf')
        print(f"\nObservation: Sharp turn Mean Absolute Error is {abs(pct_diff_mae):.1f}% {'worse' if pct_diff_mae > 0 else 'better'} than straight/gentle.")
    else:
        print("Cannot compare, missing data for one or both categories.")

if __name__ == "__main__":
    main()
