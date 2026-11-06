"""
STEP 1: Load and Inspect the 15-Minute Microgrid Benchmark Dataset.
Click the Play button (top right in VS Code) to run this script.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

print("Step 1: Inspecting microgrid solar and load benchmark data...")

csv_path = os.path.join("data", "processed", "microgrid_benchmark_15min.csv")
if not os.path.exists(csv_path):
    print("Dataset not found. Generating now...")
    from data.download_data import create_benchmark_dataset
    df = create_benchmark_dataset()
else:
    df = pd.read_csv(csv_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"Loaded {csv_path}:")
print(f"  Total observations: {len(df):,} intervals (365 days @ 15-min resolution)")
print(f"  Peak PV AC power:   {df['pv_power_kw'].max():.2f} kW")
print(f"  Peak load demand:   {df['load_power_kw'].max():.2f} kW")
print(f"  Max net deficit:    {df['net_load_kw'].max():.2f} kW")
print("\nFirst 5 rows:")
print(df[["timestamp", "poa_irradiance_wm2", "pv_power_kw", "load_power_kw", "net_load_kw"]].head())

# Plot a representative 3-day sample
sample = df.iloc[96*180 : 96*183].reset_index(drop=True)
plt.figure(figsize=(10, 4.5))
plt.plot(sample["pv_power_kw"], label="PV Output (kW)", color="#e67e22", lw=2)
plt.plot(sample["load_power_kw"], label="Building Load (kW)", color="#2980b9", lw=2)
plt.title("Sample 3-Day Microgrid Generation & Load Profiles", fontweight="bold")
plt.xlabel("15-Minute Intervals")
plt.ylabel("Power (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Step 1 complete.")
