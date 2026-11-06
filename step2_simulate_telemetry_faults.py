"""
STEP 2: Simulate IoT Sensor Telemetry Faults (Packet Drops, Latency, Blackouts).
Click the Play button (top right in VS Code) to run this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.corruption.telemetry_faults import TelemetryChannel

print("Step 2: Simulating telemetry degradation (packet drop, latency, blackout)...")

# Load data
df = pd.read_csv("data/processed/microgrid_benchmark_15min.csv")
sample_slice = df.iloc[96*80 : 96*83].reset_index(drop=True)
clean_pv = sample_slice["pv_power_kw"].values

# Create telemetry channel: 25% packet drop, 30-min latency (2 steps), 4% noise
channel = TelemetryChannel(packet_drop_rate=0.25, latency_steps=2, noise_std_fraction=0.04, random_seed=42)

observed_pv = []
dropped_flags = []

# Simulate 3-hour cellular blackout on day 2 (steps 96 to 108)
for i, val in enumerate(clean_pv):
    in_blackout = (96 <= i <= 108)
    obs, dropped = channel.transmit(val, is_blackout=in_blackout)
    observed_pv.append(obs)
    dropped_flags.append(dropped)

observed_pv = np.array(observed_pv)
dropped_flags = np.array(dropped_flags)

drop_count = int(np.sum(dropped_flags))
print(f"Total steps: {len(clean_pv)}")
print(f"Dropped packets: {drop_count} ({drop_count/len(clean_pv)*100:.1f}%)")
print(f"Latency staleness: 30 minutes")
print(f"Simulated blackout: steps 96-108")

# Visualize Ground Truth vs Corrupted Stream
plt.figure(figsize=(11, 5))
plt.plot(clean_pv, label="Ground Truth PV (Physical)", color="#2c3e50", lw=2)
plt.plot(observed_pv, label="Observed Telemetry (Corrupted)", color="#e74c3c", ls="--", lw=1.5)
dropped_idx = np.where(dropped_flags)[0]
plt.scatter(dropped_idx, clean_pv[dropped_idx], color="#c0392b", s=25, zorder=5, label="Dropped / Stale Hold")
plt.axvspan(96, 108, color="#e74c3c", alpha=0.15, label="3-Hour Outage Window")

plt.title("Telemetry Channel Degradation", fontweight="bold")
plt.xlabel("15-Minute Intervals")
plt.ylabel("PV Power (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Step 2 complete.")
