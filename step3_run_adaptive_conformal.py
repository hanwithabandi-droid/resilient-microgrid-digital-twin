"""
STEP 3: Run Adaptive Conformal Inference (ACI) to Generate Guaranteed 90% Intervals.
Click the Play button (top right in VS Code) to run this script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.conformal.base_forecaster import HarmonicRidgeForecaster
from src.conformal.aci import AdaptiveConformalInference
from src.evaluation.metrics import calculate_empirical_coverage, calculate_mean_interval_width

print("Step 3: Running Adaptive Conformal Inference (ACI)...")

# Load data
df = pd.read_csv("data/processed/microgrid_benchmark_15min.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
hours = df["timestamp"].dt.hour.values + df["timestamp"].dt.minute.values / 60.0
dow = df["timestamp"].dt.dayofweek.values
pv_all = df["pv_power_kw"].values

# 1. Train base ridge model on first 5,000 steps
print("Fitting autoregressive ridge model...")
forecaster = HarmonicRidgeForecaster(alpha_reg=2.0, n_lags=4)
forecaster.fit(pv_all[:5000], hours[:5000], dow[:5000])

# 2. Seed ACI with calibration residuals
print("Calibrating ACI quantiles (target: 90%)...")
calib_preds = [forecaster.predict_one_step(pv_all[i-4:i], hours[i], dow[i]) for i in range(5000, 7000)]
calib_res = np.abs(pv_all[5000:7000] - calib_preds)

aci = AdaptiveConformalInference(target_coverage=0.90, gamma=0.01, window_size=200, min_quantile=1.0)
aci.calibrate_initial(calib_res)

# 3. Test on an unseen 5-day period (480 timesteps)
test_start = 8000
test_len = 480
y_test = pv_all[test_start : test_start + test_len]
h_test = hours[test_start : test_start + test_len]
d_test = dow[test_start : test_start + test_len]

preds, lows, highs = [], [], []
lags = list(pv_all[test_start-4 : test_start])

for t in range(test_len):
    y_hat = forecaster.predict_one_step(lags, h_test[t], d_test[t])
    low, high, _ = aci.get_prediction_interval(y_hat)
    
    aci.update(y_test[t], y_hat, low, high)
    lags.pop(0)
    lags.append(y_test[t])
    
    preds.append(y_hat)
    lows.append(low)
    highs.append(high)

# Calculate statistical metrics
cov = calculate_empirical_coverage(y_test, lows, highs)
width = calculate_mean_interval_width(lows, highs)

print(f"Target coverage:    90.0%")
print(f"Empirical coverage: {cov*100:.2f}%")
print(f"Average width:      {width:.2f} kW")

# Plot Conformal Bands
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="Actual PV (kW)", color="#2c3e50", lw=2)
plt.plot(preds, label="Forecast", color="#2980b9", ls="--", lw=1.2)
plt.fill_between(range(test_len), lows, highs, color="#3498db", alpha=0.35, label=f"90% Prediction Band (Coverage: {cov*100:.1f}%)")
plt.title("Adaptive Conformal Prediction Intervals", fontweight="bold")
plt.xlabel("15-Minute Intervals")
plt.ylabel("PV Power (kW)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("Step 3 complete.")
