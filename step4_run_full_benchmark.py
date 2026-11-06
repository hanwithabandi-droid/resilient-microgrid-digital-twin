"""
STEP 4: Run the Full Comparative Benchmark Across 4 Telemetry Regimes.
Compares Deterministic vs Gaussian vs Proposed Conformal Risk Controller.
Click the Play button (top right in VS Code) to run this script.
"""

import os
import pandas as pd
from run_experiments import run_all_benchmarks

print("Step 4: Running comparative benchmark across 4 telemetry regimes...")

# Executes the master experiment suite
run_all_benchmarks()

# Load and print the generated benchmark summary CSV
summary_csv = "results/benchmark_summary.csv"
if os.path.exists(summary_csv):
    df_sum = pd.read_csv(summary_csv)
    print("\nBenchmark summary:")
    print(df_sum.to_string(index=False))

print("\nStep 4 complete.")
