"""
Master Automated Benchmark Experiment Suite.
Executes reproducible, rigorous Monte-Carlo and sequential comparative evaluations
across Deterministic, Gaussian, and Conformal Risk-Constrained microgrid controllers
under multiple verified telemetry degradation regimes.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import project modules
from src.physics.pv_model import PVSystem
from src.physics.battery_model import BatteryStorage
from src.corruption.telemetry_faults import TelemetryChannel
from src.conformal.base_forecaster import HarmonicRidgeForecaster
from src.conformal.aci import AdaptiveConformalInference
from src.control.deterministic import DeterministicController
from src.control.gaussian import GaussianController
from src.control.conformal_risk import ConformalRiskController
from src.evaluation.metrics import (
    calculate_empirical_coverage,
    calculate_mean_interval_width,
    calculate_winkler_score,
    calculate_operational_kpis
)

def run_simulation(data_path="data/processed/microgrid_benchmark_15min.csv",
                   test_days=30,
                   packet_drop_rate=0.20,
                   latency_steps=2,
                   noise_std_frac=0.03,
                   blackout_window=None,
                   target_coverage=0.90,
                   seed=42):
    """
    Executes a head-to-head simulation run on real time series data.
    """
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Split into Calibration (first 6 months) and Test evaluation (selected test slice)
    calib_steps = 15000
    eval_steps = test_days * 96 # 96 steps per day
    
    times = df["timestamp"].values
    hours = df["timestamp"].dt.hour.values + df["timestamp"].dt.minute.values / 60.0
    dayofweek = df["timestamp"].dt.dayofweek.values
    
    pv_actual = df["pv_power_kw"].values
    load_actual = df["load_power_kw"].values
    
    # 1. Train base autoregressive forecasters on calibration data
    pv_forecaster = HarmonicRidgeForecaster(alpha_reg=2.0, n_lags=4)
    pv_forecaster.fit(pv_actual[:calib_steps], hours[:calib_steps], dayofweek[:calib_steps])
    
    load_forecaster = HarmonicRidgeForecaster(alpha_reg=2.0, n_lags=4)
    load_forecaster.fit(load_actual[:calib_steps], hours[:calib_steps], dayofweek[:calib_steps])
    
    # 2. Seed ACI with historical calibration residuals
    pv_calib_preds = [pv_forecaster.predict_one_step(pv_actual[i-4:i], hours[i], dayofweek[i]) for i in range(1000, 2000)]
    pv_calib_res = np.abs(pv_actual[1000:2000] - pv_calib_preds)
    
    load_calib_preds = [load_forecaster.predict_one_step(load_actual[i-4:i], hours[i], dayofweek[i]) for i in range(1000, 2000)]
    load_calib_res = np.abs(load_actual[1000:2000] - load_calib_preds)
    
    pv_aci = AdaptiveConformalInference(target_coverage=target_coverage, gamma=0.008, window_size=200, min_quantile=1.0)
    pv_aci.calibrate_initial(pv_calib_res)
    
    load_aci = AdaptiveConformalInference(target_coverage=target_coverage, gamma=0.008, window_size=200, min_quantile=1.0)
    load_aci.calibrate_initial(load_calib_res)
    
    # 3. Setup Telemetry Channel
    channel_pv = TelemetryChannel(packet_drop_rate, latency_steps, noise_std_frac, random_seed=seed)
    channel_load = TelemetryChannel(packet_drop_rate, latency_steps, noise_std_frac, random_seed=seed+10)
    
    # 4. Instantiate 3 Batteries for the 3 controllers
    b_det = BatteryStorage(capacity_kwh=100.0, soc_min=0.20, initial_soc=0.60)
    b_gauss = BatteryStorage(capacity_kwh=100.0, soc_min=0.20, initial_soc=0.60)
    b_conf = BatteryStorage(capacity_kwh=100.0, soc_min=0.20, initial_soc=0.60)
    
    ctrl_det = DeterministicController(b_det)
    ctrl_gauss = GaussianController(b_gauss)
    ctrl_conf = ConformalRiskController(b_conf, lookahead_steps=4, safety_buffer=0.03)
    
    # Tracking logs
    test_start = calib_steps
    test_end = test_start + eval_steps
    
    records = {
        "timestamp": [],
        "pv_actual": [],
        "load_actual": [],
        "pv_observed": [],
        "load_observed": [],
        "dropped_flag": [],
        # Forecasts & bounds
        "pv_pred": [],
        "pv_lower": [],
        "pv_upper": [],
        "load_pred": [],
        "load_lower": [],
        "load_upper": [],
        # Deterministic
        "soc_det": [],
        "unserved_det": [],
        # Gaussian
        "soc_gauss": [],
        "unserved_gauss": [],
        "shed_gauss": [],
        # Conformal
        "soc_conf": [],
        "unserved_conf": [],
        "shed_conf": [],
        "conf_alert": []
    }
    
    # Circular lags buffer for forecasters
    pv_lags = list(pv_actual[test_start-4:test_start])
    load_lags = list(load_actual[test_start-4:test_start])
    
    for t_idx in range(test_start, test_end):
        t_now = times[t_idx]
        h = hours[t_idx]
        dow = dayofweek[t_idx]
        
        y_pv = pv_actual[t_idx]
        y_load = load_actual[t_idx]
        
        # Telemetry transmission
        in_blackout = (blackout_window is not None and blackout_window[0] <= (t_idx - test_start) <= blackout_window[1])
        obs_pv, drop_pv = channel_pv.transmit(y_pv, is_blackout=in_blackout)
        obs_load, drop_load = channel_load.transmit(y_load, is_blackout=in_blackout)
        is_dropped = drop_pv or drop_load
        
        # Point predictions based on received (potentially stale) lags
        pred_pv = pv_forecaster.predict_one_step(pv_lags, h, dow)
        pred_load = load_forecaster.predict_one_step(load_lags, h, dow)
        
        # Conformal bounds via Adaptive Conformal Inference
        staleness = 1.35 if is_dropped else 1.0
        pv_low, pv_high, _ = pv_aci.get_prediction_interval(pred_pv, telemetry_dropped=is_dropped, staleness_penalty=staleness)
        load_low, load_high, _ = load_aci.get_prediction_interval(pred_load, telemetry_dropped=is_dropped, staleness_penalty=staleness)
        
        # Update ACI states using true observed values if packet was received
        if not drop_pv:
            pv_aci.update(obs_pv, pred_pv, pv_low, pv_high)
        if not drop_load:
            load_aci.update(obs_load, pred_load, load_low, load_high)
            
        # Update lag buffers with observed values
        pv_lags.pop(0)
        pv_lags.append(obs_pv)
        load_lags.pop(0)
        load_lags.append(obs_load)
        
        # 1. Dispatch Deterministic Controller
        cmd_det, _ = ctrl_det.dispatch(pred_pv, pred_load, b_det.soc)
        act_p_det, next_soc_det, viol_det = b_det.step(cmd_det)
        # Power balance check: unserved critical load
        net_actual = y_load - y_pv
        served_det = y_pv + (-act_p_det if act_p_det < 0 else 0.0)
        unserved_det = max(0.0, y_load - served_det)
        
        # 2. Dispatch Gaussian Controller
        ctrl_gauss.update_error(abs(obs_pv - pred_pv) + abs(obs_load - pred_load))
        cmd_gauss, shed_gauss = ctrl_gauss.dispatch(pred_pv, pred_load, b_gauss.soc)
        act_p_gauss, next_soc_gauss, _ = b_gauss.step(cmd_gauss)
        served_gauss = y_pv + (-act_p_gauss if act_p_gauss < 0 else 0.0)
        unserved_gauss = max(0.0, (y_load - shed_gauss) - served_gauss)
        
        # 3. Dispatch Conformal Risk Controller
        cmd_conf, shed_conf, alert_conf = ctrl_conf.dispatch(
            pred_pv, pred_load, b_conf.soc, pv_low, load_high, is_telemetry_degraded=is_dropped
        )
        act_p_conf, next_soc_conf, _ = b_conf.step(cmd_conf)
        served_conf = y_pv + (-act_p_conf if act_p_conf < 0 else 0.0)
        unserved_conf = max(0.0, (y_load - shed_conf) - served_conf)
        
        # Log step
        records["timestamp"].append(t_now)
        records["pv_actual"].append(y_pv)
        records["load_actual"].append(y_load)
        records["pv_observed"].append(obs_pv)
        records["load_observed"].append(obs_load)
        records["dropped_flag"].append(is_dropped)
        records["pv_pred"].append(pred_pv)
        records["pv_lower"].append(pv_low)
        records["pv_upper"].append(pv_high)
        records["load_pred"].append(pred_load)
        records["load_lower"].append(load_low)
        records["load_upper"].append(load_high)
        records["soc_det"].append(next_soc_det)
        records["unserved_det"].append(unserved_det)
        records["soc_gauss"].append(next_soc_gauss)
        records["unserved_gauss"].append(unserved_gauss)
        records["shed_gauss"].append(shed_gauss)
        records["soc_conf"].append(next_soc_conf)
        records["unserved_conf"].append(unserved_conf)
        records["shed_conf"].append(shed_conf)
        records["conf_alert"].append(alert_conf)
        
    res_df = pd.DataFrame(records)
    return res_df

def run_all_benchmarks():
    """
    Executes all experimental evaluation regimes and compiles comparative tables.
    """
    os.makedirs("results", exist_ok=True)
    print("=" * 80)
    print("STARTING FULL REPRODUCIBLE EXPERIMENT BENCHMARK")
    print("=" * 80)
    
    regimes = [
        {"name": "Ideal (Clean Telemetry)", "drop": 0.0, "lat": 0, "noise": 0.0, "blackout": None},
        {"name": "Mild Drop (10% drop, 15m lag)", "drop": 0.10, "lat": 1, "noise": 0.02, "blackout": None},
        {"name": "Severe Faults (25% drop, 30m lag, 4% noise)", "drop": 0.25, "lat": 2, "noise": 0.04, "blackout": None},
        {"name": "Stress Blackout (25% drop + 3h Comms Outage)", "drop": 0.25, "lat": 2, "noise": 0.04, "blackout": (480, 492)} # 3h during day 5
    ]
    
    summary_rows = []
    
    for r in regimes:
        print(f"\n--- Running Regime: {r['name']} ---")
        df_res = run_simulation(
            test_days=20,
            packet_drop_rate=r["drop"],
            latency_steps=r["lat"],
            noise_std_frac=r["noise"],
            blackout_window=r["blackout"],
            seed=42
        )
        
        # Conformal statistical coverage
        pv_cov = calculate_empirical_coverage(df_res["pv_actual"], df_res["pv_lower"], df_res["pv_upper"])
        load_cov = calculate_empirical_coverage(df_res["load_actual"], df_res["load_lower"], df_res["load_upper"])
        pv_wis = calculate_winkler_score(df_res["pv_actual"], df_res["pv_lower"], df_res["pv_upper"])
        
        # Operational KPIs
        kpi_det = calculate_operational_kpis(df_res["soc_det"], df_res["unserved_det"], np.zeros(len(df_res)))
        kpi_gauss = calculate_operational_kpis(df_res["soc_gauss"], df_res["unserved_gauss"], df_res["shed_gauss"])
        kpi_conf = calculate_operational_kpis(df_res["soc_conf"], df_res["unserved_conf"], df_res["shed_conf"])
        
        summary_rows.append({
            "Regime": r["name"],
            "PV Empirical Coverage": f"{pv_cov*100:.1f}%",
            "Load Empirical Coverage": f"{load_cov*100:.1f}%",
            "Det. Battery Violations": kpi_det["soc_violation_count"],
            "Det. Blackout Energy (kWh)": kpi_det["unserved_blackout_energy_kwh"],
            "Gauss. Battery Violations": kpi_gauss["soc_violation_count"],
            "Gauss. Blackout Energy (kWh)": kpi_gauss["unserved_blackout_energy_kwh"],
            "Conformal Violations": kpi_conf["soc_violation_count"],
            "Conformal Blackout Energy (kWh)": kpi_conf["unserved_blackout_energy_kwh"],
            "Proactive Flexible Shed (kWh)": kpi_conf["proactive_flexible_shed_kwh"]
        })
        
        # Save time-series trace of the most severe regime for plotting
        if "Severe" in r["name"]:
            severe_df = df_res
            
    summary_df = pd.DataFrame(summary_rows)
    csv_path = "results/benchmark_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("BENCHMARK EXPERIMENT COMPLETED. SUMMARY TABLE:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to: {csv_path}")
    
    # Generate High-Resolution Publication Figure
    print("\nGenerating publication-quality figure: results/figure_conformal_vs_baseline.png")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Slice 5 representative days (480 timesteps)
    slice_df = severe_df.iloc[384:864].reset_index(drop=True)
    x = range(len(slice_df))
    
    # Subplot 1: PV Actual vs Conformal Prediction Interval
    ax1 = axes[0]
    ax1.plot(x, slice_df["pv_actual"], color="#2c3e50", lw=1.8, label="Ground Truth PV (kW)")
    ax1.plot(x, slice_df["pv_pred"], color="#2980b9", ls="--", lw=1.2, label="Point Forecast $\\hat{P}_t$")
    ax1.fill_between(x, slice_df["pv_lower"], slice_df["pv_upper"], color="#3498db", alpha=0.3, label="90% Conformal Interval $[L_t, U_t]$")
    dropped_indices = np.where(slice_df["dropped_flag"])[0]
    ax1.scatter(dropped_indices, slice_df["pv_actual"].iloc[dropped_indices], color="#e74c3c", s=12, alpha=0.6, label="Telemetry Dropped / Stale")
    ax1.set_ylabel("PV Power (kW)", fontsize=11, fontweight="bold")
    ax1.set_title("Adaptive Conformal State Estimation Under 25% Telemetry Loss & Stale Readings", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # Subplot 2: Battery SOC Trajectories
    ax2 = axes[1]
    ax2.plot(x, slice_df["soc_det"] * 100, color="#e74c3c", lw=1.8, label="Deterministic Twin (Blind to Dropouts)")
    ax2.plot(x, slice_df["soc_gauss"] * 100, color="#f39c12", lw=1.5, ls="-.", label="Gaussian Interval Twin")
    ax2.plot(x, slice_df["soc_conf"] * 100, color="#27ae60", lw=2.2, label="Proposed Conformal Risk Twin")
    ax2.axhline(20.0, color="#c0392b", ls=":", lw=1.8, label="Physical Safety Minimum (20% SOC)")
    ax2.set_ylabel("Battery SOC (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Battery State-of-Charge Dynamics and Safety Violation Regimes", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    # Subplot 3: Unserved Energy / Blackout Impact
    ax3 = axes[2]
    ax3.bar(x, slice_df["unserved_det"], width=1.0, color="#e74c3c", alpha=0.7, label="Deterministic Unserved Blackout Energy (kW)")
    ax3.bar(x, slice_df["shed_conf"], width=1.0, color="#27ae60", alpha=0.5, label="Conformal Proactive Flexible Load Shed (kW)")
    ax3.set_ylabel("Power Curtailment (kW)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time Step (15-Minute Intervals)", fontsize=11, fontweight="bold")
    ax3.set_title("Preventing Unserved Energy via Proactive Tiered Demand Relief", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    fig.savefig("results/figure_conformal_vs_baseline.png", dpi=300)
    plt.close()
    print("Figure saved successfully to: results/figure_conformal_vs_baseline.png")

if __name__ == "__main__":
    run_all_benchmarks()
