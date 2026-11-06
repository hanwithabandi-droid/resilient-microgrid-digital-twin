"""
Scholarly Evaluation Metrics for Probabilistic Forecasting and Microgrid Operational Safety.
Computes standard statistical metrics (Empirical Coverage, Winkler Score)
and operational power system KPIs (Unserved Energy, Battery Safety Violations).
"""

import numpy as np

def calculate_empirical_coverage(actuals, lower_bounds, upper_bounds):
    """
    Computes empirical coverage probability: fraction of ground truth inside intervals.
    """
    actuals = np.asarray(actuals)
    lower = np.asarray(lower_bounds)
    upper = np.asarray(upper_bounds)
    
    covered = (actuals >= lower - 1e-5) & (actuals <= upper + 1e-5)
    return float(np.mean(covered))

def calculate_mean_interval_width(lower_bounds, upper_bounds):
    """
    Computes sharpness / average interval width.
    """
    lower = np.asarray(lower_bounds)
    upper = np.asarray(upper_bounds)
    return float(np.mean(upper - lower))

def calculate_winkler_score(actuals, lower_bounds, upper_bounds, alpha=0.10):
    """
    Computes the Winkler Score (Weighted Interval Score).
    Lower is better. Heavily penalizes intervals that miss the target.
    
    W = (U - L) + (2/alpha) * (L - y) * I(y < L) + (2/alpha) * (y - U) * I(y > U)
    """
    actuals = np.asarray(actuals)
    lower = np.asarray(lower_bounds)
    upper = np.asarray(upper_bounds)
    
    width = upper - lower
    penalty_under = (2.0 / alpha) * (lower - actuals) * (actuals < lower)
    penalty_over = (2.0 / alpha) * (actuals - upper) * (actuals > upper)
    
    scores = width + penalty_under + penalty_over
    return float(np.mean(scores))

def calculate_operational_kpis(actual_soc_history, unserved_load_history, 
                               proactive_shed_history, soc_min=0.20, dt_hours=0.25):
    """
    Computes power system reliability and battery safety metrics.
    """
    socs = np.asarray(actual_soc_history)
    unserved = np.asarray(unserved_load_history)
    shed = np.asarray(proactive_shed_history)
    
    # 1. Battery SOC violation rate (time spent below 20%)
    violations = np.sum(socs < (soc_min - 1e-4))
    violation_rate_pct = (violations / len(socs)) * 100.0
    
    # 2. Total unserved energy (unserved demand in kWh)
    total_unserved_kwh = float(np.sum(unserved * dt_hours))
    
    # 3. Total proactively managed non-critical load shed (kWh)
    total_proactive_shed_kwh = float(np.sum(shed * dt_hours))
    
    # 4. Minimum recorded SOC
    min_soc = float(np.min(socs))
    
    return {
        "soc_violation_count": int(violations),
        "soc_violation_rate_pct": round(violation_rate_pct, 2),
        "min_soc_reached": round(min_soc, 4),
        "unserved_blackout_energy_kwh": round(total_unserved_kwh, 2),
        "proactive_flexible_shed_kwh": round(total_proactive_shed_kwh, 2)
    }
