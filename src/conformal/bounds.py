"""
Conformal Interval State Propagation for Battery State-of-Charge (SOC).
Transforms point-in-time conformal bounds on PV and Load into trajectory safety guarantees.
"""

import numpy as np

def compute_net_load_bounds(pv_lower, pv_upper, load_lower, load_upper):
    """
    Computes worst-case (upper) and best-case (lower) net load demand:
    Net Load = Load - PV
    Worst case (highest demand strain on battery) = Load_Upper - PV_Lower
    Best case (lowest demand / surplus PV) = Load_Lower - PV_Upper
    """
    net_worst = load_upper - pv_lower
    net_best = load_lower - pv_upper
    return net_best, net_worst

def propagate_battery_soc_envelope(current_soc, net_load_predictions, net_load_radii, 
                                   battery_capacity_kwh=100.0, eta_dis=0.95, dt_hours=0.25):
    """
    Propagates the conformal bounds forward over a rolling multi-step lookahead horizon.
    
    Returns:
        soc_nominal: Expected trajectory under nominal point forecast.
        soc_lower: Calibrated worst-case lower bound trajectory.
        soc_upper: Calibrated best-case upper bound trajectory.
    """
    H = len(net_load_predictions)
    soc_nom = np.zeros(H + 1)
    soc_low = np.zeros(H + 1)
    soc_high = np.zeros(H + 1)
    
    soc_nom[0] = current_soc
    soc_low[0] = current_soc
    soc_high[0] = current_soc
    
    for k in range(H):
        p_nom = net_load_predictions[k]
        q_rad = net_load_radii[k]
        
        p_worst = p_nom + q_rad # Max net deficit
        p_best = p_nom - q_rad  # Min net deficit
        
        # Nominal step
        delta_nom = -(p_nom * dt_hours) / (eta_dis * battery_capacity_kwh) if p_nom > 0 else -(p_nom * dt_hours * 0.95) / battery_capacity_kwh
        soc_nom[k+1] = np.clip(soc_nom[k] + delta_nom, 0.0, 1.0)
        
        # Worst-case (lower bound) step
        delta_low = -(p_worst * dt_hours) / (eta_dis * battery_capacity_kwh) if p_worst > 0 else 0.0
        soc_low[k+1] = np.clip(soc_low[k] + delta_low, 0.0, 1.0)
        
        # Best-case (upper bound) step
        delta_high = -(p_best * dt_hours * 0.95) / battery_capacity_kwh if p_best < 0 else 0.0
        soc_high[k+1] = np.clip(soc_high[k] + delta_high, 0.0, 1.0)
        
    return soc_nom, soc_low, soc_high
