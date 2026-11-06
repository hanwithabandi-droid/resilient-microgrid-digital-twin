"""
Baseline 2: Uncalibrated Gaussian Interval EMS.
Estimates error variance under a stationary normal distribution assumption.
"""

import numpy as np

class GaussianController:
    def __init__(self, battery, dt_hours=0.25, confidence_z=1.645):
        """
        confidence_z: 1.645 corresponds to 90% two-sided normal interval.
        """
        self.battery = battery
        self.dt_hours = dt_hours
        self.z = confidence_z
        self.rolling_var = 4.0 # Initial variance estimate

    def update_error(self, actual_error):
        # Update rolling variance
        self.rolling_var = 0.95 * self.rolling_var + 0.05 * (actual_error ** 2)

    def dispatch(self, predicted_pv, predicted_load, current_soc):
        sigma = np.sqrt(max(0.1, self.rolling_var))
        # Heuristic 90% upper demand = predicted net + z * sigma
        net_demand_nom = predicted_load - predicted_pv
        net_demand_risk = net_demand_nom + self.z * sigma
        
        proactive_shed = 0.0
        
        # If expected worst-case would breach battery safety in 1 hour
        potential_soc = current_soc - (net_demand_risk * self.dt_hours * 4) / (self.battery.eta_discharge * self.battery.capacity_kwh)
        
        if potential_soc < self.battery.soc_min:
            # Shed a fixed fraction of non-critical load
            proactive_shed = min(predicted_load * 0.15, max(0.0, (self.battery.soc_min - potential_soc) * self.battery.capacity_kwh / self.dt_hours))
            net_demand_adjusted = (predicted_load - proactive_shed) - predicted_pv
            command = -net_demand_adjusted
        else:
            command = -net_demand_nom
            
        return command, proactive_shed
