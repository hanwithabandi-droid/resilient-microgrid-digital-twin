"""
Proposed Strategy: Conformal Risk-Constrained Energy Management System (CRC-EMS).
Leverages Adaptive Conformal Inference (ACI) guarantees to enforce a non-parametric
chance-constraint on battery safety and prevent unserved critical load blackouts.
"""

import numpy as np

class ConformalRiskController:
    def __init__(self, battery, dt_hours=0.25, lookahead_steps=4, safety_buffer=0.03):
        """
        Parameters:
            battery: BatteryStorage instance.
            dt_hours: Time step duration (0.25h = 15 min).
            lookahead_steps: Horizon H over which worst-case SOC trajectory is verified (e.g. 4 steps = 1 hr).
            safety_buffer: Additional safety margin above soc_min (e.g., 0.03 = 3% SOC).
        """
        self.battery = battery
        self.dt_hours = dt_hours
        self.lookahead_steps = lookahead_steps
        self.safety_buffer = safety_buffer

    def dispatch(self, predicted_pv, predicted_load, current_soc, 
                 conformal_pv_lower, conformal_load_upper, is_telemetry_degraded=False):
        """
        Computes safety-calibrated dispatch using conformal worst-case bounds:
        Worst-case net demand = conformal_load_upper - conformal_pv_lower.
        
        Returns:
            battery_command_kw: Optimal battery power command.
            proactive_shed_kw: Proactively curtailed non-critical load (kWh).
            risk_alert: String flag ('NORMAL', 'RESERVE_WARNING', 'PROACTIVE_SHED').
        """
        net_nominal = predicted_load - predicted_pv
        net_worst = max(0.0, conformal_load_upper - conformal_pv_lower)
        
        # 1. Project worst-case battery SOC over lookahead horizon H
        energy_drain_kwh = (net_worst * self.dt_hours * self.lookahead_steps) / self.battery.eta_discharge
        projected_worst_soc = current_soc - (energy_drain_kwh / self.battery.capacity_kwh)
        
        threshold = self.battery.soc_min + self.safety_buffer
        proactive_shed_kw = 0.0
        risk_alert = "NORMAL"
        
        # 2. Check for violation of chance-constraint
        if projected_worst_soc < threshold:
            risk_alert = "RESERVE_WARNING"
            
            # Compute required power reduction to preserve battery above threshold
            soc_shortfall = threshold - projected_worst_soc
            required_energy_savings_kwh = soc_shortfall * self.battery.capacity_kwh
            required_power_relief_kw = required_energy_savings_kwh / (self.dt_hours * self.lookahead_steps)
            
            # Tier-1 flexible non-critical load shedding (up to 30% of total load)
            max_flexible_load = predicted_load * 0.30
            proactive_shed_kw = float(np.clip(required_power_relief_kw, 0.0, max_flexible_load))
            
            if proactive_shed_kw > 0.0:
                risk_alert = "PROACTIVE_SHED"
                
            # Recalculate net demand after proactive flexible curtailment
            effective_net = (predicted_load - proactive_shed_kw) - predicted_pv
            command = -effective_net
        else:
            # System has certified safety margin: dispatch to cover nominal deficit
            command = -net_nominal
            
        return command, proactive_shed_kw, risk_alert
