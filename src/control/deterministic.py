"""
Baseline 1: Deterministic Energy Management System (EMS).
Makes dispatch decisions based solely on point predictions without uncertainty buffers.
"""

import numpy as np

class DeterministicController:
    def __init__(self, battery, dt_hours=0.25):
        self.battery = battery
        self.dt_hours = dt_hours

    def dispatch(self, predicted_pv, predicted_load, current_soc):
        """
        Dispatches battery purely according to expected net demand.
        Net demand = predicted_load - predicted_pv.
        If Net demand > 0 (deficit): Discharge battery to meet load.
        If Net demand < 0 (surplus): Charge battery with excess solar.
        
        Returns:
            battery_command_kw: Positive = charge, Negative = discharge.
            proactive_shed_kw: Proactive load shed (always 0 for deterministic).
        """
        net_demand = predicted_load - predicted_pv
        
        if net_demand > 0:
            # Deficit: request battery discharge up to available power
            command = -net_demand
        else:
            # Surplus: request battery charge with excess PV
            command = -net_demand # positive value
            
        proactive_shed_kw = 0.0
        return command, proactive_shed_kw
