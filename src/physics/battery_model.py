"""
Battery Energy Storage System (BESS) Physics and Dynamics Model.
Implements discrete-time state-of-charge (SOC) evolution, physical C-rate limits,
internal round-trip losses, and degradation throughput tracking.
"""

import numpy as np

class BatteryStorage:
    def __init__(self, capacity_kwh=100.0, soc_min=0.20, soc_max=0.95, 
                 p_max_charge_kw=30.0, p_max_discharge_kw=30.0,
                 eta_charge=0.95, eta_discharge=0.95, initial_soc=0.50):
        """
        Parameters:
            capacity_kwh: Nominal battery energy capacity (kWh).
            soc_min: Minimum allowable state of charge (Depth of Discharge protection, e.g. 0.20).
            soc_max: Maximum allowable state of charge (Overcharge protection, e.g. 0.95).
            p_max_charge_kw: Maximum charging power limit (kW).
            p_max_discharge_kw: Maximum discharging power limit (kW).
            eta_charge: Charging efficiency (0 < eta <= 1.0).
            eta_discharge: Discharging efficiency (0 < eta <= 1.0).
            initial_soc: Initial SOC fraction [soc_min, soc_max].
        """
        self.capacity_kwh = capacity_kwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.p_max_charge_kw = p_max_charge_kw
        self.p_max_discharge_kw = p_max_discharge_kw
        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        
        # State variables
        self.soc = initial_soc
        self.cumulative_energy_throughput_kwh = 0.0
        self.total_cycle_count = 0.0

    def reset(self, initial_soc=0.50):
        self.soc = initial_soc
        self.cumulative_energy_throughput_kwh = 0.0
        self.total_cycle_count = 0.0

    def step(self, p_command_kw, dt_hours=0.25):
        """
        Executes one discrete simulation step of length dt_hours.
        p_command_kw > 0: Request to charge battery from microgrid.
        p_command_kw < 0: Request to discharge battery to microgrid.
        
        Returns:
            actual_power_kw: Actual power dispatched (positive = charging, negative = discharging).
            next_soc: Resulting state of charge in [0, 1].
            violation_flag: True if a physical safety constraint was breached.
        """
        violation_flag = False
        
        if p_command_kw >= 0.0:
            # Charging branch
            # Physical power limit
            p_attempt = min(p_command_kw, self.p_max_charge_kw)
            
            # Energy needed to reach soc_max
            max_energy_in = (self.soc_max - self.soc) * self.capacity_kwh
            max_power_allowed = max_energy_in / (self.eta_charge * dt_hours)
            
            actual_power = max(0.0, min(p_attempt, max_power_allowed))
            energy_into_cells = actual_power * self.eta_charge * dt_hours
            
            self.soc += energy_into_cells / self.capacity_kwh
            self.cumulative_energy_throughput_kwh += actual_power * dt_hours
            
        else:
            # Discharging branch
            p_dis_request = abs(p_command_kw)
            p_attempt = min(p_dis_request, self.p_max_discharge_kw)
            
            # Energy available above soc_min
            max_energy_out = (self.soc - self.soc_min) * self.capacity_kwh
            max_power_allowed = (max_energy_out * self.eta_discharge) / dt_hours
            
            # Check if command violates safety minimum
            if p_attempt > max_power_allowed + 1e-4:
                violation_flag = True
                
            actual_dis_power = max(0.0, min(p_attempt, max_power_allowed))
            energy_out_of_cells = (actual_dis_power / self.eta_discharge) * dt_hours
            
            self.soc -= energy_out_of_cells / self.capacity_kwh
            self.cumulative_energy_throughput_kwh += actual_dis_power * dt_hours
            actual_power = -actual_dis_power

        # Safeguard floating point drift
        self.soc = float(np.clip(self.soc, 0.0, 1.0))
        if self.soc < self.soc_min - 1e-4:
            violation_flag = True
            
        # Equivalent full cycle (EFC) estimation: Throughput / (2 * Capacity)
        self.total_cycle_count = self.cumulative_energy_throughput_kwh / (2.0 * self.capacity_kwh)
        
        return actual_power, self.soc, violation_flag
