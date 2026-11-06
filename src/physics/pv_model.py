"""
Physics-based Photovoltaic (PV) conversion model.
Implements the King / Evans thermal balance and temperature-derated electrical generation.
"""

import numpy as np

class PVSystem:
    def __init__(self, p_stc_kw=50.0, p_inv_max_kw=40.0, eta_inv=0.96, gamma_temp=-0.004, noct=45.0):
        """
        Parameters:
            p_stc_kw: Rated DC peak power at Standard Test Conditions (STC: 1000 W/m^2, 25 deg C).
            p_inv_max_kw: Rated AC inverter capacity.
            eta_inv: Peak inverter conversion efficiency.
            gamma_temp: Temperature coefficient of power (% / deg C, typically -0.004).
            noct: Nominal Operating Cell Temperature (deg C).
        """
        self.p_stc_kw = p_stc_kw
        self.p_inv_max_kw = p_inv_max_kw
        self.eta_inv = eta_inv
        self.gamma_temp = gamma_temp
        self.noct = noct

    def calculate_cell_temperature(self, poa_irradiance, ambient_temp):
        """
        Computes PV cell temperature from POA irradiance and ambient temperature.
        """
        t_cell = ambient_temp + poa_irradiance * ((self.noct - 20.0) / 800.0)
        return t_cell

    def compute_power(self, poa_irradiance, ambient_temp):
        """
        Computes AC electrical power output (kW).
        """
        poa = np.maximum(0.0, np.asarray(poa_irradiance))
        t_amb = np.asarray(ambient_temp)
        t_cell = self.calculate_cell_temperature(poa, t_amb)
        
        # DC Power with temperature derating
        p_dc = self.p_stc_kw * (poa / 1000.0) * (1.0 + self.gamma_temp * (t_cell - 25.0))
        p_dc = np.maximum(0.0, p_dc)
        
        # AC Power after inverter conversion and clipping
        p_ac = np.minimum(self.p_inv_max_kw, p_dc * self.eta_inv)
        return p_ac
