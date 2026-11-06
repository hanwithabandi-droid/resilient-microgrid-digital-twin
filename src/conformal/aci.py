"""
Adaptive Conformal Inference (ACI) for Dynamic Time Series.
Based on Gibbs & Candes (2021), "Adaptive Conformal Inference Under Distribution Shift",
and Xu & Xie (2021), "Conformal Prediction Interval for Dynamic Time-Series" (EnbPI).

Guarantees long-run valid empirical coverage (1 - alpha) under non-stationary weather regimes
and intermittent telemetry losses.
"""

import numpy as np

class AdaptiveConformalInference:
    def __init__(self, target_coverage=0.90, gamma=0.01, window_size=200, min_quantile=0.5):
        """
        Parameters:
            target_coverage: Target empirical coverage probability (e.g., 0.90 for 90%).
            gamma: Adaptive learning rate for quantile adjustment (typically 0.005 - 0.02).
            window_size: Rolling window size of recent residuals to calibrate on.
            min_quantile: Minimum absolute interval radius (kW) to prevent degenerate zero-width intervals.
        """
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage # e.g. 0.10
        self.gamma = gamma
        self.window_size = window_size
        self.min_quantile = min_quantile
        
        # Adaptive state
        self.alpha_t = self.alpha
        self.residuals_buffer = []
        self.coverage_history = []
        self.interval_widths = []

    def calibrate_initial(self, initial_residuals):
        """
        Seeds the residual buffer with historical calibration residuals.
        """
        self.residuals_buffer = list(np.abs(initial_residuals)[-self.window_size:])
        self.alpha_t = self.alpha

    def get_prediction_interval(self, point_prediction, telemetry_dropped=False, staleness_penalty=1.0):
        """
        Computes the calibrated lower and upper prediction bounds for step t:
        C_t = [y_hat - q_t, y_hat + q_t]
        """
        if len(self.residuals_buffer) == 0:
            q_t = self.min_quantile
        else:
            # Empirical (1 - alpha_t) quantile of recent residuals
            quantile_level = float(np.clip(1.0 - self.alpha_t, 0.50, 0.999))
            q_t = float(np.quantile(self.residuals_buffer, quantile_level))
            
        # Apply staleness inflation if telemetry was lost
        if telemetry_dropped:
            q_t = q_t * staleness_penalty
            
        q_t = max(self.min_quantile, q_t)
        
        lower_bound = max(0.0, point_prediction - q_t)
        upper_bound = point_prediction + q_t
        self.interval_widths.append(upper_bound - lower_bound)
        
        return lower_bound, upper_bound, q_t

    def update(self, actual_val, point_prediction, lower_bound, upper_bound):
        """
        Updates the non-conformity score buffer and performs the Gibbs-Candes adaptive step.
        """
        # 1. Check if ground truth was covered
        covered = (lower_bound <= actual_val <= upper_bound)
        err_t = 0.0 if covered else 1.0
        self.coverage_history.append(1.0 - err_t)
        
        # 2. Record new residual
        res = abs(actual_val - point_prediction)
        self.residuals_buffer.append(res)
        if len(self.residuals_buffer) > self.window_size:
            self.residuals_buffer.pop(0)
            
        # 3. Gibbs-Candes Adaptive update: alpha_{t+1} = alpha_t + gamma * (alpha - err_t)
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err_t), 0.01, 0.40))
        
        return covered, self.alpha_t
