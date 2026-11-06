"""
Base Autoregressive Forecaster for PV Generation and Load Demand.
Implemented using regularized Ridge Regression with harmonic diurnal features.
Pure NumPy/SciPy formulation ensuring zero external dependency fragility.
"""

import numpy as np

class HarmonicRidgeForecaster:
    def __init__(self, alpha_reg=1.0, n_lags=4):
        """
        Parameters:
            alpha_reg: L2 Tikhonov regularization penalty.
            n_lags: Number of autoregressive historical 15-minute lags to include.
        """
        self.alpha_reg = alpha_reg
        self.n_lags = n_lags
        self.weights = None
        self.mean_y = 0.0

    def _extract_features(self, series, hours, days_of_week):
        """
        Constructs design matrix X:
        - Autoregressive lags: y_{t-1}, ..., y_{t-p}
        - Diurnal harmonics: sin(2*pi*h/24), cos(2*pi*h/24), sin(4*pi*h/24), cos(4*pi*h/24)
        - Day-of-week indicators (is_weekend)
        - Bias constant
        """
        n = len(series)
        p = self.n_lags
        valid_n = n - p
        
        X = np.zeros((valid_n, p + 5 + 1))
        y = np.zeros(valid_n)
        
        h_norm = hours[p:] / 24.0 * 2.0 * np.pi
        is_weekend = (days_of_week[p:] >= 5).astype(float)
        
        for i in range(valid_n):
            idx = i + p
            # Lags
            X[i, :p] = series[idx-p:idx][::-1]
            # Diurnal harmonics
            X[i, p] = np.sin(h_norm[i])
            X[i, p+1] = np.cos(h_norm[i])
            X[i, p+2] = np.sin(2 * h_norm[i])
            X[i, p+3] = np.cos(2 * h_norm[i])
            X[i, p+4] = is_weekend[i]
            X[i, p+5] = 1.0 # intercept
            y[i] = series[idx]
            
        return X, y

    def fit(self, series, hours, days_of_week):
        """
        Fits ridge regression weights via analytical normal equations:
        w = (X^T X + alpha * I)^{-1} X^T y
        """
        X, y = self._extract_features(series, hours, days_of_week)
        self.mean_y = np.mean(y)
        
        d = X.shape[1]
        reg_matrix = self.alpha_reg * np.eye(d)
        reg_matrix[-1, -1] = 0.0 # Do not regularize intercept
        
        # Solve regularized normal equations
        A = X.T @ X + reg_matrix
        b = X.T @ y
        self.weights = np.linalg.solve(A, b)

    def predict_one_step(self, recent_lags, hour, day_of_week):
        """
        Predicts y_{t} given the past p lags and current time metadata.
        """
        if self.weights is None:
            return float(np.mean(recent_lags))
            
        p = self.n_lags
        feat = np.zeros(p + 5 + 1)
        feat[:p] = np.asarray(recent_lags)[:p][::-1]
        
        h_norm = (hour / 24.0) * 2.0 * np.pi
        feat[p] = np.sin(h_norm)
        feat[p+1] = np.cos(h_norm)
        feat[p+2] = np.sin(2 * h_norm)
        feat[p+3] = np.cos(2 * h_norm)
        feat[p+4] = 1.0 if day_of_week >= 5 else 0.0
        feat[p+5] = 1.0
        
        pred = float(feat @ self.weights)
        return max(0.0, pred)
