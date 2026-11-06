"""
Telemetry Fault & Sensor Degradation Simulator.
Simulates real-world IoT sensor communication challenges:
1. Bernoulli packet loss (missing measurements).
2. Transmission latency buffers (stale data).
3. Zero-mean Gaussian and systematic sensor drift.
4. Continuous burst communication blackouts.
"""

import numpy as np

class TelemetryChannel:
    def __init__(self, packet_drop_rate=0.15, latency_steps=2, noise_std_fraction=0.03, random_seed=42):
        """
        Parameters:
            packet_drop_rate: Probability p in [0, 1] of an edge reading failing to arrive.
            latency_steps: Number of timesteps (each 15 min) data is delayed in transit.
            noise_std_fraction: Relative Gaussian measurement noise (e.g. 0.03 = 3% noise).
            random_seed: Seed for scientific repeatability.
        """
        self.packet_drop_rate = packet_drop_rate
        self.latency_steps = max(0, int(latency_steps))
        self.noise_std_fraction = noise_std_fraction
        self.rng = np.random.RandomState(random_seed)
        
        # Internal circular latency buffer
        self.buffer = []
        self.last_valid_reading = None

    def reset(self):
        self.buffer = []
        self.last_valid_reading = None

    def transmit(self, ground_truth_val, is_blackout=False):
        """
        Passes a clean ground-truth value through the degraded cyber-physical channel.
        
        Returns:
            observed_val: What the digital twin observes (possibly noisy, stale, or held from last packet).
            is_dropped: Boolean indicating whether the raw reading was lost in transit.
            effective_latency: Timesteps of staleness.
        """
        # 1. Inject physical sensor noise
        if ground_truth_val is not None:
            noise = self.rng.normal(0.0, max(0.5, abs(ground_truth_val) * self.noise_std_fraction))
            sensor_reading = max(0.0, ground_truth_val + noise)
        else:
            sensor_reading = 0.0
            
        # 2. Push to latency buffer
        self.buffer.append(sensor_reading)
        
        # 3. Retrieve value with latency
        if len(self.buffer) > self.latency_steps:
            delayed_reading = self.buffer.pop(0)
        else:
            delayed_reading = self.buffer[0]
            
        # 4. Check for packet loss or burst blackout
        dropped = False
        if is_blackout or (self.rng.uniform(0.0, 1.0) < self.packet_drop_rate):
            dropped = True
            # Zero-order hold (ZOH) from last valid telemetry, or fallback if none
            observed_val = self.last_valid_reading if self.last_valid_reading is not None else delayed_reading
        else:
            observed_val = delayed_reading
            self.last_valid_reading = delayed_reading
            
        return observed_val, dropped

def corrupt_time_series(series, packet_drop_rate=0.15, latency_steps=2, noise_fraction=0.03, blackout_window=None, seed=42):
    """
    Corrupts an entire 1D array or pandas Series according to the specified degradation protocol.
    """
    channel = TelemetryChannel(packet_drop_rate, latency_steps, noise_fraction, seed)
    n = len(series)
    observed = np.zeros(n)
    dropped_mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        in_blackout = False
        if blackout_window is not None:
            start_idx, end_idx = blackout_window
            if start_idx <= i <= end_idx:
                in_blackout = True
                
        obs, drop = channel.transmit(series[i], is_blackout=in_blackout)
        observed[i] = obs
        dropped_mask[i] = drop
        
    return observed, dropped_mask
