import numpy as np
from typing import Dict, Tuple, Optional, List

class SensorDataGenerator:
    def __init__(self, n_sensors: int = 5, sampling_rate: float = 1.0, 
                 noise_level: float = 0.1, correlation: float = 0.0, 
                 seed: Optional[int] = None):
        self.n_sensors = n_sensors
        self.sampling_rate = sampling_rate
        self.noise_level = noise_level
        self.correlation = correlation
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
    def generate(self, n_timesteps: int, attack_type: str = 'none', 
                 t0: Optional[int] = None, **attack_params) -> Tuple[np.ndarray, Dict]:
        if t0 is None:
            t0 = n_timesteps // 2
            
        # Generate baseline: low-frequency sine waves with higher amplitude.
        # Frequencies are chosen so each sensor completes an INTEGER number of
        # periods over the full window (2*(s+1) periods over T, (s+1) per
        # half-window). This keeps half-series means at ~0 and makes the
        # first-difference noise estimator unbiased (sine derivative is
        # negligible). Baseline + noise are drawn ONCE per call (before attack
        # injection) so the RNG state sequence is identical for every
        # attack_type under a fixed seed.
        t = np.arange(n_timesteps) / self.sampling_rate
        baseline = np.zeros((n_timesteps, self.n_sensors))
        if self.correlation != 0.0:
            # Correlated sensors: share the SAME sine across all sensors so the
            # empirical correlation is ~1.0 (the iid noise of std noise_level
            # is negligible against the amplitude-10 signal).
            freq = 2.0 * self.sampling_rate / n_timesteps
            shared = 10.0 * np.sin(2 * np.pi * freq * t)
            baseline[:, :] = shared[:, np.newaxis]
        else:
            for s in range(self.n_sensors):
                freq = 2.0 * (s + 1) * self.sampling_rate / n_timesteps  # integer periods over T
                baseline[:, s] = 10.0 * np.sin(2 * np.pi * freq * t)
            
        # Generate noise: iid Gaussian with std EXACTLY noise_level
        noise = self.rng.normal(0, self.noise_level, (n_timesteps, self.n_sensors))
        
        # Start with baseline + noise
        data = baseline + noise
        
        # Apply attack
        if attack_type != 'none':
            delta_x = self._compute_attack_delta(t, t0, attack_type, **attack_params)
            # Broadcast delta_x to all sensors if needed
            if delta_x.ndim == 1:
                delta_x = np.tile(delta_x[:, np.newaxis], (1, self.n_sensors))
            data = data + delta_x
            
        # Create metadata
        meta = {
            'attack_type': attack_type,
            't0': t0,
            'n_timesteps': n_timesteps,
            'n_sensors': self.n_sensors,
            'attack_params': attack_params,
            'noise_level': self.noise_level,
            'seed': self.seed,
            'changepoints': [t0]
        }
        
        return data, meta
        
    def _compute_attack_delta(self, t: np.ndarray, t0: int, 
                             attack_type: str, **attack_params) -> np.ndarray:
        """Compute the attack delta_x for all sensors."""
        n_timesteps = len(t)
        
        # Time mask for attack
        attack_mask = t >= t0
        
        if attack_type == 'step':
            c = attack_params.get('c', 1.0)
            delta_x = np.zeros((n_timesteps, self.n_sensors))
            delta_x[attack_mask, :] = c
            
        elif attack_type == 'ramp':
            r = attack_params.get('r', 0.01)
            delta_x = np.zeros((n_timesteps, self.n_sensors))
            # Create 2D array for ramp attack
            ramp_values = r * (t[attack_mask] - t0)
            delta_x[attack_mask, :] = ramp_values[:, np.newaxis]
            
        elif attack_type == 'periodic':
            A = attack_params.get('A', 1.0)
            f = attack_params.get('f', 0.01)
            delta_x = np.zeros((n_timesteps, self.n_sensors))
            # Create 2D array for periodic attack
            periodic_values = A * np.sin(2 * np.pi * f * (t[attack_mask] - t0))
            delta_x[attack_mask, :] = periodic_values[:, np.newaxis]
            
        elif attack_type == 'coordinated':
            c = attack_params.get('c', 1.0)
            sensors = attack_params.get('sensors', list(range(self.n_sensors)))
            delta_x = np.zeros((n_timesteps, self.n_sensors))
            # Apply the step only to the selected sensors (loop avoids the
            # bool-mask/int-index broadcast error).
            for s in sensors:
                delta_x[attack_mask, s] = c
            
        return delta_x
