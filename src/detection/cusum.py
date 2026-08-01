import numpy as np
from src.detection.base import BaseDetector

class CUSUMDetector(BaseDetector):
    def __init__(self, mu0: float, k: float, h: float, reset_after_detection: bool = True):
        self.mu0 = mu0
        self.k = k
        self.h = h
        self.reset_after_detection = reset_after_detection
        self.reset()
    
    def update(self, y: float):
        """Update the detector with new observation y.
        
        Returns:
            tuple[bool, float]: (alarm?, cumulative statistic S_t)
        """
        # Update cumulative statistic
        self.S_t = max(0, self.S_t + y - self.mu0 - self.k)
        
        # Check for alarm
        is_cp = self.S_t > self.h
        
        # Reset if needed
        if is_cp and self.reset_after_detection:
            self.reset()
        
        return is_cp, self.S_t
    
    def estimate_delta_x(self, window: np.ndarray):
        """Estimate the post-shift level relative to mu0.
        
        Args:
            window: numpy array of observations
            
        Returns:
            float: mean(window) - mu0
        """
        return np.mean(window) - self.mu0
    
    def reset(self):
        """Reset the detector state."""
        self.S_t = 0.0