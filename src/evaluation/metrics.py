import numpy as np
from typing import Tuple

def detection_delay(detection_time: float, attack_time: float) -> float:
    """Calculate detection delay in time units.
    
    Args:
        detection_time: Time when detection occurred
        attack_time: Time when attack started
        
    Returns:
        Detection delay (detection_time - attack_time). Negative values indicate detection before attack.
    """
    return detection_time - attack_time

def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate false positive rate.
    
    Args:
        y_true: True binary labels (0=normal, 1=attack)
        y_pred: Predicted binary labels (0=normal, 1=attack)
        
    Returns:
        False positive rate = FP / (FP + TN)
    """
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if tn + fp == 0:
        return 0.0
    return fp / (tn + fp)

def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate true positive rate (recall).
    
    Args:
        y_true: True binary labels (0=normal, 1=attack)
        y_pred: Predicted binary labels (0=normal, 1=attack)
        
    Returns:
        True positive rate = TP / (TP + FN)
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def estimation_mse(estimated: np.ndarray, true: np.ndarray) -> float:
    """Calculate mean squared error between estimated and true values.
    
    Args:
        estimated: Estimated values
        true: True values
        
    Returns:
        Mean squared error
    """
    return np.mean((estimated - true) ** 2)

def estimation_bias(estimated: np.ndarray, true: np.ndarray) -> float:
    """Calculate bias between estimated and true values.
    
    Args:
        estimated: Estimated values
        true: True values
        
    Returns:
        Bias = mean(estimated - true)
    """
    return np.mean(estimated - true)