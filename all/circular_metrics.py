import numpy as np

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angle to [-180, 180] range.
    """
    return (angle + 180) % 360 - 180

def reconstruct_angle(sin_val: np.ndarray, cos_val: np.ndarray) -> np.ndarray:
    """
    Reconstruct angle (degrees) from sin/cos components.
    Range: [0, 360)
    """
    angle_rad = np.arctan2(sin_val, cos_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg % 360.0

def compute_cmae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Circular Mean Absolute Error.
    """
    error = wrap_angle(y_true - y_pred)
    return float(np.mean(np.abs(error)))

def compute_crmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Circular Root Mean Squared Error.
    """
    error = wrap_angle(y_true - y_pred)
    return float(np.sqrt(np.mean(error**2)))