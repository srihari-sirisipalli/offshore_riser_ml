import joblib
from pathlib import Path
from sklearn.base import BaseEstimator
from utils.exceptions import ModelTrainingError

def safe_load_model(path: Path) -> BaseEstimator:
    """Safely load model with validation."""
    try:
        model = joblib.load(path)
        if not isinstance(model, BaseEstimator):
            raise ValueError("Invalid model type")
        return model
    except Exception as e:
        raise ModelTrainingError(f"Failed to load model: {e}")
