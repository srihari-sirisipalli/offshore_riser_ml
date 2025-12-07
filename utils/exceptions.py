"""
Custom exception hierarchy for the Offshore Riser Angle Prediction System.
"""

class RiserMLException(Exception):
    """Base exception for all system errors."""
    pass

class ConfigurationError(RiserMLException):
    """Configuration validation failed."""
    pass

class DataValidationError(RiserMLException):
    """Data validation failed."""
    pass

class ModelTrainingError(RiserMLException):
    """Model training failed."""
    pass

class PredictionError(RiserMLException):
    """Prediction generation failed."""
    pass