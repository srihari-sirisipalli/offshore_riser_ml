import inspect
from typing import Dict, Any, List
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

class ModelFactory:
    """
    Factory for creating Machine Learning models with a unified interface.
    Handles distinction between native multi-output models and those requiring wrappers.
    """

    # Models that support multi-output regression natively (y shape: [n_samples, 2])
    NATIVE_MODELS = {
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor
    }

    # Models that require MultiOutputRegressor wrapper
    WRAPPED_MODELS = {
        'KNeighborsRegressor': KNeighborsRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet
    }

    @classmethod
    def create(cls, model_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Create and return an instantiated model.

        Parameters:
            model_name: Name of the model class (e.g., 'ExtraTreesRegressor')
            params: Dictionary of hyperparameters.

        Returns:
            A scikit-learn estimator (or MultiOutputRegressor wrapping one).
        """
        if params is None:
            params = {}

        # 1. Check Native Models
        if model_name in cls.NATIVE_MODELS:
            model_class = cls.NATIVE_MODELS[model_name]
            valid_params = cls._filter_params(model_class, params)
            return model_class(**valid_params)

        # 2. Check Wrapped Models
        elif model_name in cls.WRAPPED_MODELS:
            model_class = cls.WRAPPED_MODELS[model_name]
            valid_params = cls._filter_params(model_class, params)
            base_estimator = model_class(**valid_params)
            
            # Wrap logic: MultiOutputRegressor(estimator=base)
            return MultiOutputRegressor(base_estimator)

        else:
            raise ValueError(f"Unknown model name: {model_name}. "
                             f"Available: {list(cls.NATIVE_MODELS.keys()) + list(cls.WRAPPED_MODELS.keys())}")

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of all supported model names."""
        return list(cls.NATIVE_MODELS.keys()) + list(cls.WRAPPED_MODELS.keys())

    @staticmethod
    def _filter_params(model_class, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove parameters from `params` that are not accepted by `model_class` constructor.
        Prevents TypeError: __init__() got an unexpected keyword argument...
        """
        sig = inspect.signature(model_class.__init__)
        
        # FIX: Accept both POSITIONAL_OR_KEYWORD and KEYWORD_ONLY arguments
        valid_keys = [
            p.name for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        
        # Always allow **kwargs if the model supports it
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        
        if has_kwargs:
            return params
            
        return {k: v for k, v in params.items() if k in valid_keys}