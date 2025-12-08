import inspect
from typing import Dict, Any, List
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
    HuberRegressor,
    TheilSenRegressor,
    PassiveAggressiveRegressor
)
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor

class ModelFactory:
    """
    Factory for creating Machine Learning models with a unified interface.
    Handles distinction between native multi-output models and those requiring wrappers.
    """

    # Models that support multi-output regression natively
    # (These fit ONE model that predicts [Sin, Cos] simultaneously)
    NATIVE_MODELS = {
        # Ensembles (Trees)
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'RandomForestRegressor': RandomForestRegressor,
        'DecisionTreeRegressor': DecisionTreeRegressor,
        
        # Nearest Neighbors
        'KNeighborsRegressor': KNeighborsRegressor,
        'RadiusNeighborsRegressor': RadiusNeighborsRegressor,
        
        # Neural Networks (Very powerful for multi-target)
        'MLPRegressor': MLPRegressor,
        
        # Linear / Kernel (Efficient native implementation via matrix math)
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
        'KernelRidge': KernelRidge
    }

    # Models that DO NOT support multi-output natively and MUST be wrapped
    # (These fit TWO separate models: Model_Sin and Model_Cos)
    WRAPPED_MODELS = {
        # Boosting
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'BaggingRegressor': BaggingRegressor,
        
        # Support Vector Machines
        'SVR': SVR,
        'LinearSVR': LinearSVR,
        'NuSVR': NuSVR,
        
        # Specialized Linear / Robust
        'BayesianRidge': BayesianRidge,
        'SGDRegressor': SGDRegressor,
        'HuberRegressor': HuberRegressor,
        'TheilSenRegressor': TheilSenRegressor,
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor
    }

    @classmethod
    def create(cls, model_name: str, params: Dict[str, Any] = None) -> Any:
        """
        Create and return an instantiated model.
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
            all_models = list(cls.NATIVE_MODELS.keys()) + list(cls.WRAPPED_MODELS.keys())
            raise ValueError(f"Unknown model name: {model_name}. Available: {all_models}")

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Return list of all supported model names."""
        return list(cls.NATIVE_MODELS.keys()) + list(cls.WRAPPED_MODELS.keys())

    @staticmethod
    def _filter_params(model_class, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove parameters from `params` that are not accepted by `model_class` constructor.
        """
        sig = inspect.signature(model_class.__init__)
        
        valid_keys = [
            p.name for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        ]
        
        # Always allow **kwargs if the model supports it
        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        
        if has_kwargs:
            return params
            
        return {k: v for k, v in params.items() if k in valid_keys}