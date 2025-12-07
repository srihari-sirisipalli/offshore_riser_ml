import pytest
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from modules.model_factory import ModelFactory

def test_create_native_model():
    """Test creating a model that supports multi-output natively."""
    params = {'n_estimators': 10, 'random_state': 42}
    model = ModelFactory.create('ExtraTreesRegressor', params)
    
    assert isinstance(model, ExtraTreesRegressor)
    assert model.n_estimators == 10
    assert model.random_state == 42

def test_create_wrapped_model():
    """Test creating a model that needs MultiOutputRegressor wrapper."""
    params = {'n_neighbors': 5}
    model = ModelFactory.create('KNeighborsRegressor', params)
    
    assert isinstance(model, MultiOutputRegressor)
    assert isinstance(model.estimator, KNeighborsRegressor)
    assert model.estimator.n_neighbors == 5

def test_unknown_model_error():
    """Test error handling for unknown models."""
    with pytest.raises(ValueError, match="Unknown model name"):
        ModelFactory.create('SuperAdvancedAIModel')

def test_parameter_filtering():
    """
    Test that invalid parameters are filtered out (e.g., passing 
    'random_state' to KNN which doesn't accept it).
    """
    # KNN does not take random_state
    params = {'n_neighbors': 3, 'random_state': 123} 
    
    # Should not raise TypeError
    model = ModelFactory.create('KNeighborsRegressor', params)
    
    assert isinstance(model, MultiOutputRegressor)
    assert model.estimator.n_neighbors == 3
    # Check that random_state was effectively ignored (not an attribute of KNN)
    assert not hasattr(model.estimator, 'random_state')

def test_get_available_models():
    """Test listing available models."""
    models = ModelFactory.get_available_models()
    assert 'ExtraTreesRegressor' in models
    assert 'KNeighborsRegressor' in models
    assert 'GradientBoostingRegressor' in models