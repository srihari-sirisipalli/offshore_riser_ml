import pytest
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.training_engine import TrainingEngine
from utils.exceptions import ModelTrainingError
from sklearn.ensemble import ExtraTreesRegressor

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the TrainingEngine."""
    return {
        "data": {
            "target_sin": "target_sin",
            "target_cos": "target_cos",
            "drop_columns": ["other_col"]
        },
        "outputs": {
            "base_results_dir": str(tmp_path),
            "save_models": True
        }
    }

@pytest.fixture
def sample_train_df():
    """Provides a sample training DataFrame."""
    n = 20
    return pd.DataFrame({
        'feature_1': np.random.rand(n),
        'target_sin': np.sin(np.linspace(0, 2, n)),
        'target_cos': np.cos(np.linspace(0, 2, n)),
        'other_col': np.random.rand(n)
    })

@pytest.fixture
def model_config():
    """Provides a sample model configuration."""
    return {
        "model": "ExtraTreesRegressor",
        "params": {"n_estimators": 10}
    }

# --- Test Cases ---

class TestTrainingEngine:

    def test_train_success_and_artifacts(self, base_config, sample_train_df, model_config, mock_logger, tmp_path):
        """Tests a successful training run, checking the returned model and saved artifacts."""
        engine = TrainingEngine(base_config, mock_logger)
        
        trained_model = engine.train(sample_train_df, model_config, "test_run")
        
        # Check if a model object is returned
        assert isinstance(trained_model, ExtraTreesRegressor)
        mock_logger.info.assert_any_call("Starting Final Model Training...")
        mock_logger.info.assert_any_call("Training ExtraTreesRegressor on 20 samples with 1 features.")
        mock_logger.info.assert_any_call("Training completed in 0.03 seconds.") # This assertion is brittle, we can patch time
        
        # Check for created files
        output_dir = Path(tmp_path) / "05_FINAL_MODEL"
        model_path = output_dir / "final_model.pkl"
        metadata_path = output_dir / "training_metadata.json"
        
        assert model_path.exists()
        assert metadata_path.exists()
        
        # Check metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['model'] == "ExtraTreesRegressor"
        assert metadata['features'] == ['feature_1']
        assert metadata['params']['n_estimators'] == 10

    def test_train_no_features_error(self, base_config, sample_train_df, model_config, mock_logger):
        """Tests that a ModelTrainingError is raised if there are no features to train on."""
        # Drop all columns that would be features
        df = sample_train_df.drop(columns=['feature_1'])
        engine = TrainingEngine(base_config, mock_logger)
        
        with pytest.raises(ModelTrainingError, match="No features available for training"):
            engine.train(df, model_config, "test_run")

    def test_train_missing_model_name_error(self, base_config, sample_train_df, mock_logger):
        """Tests that a ModelTrainingError is raised if the model name is missing from the config."""
        engine = TrainingEngine(base_config, mock_logger)
        invalid_model_config = {"params": {}} # Missing 'model' key
        
        with pytest.raises(ModelTrainingError, match="Model configuration missing 'model' name"):
            engine.train(sample_train_df, invalid_model_config, "test_run")
            
    @patch('modules.training_engine.ModelFactory.create')
    def test_train_model_creation_fails(self, mock_create, base_config, sample_train_df, model_config, mock_logger):
        """Tests that a ModelTrainingError is raised if the model factory fails."""
        mock_create.side_effect = ValueError("Unknown model")
        engine = TrainingEngine(base_config, mock_logger)
        
        with pytest.raises(ModelTrainingError, match="Failed to train model: Unknown model"):
            engine.train(sample_train_df, model_config, "test_run")

    def test_save_models_disabled(self, base_config, sample_train_df, model_config, mock_logger, tmp_path):
        """Tests that no artifacts are saved if 'save_models' is false."""
        base_config['outputs']['save_models'] = False
        engine = TrainingEngine(base_config, mock_logger)
        
        engine.train(sample_train_df, model_config, "test_run")
        
        output_dir = Path(tmp_path) / "05_FINAL_MODEL"
        model_path = output_dir / "final_model.pkl"
        metadata_path = output_dir / "training_metadata.json"
        
        assert not model_path.exists()
        assert not metadata_path.exists()
        
    @patch('modules.training_engine.joblib.dump')
    def test_save_model_exception_handling(self, mock_dump, base_config, sample_train_df, model_config, mock_logger):
        """Tests that an exception during model saving is caught and logged."""
        mock_dump.side_effect = IOError("Disk full")
        engine = TrainingEngine(base_config, mock_logger)
        
        # The train method should complete without raising an error
        engine.train(sample_train_df, model_config, "test_run")
        
        # Check that a warning was logged
        mock_logger.warning.assert_called_with("Failed to save model and/or metadata. Error: Disk full")
