import pytest
import pandas as pd
import numpy as np
import logging
import json
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.training_engine import TrainingEngine
from utils.exceptions import ModelTrainingError
from sklearn.ensemble import ExtraTreesRegressor

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration pointing to a temp dir."""
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
        'other_col': np.random.rand(n),
        'row_index': range(n)
    })

@pytest.fixture
def model_config():
    """Provides a sample model configuration."""
    return {
        "model": "ExtraTreesRegressor",
        "params": {"n_estimators": 5}
    }

# --- Test Cases ---

class TestTrainingEngine:

    def test_train_success_and_artifacts(self, base_config, sample_train_df, model_config, mock_logger, tmp_path):
        """Tests a successful training run and artifact persistence."""
        engine = TrainingEngine(base_config, mock_logger)
        
        trained_model = engine.train(sample_train_df, model_config, "test_run")
        
        # Check if model is returned and matches type
        assert isinstance(trained_model, ExtraTreesRegressor)
        assert trained_model.n_estimators == 5
        
        # Check logs
        mock_logger.info.assert_any_call("Starting Final Model Training...")
        
        # Check Files
        output_dir = Path(tmp_path) / "05_FINAL_MODEL"
        model_path = output_dir / "final_model.pkl"
        metadata_path = output_dir / "training_metadata.json"
        
        assert model_path.exists()
        assert metadata_path.exists()
        
        # Verify Metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['model'] == "ExtraTreesRegressor"
        assert metadata['features'] == ['feature_1']
        assert metadata['input_shape'] == [20, 1]
        assert 'training_time_sec' in metadata

    def test_train_no_features_error(self, base_config, sample_train_df, model_config, mock_logger):
        """Tests error raised when all columns are dropped."""
        # Drop the only feature column manually
        df_no_features = sample_train_df.drop(columns=['feature_1'])
        
        engine = TrainingEngine(base_config, mock_logger)
        
        with pytest.raises(ModelTrainingError, match="No features available"):
            engine.train(df_no_features, model_config, "test_run")

    def test_train_missing_model_name_error(self, base_config, sample_train_df, mock_logger):
        """Tests error raised when config lacks model name."""
        engine = TrainingEngine(base_config, mock_logger)
        invalid_config = {"params": {}} # Missing 'model'
        
        with pytest.raises(ModelTrainingError, match="missing 'model' name"):
            engine.train(sample_train_df, invalid_config, "test_run")

    @patch('modules.training_engine.training_engine.ModelFactory.create')
    def test_factory_failure_handling(self, mock_create, base_config, sample_train_df, model_config, mock_logger):
        """Tests that factory errors are wrapped in ModelTrainingError."""
        mock_create.side_effect = ValueError("Unknown model type")
        engine = TrainingEngine(base_config, mock_logger)
        
        with pytest.raises(ModelTrainingError, match="Failed to train model"):
            engine.train(sample_train_df, model_config, "test_run")

    def test_save_models_disabled(self, base_config, sample_train_df, model_config, mock_logger, tmp_path):
        """Tests that models are not saved if config flag is False."""
        base_config['outputs']['save_models'] = False
        engine = TrainingEngine(base_config, mock_logger)
        
        engine.train(sample_train_df, model_config, "test_run")
        
        output_dir = Path(tmp_path) / "05_FINAL_MODEL"
        assert not (output_dir / "final_model.pkl").exists()

    @patch('modules.training_engine.training_engine.gc.collect')
    def test_garbage_collection_called(self, mock_gc, base_config, sample_train_df, model_config, mock_logger):
        """Tests that garbage collection is triggered after training."""
        engine = TrainingEngine(base_config, mock_logger)
        engine.train(sample_train_df, model_config, "test_run")
        assert mock_gc.called