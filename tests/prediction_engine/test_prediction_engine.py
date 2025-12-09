import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from unittest.mock import MagicMock
from modules.prediction_engine import PredictionEngine
from utils.exceptions import PredictionError

# --- Fixtures ---

@pytest.fixture
def mock_logger():
    """Provides a mock logger for tests."""
    return MagicMock(spec=logging.Logger)

@pytest.fixture
def base_config(tmp_path):
    """Provides a base configuration for the PredictionEngine."""
    return {
        "data": {
            "target_sin": "true_sin",
            "target_cos": "true_cos",
            "hs_column": "hs",
            "drop_columns": ["other_col"]
        },
        "outputs": {
            "base_results_dir": str(tmp_path),
            "save_predictions": True
        }
    }

@pytest.fixture
def sample_data_df():
    """Provides a sample DataFrame for prediction."""
    n = 10
    return pd.DataFrame({
        'feature_1': np.random.rand(n),
        'feature_2': np.random.rand(n),
        'true_sin': np.sin(np.linspace(0, 1, n)),
        'true_cos': np.cos(np.linspace(0, 1, n)),
        'angle_deg': np.degrees(np.arctan2(np.sin(np.linspace(0, 1, n)), np.cos(np.linspace(0, 1, n)))) % 360,
        'hs': np.random.rand(n) * 5,
        'other_col': range(n)
    }, index=pd.RangeIndex(start=100, stop=110, step=1)) # Use a non-standard index

@pytest.fixture
def mock_model():
    """Provides a mock trained model."""
    model = MagicMock()
    # Mock the predict method to return dummy sin/cos values
    dummy_preds = np.array([[0.1, 0.9]] * 10)
    model.predict.return_value = dummy_preds
    # Mock the feature_names_in_ attribute for consistency checks
    model.feature_names_in_ = ['feature_1', 'feature_2']
    return model

# --- Test Cases ---

class TestPredictionEngine:

    def test_predict_success_and_artifacts(self, base_config, sample_data_df, mock_model, mock_logger, tmp_path):
        """Tests a successful prediction run and artifact creation."""
        engine = PredictionEngine(base_config, mock_logger)
        
        results_df = engine.predict(mock_model, sample_data_df, "test", "run1")
        
        # Check returned DataFrame
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(sample_data_df)
        expected_cols = ['row_index', 'true_sin', 'true_cos', 'pred_sin', 'pred_cos', 'true_angle', 'pred_angle', 'error', 'abs_error']
        for col in expected_cols:
            assert col in results_df.columns
        
        # Check that row_index preserves the original DataFrame's index
        assert results_df['row_index'].tolist() == sample_data_df.index.tolist()
        
        # Check file creation
        output_path = Path(tmp_path) / "06_PREDICTIONS" / "predictions_test.xlsx"
        assert output_path.exists()
        
        # Verify the model's predict method was called with the correct features
        # The mock_model has feature_names_in_ set, so the engine should reorder
        call_args, _ = mock_model.predict.call_args
        passed_df = call_args[0]
        assert isinstance(passed_df, pd.DataFrame)
        assert passed_df.columns.tolist() == ['feature_1', 'feature_2']

    def test_predict_missing_feature_error(self, base_config, sample_data_df, mock_model, mock_logger):
        """Tests that a PredictionError is raised if a feature is missing."""
        df_missing_feature = sample_data_df.drop(columns=['feature_2'])
        engine = PredictionEngine(base_config, mock_logger)
        
        with pytest.raises(PredictionError, match="Missing features required by the model: \['feature_2'\]"):
            engine.predict(mock_model, df_missing_feature, "test", "run1")

    def test_predict_no_feature_names_in_warning(self, base_config, sample_data_df, mock_model, mock_logger):
        """Tests that a warning is logged if the model has no 'feature_names_in_' attribute."""
        delattr(mock_model, 'feature_names_in_')
        engine = PredictionEngine(base_config, mock_logger)
        
        engine.predict(mock_model, sample_data_df, "test", "run1")
        
        mock_logger.warning.assert_called_with(
            "Model does not have 'feature_names_in_' attribute. "
            "Cannot guarantee feature order consistency for prediction."
        )

    def test_predict_save_disabled(self, base_config, sample_data_df, mock_model, mock_logger, tmp_path):
        """Tests that no file is saved when 'save_predictions' is false."""
        base_config['outputs']['save_predictions'] = False
        engine = PredictionEngine(base_config, mock_logger)
        
        engine.predict(mock_model, sample_data_df, "test", "run1")
        
        output_path = Path(tmp_path) / "06_PREDICTIONS" / "predictions_test.xlsx"
        assert not output_path.exists()
        
    def test_predict_reconstructs_true_angle(self, base_config, sample_data_df, mock_model, mock_logger):
        """Tests that true_angle is reconstructed if not present in the dataframe."""
        df_no_angle = sample_data_df.drop(columns=['angle_deg'])
        engine = PredictionEngine(base_config, mock_logger)
        
        results_df = engine.predict(mock_model, df_no_angle, "test", "run1")
        
        # Check if true_angle was reconstructed and is approximately correct
        assert 'true_angle' in results_df.columns
        expected_true_angle = np.degrees(np.arctan2(df_no_angle['true_sin'], df_no_angle['true_cos'])) % 360
        assert np.allclose(results_df['true_angle'], expected_true_angle)
        
    def test_predict_raises_error_on_model_failure(self, base_config, sample_data_df, mock_model, mock_logger):
        """Tests that PredictionError is raised if the model's predict method fails."""
        mock_model.predict.side_effect = Exception("Internal model error")
        engine = PredictionEngine(base_config, mock_logger)
        
        with pytest.raises(PredictionError, match="Prediction failed for test: Internal model error"):
            engine.predict(mock_model, sample_data_df, "test", "run1")
