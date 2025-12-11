import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any

from utils.circular_metrics import reconstruct_angle, wrap_angle
from utils.exceptions import PredictionError
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils.file_io import save_dataframe
from utils import constants

class PredictionEngine(BaseEngine):
    """
    Generates predictions using a trained model and computes component-wise errors.
    Ensures feature consistency between training and inference.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        
    def _get_engine_directory_name(self) -> str:
        return constants.PREDICTIONS_DIR

    @handle_engine_errors("Prediction")
    def execute(self, model: Any, data_df: pd.DataFrame, split_name: str, run_id: str) -> pd.DataFrame:
        """
        Generate predictions and error metrics.
        
        Parameters:
            model: Trained estimator (sklearn compatible).
            data_df: DataFrame containing features and targets.
            split_name: 'val', 'test', etc.
            run_id: Run identifier.
            
        Returns:
            DataFrame with predictions and errors.
        """
        self.logger.info(f"Generating predictions for {split_name} set ({len(data_df)} samples)...")
        
        # 1. Feature Preparation
        drop_cols_config = self.config['data'].get('drop_columns', [])
        targets = [self.config['data']['target_sin'], self.config['data']['target_cos']]
        meta_cols = ['angle_deg', 'angle_bin', 'hs_bin', 'combined_bin']
        
        exclude_cols = set(drop_cols_config + targets + meta_cols)
        
        features = [c for c in data_df.columns if c not in exclude_cols]
        X = data_df[features]
        
        if hasattr(model, 'feature_names_in_'):
            model_features = list(model.feature_names_in_)
            missing = sorted(list(set(model_features) - set(X.columns)))
            if missing:
                raise PredictionError(f"Missing features required by the model: {missing}")
            X = X[model_features]
        else:
            self.logger.warning(
                "Model does not have 'feature_names_in_' attribute. "
                "Cannot guarantee feature order consistency for prediction."
            )
        
        # 2. Predict
        try:
            preds = model.predict(X)
        except Exception as e:
            self.logger.error(f"Prediction failed for {split_name}: {e}")
            raise PredictionError(f"Prediction failed for {split_name}: {e}") from e

        # 3. Process Results
        true_sin = data_df[targets[0]].values
        true_cos = data_df[targets[1]].values
        
        pred_sin = preds[:, 0]
        pred_cos = preds[:, 1]
        
        true_angle = reconstruct_angle(true_sin, true_cos)
        pred_angle = reconstruct_angle(pred_sin, pred_cos)
        
        signed_error = wrap_angle(true_angle - pred_angle)
        
        results_df = pd.DataFrame({
            'row_index': data_df.index,
            'true_sin': true_sin,
            'true_cos': true_cos,
            'pred_sin': pred_sin,
            'pred_cos': pred_cos,
            'true_angle': true_angle,
            'pred_angle': pred_angle,
            'abs_error': np.abs(signed_error),
            'error': signed_error
        })
        
        hs_col = self.config['data'].get('hs_column')
        if hs_col and hs_col in data_df.columns:
            results_df[hs_col] = data_df[hs_col]
            # Convert meters to feet for downstream analysis and add a short alias
            results_df[f"{hs_col}_ft"] = data_df[hs_col] * 3.28084
            results_df["Hs_ft"] = results_df[f"{hs_col}_ft"]
        if 'hs_bin' in data_df.columns:
            results_df['hs_bin'] = data_df['hs_bin']
            
        results_df.attrs['split'] = split_name
        
        # 4. Save
        if self.config['outputs'].get('save_predictions', True):
            excel_copy = self.config.get('outputs', {}).get('save_excel_copy', False)
            save_path = self.output_dir / f"predictions_{split_name}.parquet"
            save_dataframe(results_df, save_path, excel_copy=excel_copy, index=False)
            self.logger.info(f"Predictions saved to {save_path}")
            
        return results_df
