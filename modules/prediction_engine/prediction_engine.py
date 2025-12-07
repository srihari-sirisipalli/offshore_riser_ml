import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from utils.circular_metrics import reconstruct_angle, wrap_angle
from utils.exceptions import PredictionError

class PredictionEngine:
    """
    Generates predictions, reconstructs angles from Sin/Cos, and computes errors.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def predict(self, model, data_df: pd.DataFrame, split_name: str, run_id: str) -> pd.DataFrame:
        """
        Generate predictions for a specific split (val/test).
        
        Parameters:
            model: Trained model object.
            data_df: DataFrame containing features and targets.
            split_name: 'val' or 'test' (used for filename).
            run_id: Run identifier.
            
        Returns:
            pd.DataFrame: Predictions with ground truth, errors, and metadata.
        """
        self.logger.info(f"Generating predictions for {split_name} set ({len(data_df)} samples)...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "06_PREDICTIONS"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Prepare Features
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'], 
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]
        X = data_df.drop(columns=drop_cols, errors='ignore')
        
        try:
            # 3. Model Inference (Sin/Cos)
            # Returns shape (n_samples, 2)
            preds = model.predict(X)
            
            pred_sin = preds[:, 0]
            pred_cos = preds[:, 1]
            
            # 4. Reconstruct Angle
            pred_angle = reconstruct_angle(pred_sin, pred_cos)
            
            # 5. Retrieve Ground Truth
            target_sin_col = self.config['data']['target_sin']
            target_cos_col = self.config['data']['target_cos']
            
            true_sin = data_df[target_sin_col].values
            true_cos = data_df[target_cos_col].values
            
            # If angle_deg exists (computed in DataManager), use it. Else reconstruct.
            if 'angle_deg' in data_df.columns:
                true_angle = data_df['angle_deg'].values
            else:
                true_angle = reconstruct_angle(true_sin, true_cos)
                
            # 6. Compute Error (Wrapped)
            error = wrap_angle(true_angle - pred_angle)
            abs_error = np.abs(error)
            
            # 7. Construct Result DataFrame
            results_df = pd.DataFrame({
                'row_index': data_df.index,
                'true_sin': true_sin,
                'true_cos': true_cos,
                'pred_sin': pred_sin,
                'pred_cos': pred_cos,
                'true_angle': true_angle,
                'pred_angle': pred_angle,
                'error': error,
                'abs_error': abs_error
            })
            
            # Add Metadata for Phase 4 Analysis (Bins, Hs)
            meta_cols = ['hs_bin', 'angle_bin', 'combined_bin', self.config['data']['hs_column']]
            for col in meta_cols:
                if col in data_df.columns:
                    results_df[col] = data_df[col].values
            
            # 8. Save
            if self.config['outputs'].get('save_predictions', True):
                save_path = output_dir / f"predictions_{split_name}.xlsx"
                results_df.to_excel(save_path, index=False)
                self.logger.info(f"Predictions saved to {save_path}")
                
            return results_df
            
        except Exception as e:
            raise PredictionError(f"Prediction failed for {split_name}: {str(e)}")