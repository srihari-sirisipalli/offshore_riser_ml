import pandas as pd
import joblib
import logging
import json
import time
import os
from pathlib import Path
from typing import Any, Dict

from modules.model_factory import ModelFactory
from utils.exceptions import ModelTrainingError

class TrainingEngine:
    """
    Trains the final model using the optimal configuration and saves it for production.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def train(self, train_df: pd.DataFrame, model_config: Dict[str, Any], run_id: str) -> Any:
        """
        Train the model on the full training dataset.
        
        Parameters:
            train_df: Training data containing features and targets.
            model_config: Dictionary containing 'model' (name) and 'params'.
            run_id: Run identifier.
            
        Returns:
            Trained model object.
        """
        self.logger.info("Starting Final Model Training...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "05_FINAL_MODEL"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Prepare Data
        # Drop non-feature columns
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'], 
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]
        X = train_df.drop(columns=drop_cols, errors='ignore')
        y = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        feature_names = X.columns.tolist()
        
        # 3. Create Model
        model_name = model_config.get('model')
        params = model_config.get('params', {})
        
        if not model_name:
            raise ModelTrainingError("Model configuration missing 'model' name.")
            
        self.logger.info(f"Training {model_name} on {len(X)} samples with {len(feature_names)} features.")
        
        try:
            model = ModelFactory.create(model_name, params)
            
            # 4. Fit
            start_time = time.time()
            model.fit(X, y)
            duration = time.time() - start_time
            
            self.logger.info(f"Training completed in {duration:.2f} seconds.")
            
            # 5. Save Model & Metadata
            if self.config['outputs'].get('save_models', True):
                # Save Model Object
                model_path = output_dir / "final_model.pkl"
                joblib.dump(model, model_path)
                self.logger.info(f"Model saved to {model_path}")
                
                # Save Metadata
                metadata = {
                    'model': model_name,
                    'params': params,
                    'features': feature_names,
                    'input_shape': list(X.shape),
                    'training_time_sec': duration,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                with open(output_dir / "training_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
            return model
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to train model: {str(e)}")