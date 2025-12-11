import pandas as pd
import numpy as np
import joblib
import logging
import json
import time
import os
import gc
from pathlib import Path
from typing import Any, Dict

from modules.model_factory import ModelFactory
from utils.exceptions import ModelTrainingError
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils import constants

class NumpyEncoder(json.JSONEncoder):
    """
    Helper to serialize NumPy types in metadata JSONs.
    Prevents 'Object of type int64 is not JSON serializable' errors.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class TrainingEngine(BaseEngine):
    """
    Trains the final model using the optimal configuration and saves it for production.
    
    Improvements:
    - Explicit garbage collection (Memory Optimization).
    - Robust JSON serialization for metadata.
    - Timing metrics.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        
    def _get_engine_directory_name(self) -> str:
        return constants.FINAL_MODEL_DIR

    @handle_engine_errors("Training")
    def execute(self, train_df: pd.DataFrame, model_config: Dict[str, Any], run_id: str) -> Any:
        """
        Train the model on the full training dataset.
        
        Args:
            train_df: Training data containing features and targets.
            model_config: Dictionary containing 'model' (name) and 'params'.
            run_id: Run identifier.
            
        Returns:
            Trained model object.
        """
        self.logger.info("Starting Final Model Training...")
        
        # 1. Prepare Data
        # Drop non-feature columns
        # Note: We rely on the dataframe's own memory management here, 
        # but explicit copies are minimized by the underlying pandas implementation.
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'], 
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin', 'row_index'
        ]
        
        # Filter existing columns to avoid KeyErrors
        existing_drop_cols = [c for c in drop_cols if c in train_df.columns]
        
        X = train_df.drop(columns=existing_drop_cols)
        y = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        # Validation: Ensure features exist
        if X.empty or len(X.columns) == 0:
            raise ModelTrainingError("No features available for training after dropping metadata columns.")

        feature_names = X.columns.tolist()
        
        # 2. Create Model
        model_name = model_config.get('model')
        params = model_config.get('params', {})
        
        if not model_name:
            raise ModelTrainingError("Model configuration missing 'model' name.")
            
        self.logger.info(f"Training {model_name} on {len(X)} samples with {len(feature_names)} features.")
        
        try:
            model = ModelFactory.create(model_name, params)
            
            # 3. Fit with Timing
            start_time = time.time()
            model.fit(X, y)
            duration = time.time() - start_time
            
            self.logger.info(f"Training completed in {duration:.2f} seconds.")
            
            # 4. Save Model & Metadata
            if self.config['outputs'].get('save_models', True):
                try:
                    # Save Model Object
                    model_path = self.output_dir / "final_model.pkl"
                    joblib.dump(model, model_path)
                    self.logger.info(f"Model saved to {model_path}")
                    
                    # Save Metadata
                    # Include exact features for reproducibility checks later
                    metadata = {
                        'model': model_name,
                        'params': params,
                        'features': feature_names,
                        'input_shape': list(X.shape),
                        'training_time_sec': duration,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    with open(self.output_dir / "training_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to save model artifacts. Error: {e}")
            
            # 5. Memory Cleanup (Issue #P1)
            # Scikit-learn models can hold references to data or create large internal structures.
            # Explicitly deleting inputs and collecting garbage helps in tight loops (like RFE).
            del X, y
            gc.collect()
            
            return model
            
        except Exception as e:
            # Clean up on failure too
            if 'X' in locals(): del X
            if 'y' in locals(): del y
            gc.collect()
            raise ModelTrainingError(f"Failed to train model: {str(e)}")