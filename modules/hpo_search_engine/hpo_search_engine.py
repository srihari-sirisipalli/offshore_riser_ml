import pandas as pd
import numpy as np
import json
import os
import hashlib
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold, KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from modules.model_factory import ModelFactory
from utils.circular_metrics import compute_cmae, compute_crmse, reconstruct_angle

# FIX: Custom Encoder to handle Numpy float32/int64 types in JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, 
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class HPOSearchEngine:
    """
    Hyperparameter Optimization Engine with Resume Capability.
    Performs Grid Search with K-Fold Cross-Validation and generates detailed analytics.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.hpo_config = config.get('hyperparameters', {})
        self.progress_file: Optional[Path] = None
        self.completed_hashes = set()

    def execute(self, train_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """
        Execute Grid Search and generate comprehensive results.
        
        Returns:
            dict: The best configuration found.
        """
        self.logger.info("Starting Hyperparameter Optimization (HPO)...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "04_HYPERPARAMETER_SEARCH"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Setup Resume Capability (JSONL for flexibility with varying params)
        self.progress_file = output_dir / "hpo_progress.jsonl"
        self._load_progress()
        
        # 3. Prepare Data
        X = train_df.drop(columns=self.config['data']['drop_columns'] + 
                          [self.config['data']['target_sin'], 
                           self.config['data']['target_cos'],
                           'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'], errors='ignore')
        
        # Targets
        y = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        # Stratification key
        stratify_col = train_df['combined_bin'] if 'combined_bin' in train_df.columns else None
        
        # 4. Grid Search Loop
        grids = self.hpo_config.get('grids', {})
        config_counter = 0
        
        for model_name, param_grid in grids.items():
            self.logger.info(f"Expanding grid for {model_name}...")
            combinations = list(ParameterGrid(param_grid))
            
            for params in combinations:
                config_counter += 1
                
                # Generate unique hash
                # Ensure params are JSON serializable for the hash
                config_signature = json.dumps({'model': model_name, 'params': params}, sort_keys=True, cls=NumpyEncoder)
                config_hash = hashlib.md5(config_signature.encode()).hexdigest()
                
                # Check Resume
                if config_hash in self.completed_hashes:
                    continue
                
                # Evaluate
                try:
                    cv_results = self._evaluate_config_cv(model_name, params, X, y, stratify_col)
                    status = "success"
                except Exception as e:
                    self.logger.error(f"HPO Failed for {model_name} {params}: {str(e)}")
                    cv_results = {}
                    status = "failed"

                # Construct Record
                result_entry = {
                    'config_id': config_counter,
                    'config_hash': config_hash,
                    'model_name': model_name,
                    'status': status,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'params': params, 
                }
                
                # Flatten params into record (e.g., param_n_estimators)
                for k, v in params.items():
                    result_entry[f'param_{k}'] = v
                
                # Merge CV metrics
                result_entry.update(cv_results)
                
                # Save Progress
                self._save_progress(result_entry)
                
                cmae_val = cv_results.get('cv_cmae_deg_mean', float('nan'))
                self.logger.info(f"Evaluated {model_name} | CMAE: {cmae_val:.4f}")

        # 5. Finalize - Process all results
        return self._finalize_results(output_dir)

    def _evaluate_config_cv(self, model_name: str, params: dict, X, y, stratify_col) -> dict:
        """
        Perform K-Fold CV and calculate detailed metrics for every fold and aggregates.
        """
        n_splits = self.hpo_config.get('cv_folds', 5)
        
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            next(cv.split(X, stratify_col))
        except (ValueError, Warning):
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify_col = None 

        def run_fold(fold_idx, train_idx, val_idx):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = ModelFactory.create(model_name, params)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            
            # Metrics Calculation
            return self._calculate_fold_metrics(y_val, preds, fold_idx + 1)

        # Generate splits
        if stratify_col is not None:
            splits = list(cv.split(X, stratify_col))
        else:
            splits = list(cv.split(X))
            
        n_jobs = self.config['execution'].get('n_jobs', -1)
        
        # Run Parallel
        fold_results_list = Parallel(n_jobs=n_jobs)(
            delayed(run_fold)(i, train_idx, val_idx) for i, (train_idx, val_idx) in enumerate(splits)
        )
        
        # Aggregate Results
        aggregated_results = {}
        
        # 1. Collect per-fold columns (e.g., fold_1_cmae_deg)
        for fr in fold_results_list:
            aggregated_results.update(fr)
            
        # 2. Compute Mean and Std for each metric type
        if fold_results_list:
            # Extract keys from the first fold to know what metrics we have
            metric_names = set()
            for k in fold_results_list[0].keys():
                # Remove 'fold_X_' prefix
                parts = k.split('_')
                if len(parts) > 2:
                    metric_name = '_'.join(parts[2:]) # fold_1_cmae_deg -> cmae_deg
                    metric_names.add(metric_name)
                
            for metric in metric_names:
                values = []
                for fr in fold_results_list:
                    # Find the key for this metric in this fold result
                    for k, v in fr.items():
                        if k.endswith(metric) and not k.startswith("cv_"):
                            values.append(v)
                
                if values:
                    aggregated_results[f'cv_{metric}_mean'] = np.mean(values)
                    aggregated_results[f'cv_{metric}_std'] = np.std(values)
            
        return aggregated_results

    def _calculate_fold_metrics(self, y_true_df, y_pred_raw, fold_num) -> dict:
        """Calculate extensive metrics for a single fold."""
        # 1. Reconstruct Angles
        y_true_np = y_true_df.values
        true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
        pred_angle = reconstruct_angle(y_pred_raw[:, 0], y_pred_raw[:, 1])
        
        # 2. Angular Errors (Shortest arc)
        raw_diff = np.abs(true_angle - pred_angle)
        angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
        
        # 3. Circular Metrics
        cmae = np.mean(angle_error)
        crmse = np.sqrt(np.mean(angle_error ** 2))
        max_err = np.max(angle_error)
        
        # 4. Accuracy Metrics
        acc_0 = np.mean(angle_error <= 0.001) * 100
        acc_5 = np.mean(angle_error <= 5.0) * 100
        acc_10 = np.mean(angle_error <= 10.0) * 100
        
        # 5. Component Metrics (Sin/Cos)
        mae_sin = mean_absolute_error(y_true_np[:, 0], y_pred_raw[:, 0])
        mae_cos = mean_absolute_error(y_true_np[:, 1], y_pred_raw[:, 1])
        rmse_sin = np.sqrt(mean_squared_error(y_true_np[:, 0], y_pred_raw[:, 0]))
        rmse_cos = np.sqrt(mean_squared_error(y_true_np[:, 1], y_pred_raw[:, 1]))
        
        r2_avg = (r2_score(y_true_np[:, 0], y_pred_raw[:, 0]) + 
                  r2_score(y_true_np[:, 1], y_pred_raw[:, 1])) / 2
        
        expl_var = (explained_variance_score(y_true_np[:, 0], y_pred_raw[:, 0]) + 
                    explained_variance_score(y_true_np[:, 1], y_pred_raw[:, 1])) / 2
        
        prefix = f"fold_{fold_num}_"
        return {
            f"{prefix}cmae_deg": cmae,
            f"{prefix}crmse_deg": crmse,
            f"{prefix}max_error_deg": max_err,
            f"{prefix}accuracy_within_0deg": acc_0,
            f"{prefix}accuracy_within_5deg": acc_5,
            f"{prefix}accuracy_within_10deg": acc_10,
            f"{prefix}mae_sin": mae_sin,
            f"{prefix}mae_cos": mae_cos,
            f"{prefix}rmse_sin": rmse_sin,
            f"{prefix}rmse_cos": rmse_cos,
            f"{prefix}r2_score": r2_avg,
            f"{prefix}explained_variance": expl_var
        }

    def _finalize_results(self, output_dir: Path) -> dict:
        """Read progress, format columns, save final Excel, and return best config."""
        if not self.progress_file.exists():
            return {}
            
        # Read JSONL into DataFrame
        data = []
        with open(self.progress_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
                    
        df = pd.DataFrame(data)
        
        if df.empty:
            return {}

        # Sort columns to match the desired format
        cols = list(df.columns)
        
        def col_sort_key(c):
            if c in ['config_id', 'config_hash', 'model_name', 'status', 'timestamp']:
                return (0, c)
            if c.startswith('param_'):
                return (1, c)
            if c.startswith('cv_'):
                # prioritization within cv
                if 'cmae' in c: return (2, 0, c)
                if 'crmse' in c: return (2, 1, c)
                return (2, 2, c)
            if c.startswith('fold_'):
                parts = c.split('_')
                try:
                    fold_num = int(parts[1])
                except:
                    fold_num = 999
                return (3, fold_num, c)
            return (4, c)

        sorted_cols = sorted(cols, key=col_sort_key)
        
        if 'params' in sorted_cols:
            sorted_cols.remove('params')
            sorted_cols.insert(5, 'params') 
            
        final_df = df[sorted_cols]
        
        # Save Excel
        excel_path = output_dir / "all_config_results.xlsx"
        final_df.to_excel(excel_path, index=False)
        self.logger.info(f"Saved comprehensive results to {excel_path}")
        
        # Find Best Config
        metric_col = 'cv_cmae_deg_mean'
        if metric_col in final_df.columns:
            best_row = final_df.loc[final_df[metric_col].idxmin()]
            
            # Helper to safely load params
            params_val = best_row['params']
            if isinstance(params_val, str):
                params_val = json.loads(params_val)
                
            best_config = {
                'model': best_row['model_name'],
                'params': params_val,
                'metrics': {
                    'cmae': best_row['cv_cmae_deg_mean'],
                    'crmse': best_row.get('cv_crmse_deg_mean', 0)
                }
            }
            
            # FIX: Use NumpyEncoder here as well
            with open(output_dir / "best_config.json", 'w') as f:
                json.dump(best_config, f, indent=2, cls=NumpyEncoder)
                
            return best_config
        
        return {}

    def _load_progress(self):
        """Load completed hashes from JSONL."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        if 'config_hash' in entry:
                            self.completed_hashes.add(entry['config_hash'])
                self.logger.info(f"Resumed HPO: {len(self.completed_hashes)} configs already completed.")
            except Exception as e:
                self.logger.warning(f"Could not read HPO progress file: {e}")

    def _save_progress(self, result_entry: dict):
        """Append a single result as a JSON line."""
        # FIX: Use NumpyEncoder to prevent "float32 is not serializable" error
        with open(self.progress_file, 'a') as f:
            f.write(json.dumps(result_entry, cls=NumpyEncoder) + "\n")