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
    Hyperparameter Optimization Engine with Resume Capability & Crash-Proof Snapshot Tracking.
    Performs Grid Search with K-Fold Cross-Validation.
    Saves raw prediction snapshots to disk immediately for Global Error Tracking.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.hpo_config = config.get('hyperparameters', {})
        self.progress_file: Optional[Path] = None
        self.completed_hashes = set()

    def execute(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """
        Execute Grid Search, save snapshots, and generate detailed results.
        """
        self.logger.info("Starting Hyperparameter Optimization (HPO)...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "03_HYPERPARAMETER_OPTIMIZATION"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # [NEW] Setup Snapshot Directory for Global Tracking
        # FIX: Ensure base_dir is wrapped in Path() before using '/' operator
        snapshot_dir = Path(base_dir) / "03_HPO_SEARCH" / "tracking_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        progress_dir = output_dir / "progress"
        results_dir = output_dir / "results"
        progress_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        # 2. Setup Resume Capability
        self.progress_file = progress_dir / "hpo_progress.jsonl"
        self._load_progress()
        
        # 3. Prepare Data
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'], 
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]
        
        # Training Data
        X_train = train_df.drop(columns=drop_cols, errors='ignore')
        y_train = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        # Validation Data (for metrics + snapshot)
        X_val = val_df.drop(columns=drop_cols, errors='ignore')
        y_val = val_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        # Test Data (for metrics + snapshot)
        X_test = test_df.drop(columns=drop_cols, errors='ignore')
        y_test = test_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
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
                
                config_signature = json.dumps({'model': model_name, 'params': params}, sort_keys=True, cls=NumpyEncoder)
                config_hash = hashlib.md5(config_signature.encode()).hexdigest()
                
                if config_hash in self.completed_hashes:
                    continue
                
                try:
                    # A. Perform CV (Selection Metrics)
                    cv_results = self._evaluate_config_cv(model_name, params, X_train, y_train, stratify_col)
                    
                    # B. Train Final Model on Full Train Set (Optimization: Fit once for both Val/Test)
                    model = ModelFactory.create(model_name, params)
                    model.fit(X_train, y_train)
                    
                    # C. Predict & Snapshot (Val)
                    preds_val = model.predict(X_val)
                    val_metrics = self._calculate_metrics_with_prefix(y_val, preds_val, prefix="val")
                    # [NEW] Save Snapshot
                    self._save_snapshot(snapshot_dir, config_counter, "val", val_df.index, y_val, preds_val)
                    
                    # D. Predict & Snapshot (Test)
                    preds_test = model.predict(X_test)
                    test_metrics = self._calculate_metrics_with_prefix(y_test, preds_test, prefix="test")
                    # [NEW] Save Snapshot
                    self._save_snapshot(snapshot_dir, config_counter, "test", test_df.index, y_test, preds_test)
                    
                    status = "success"
                except Exception as e:
                    self.logger.error(f"HPO Failed for {model_name} {params}: {str(e)}")
                    cv_results = {}
                    val_metrics = {}
                    test_metrics = {}
                    status = "failed"

                result_entry = {
                    'config_id': config_counter,
                    'config_hash': config_hash,
                    'model_name': model_name,
                    'status': status,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'params': params, 
                }
                
                # Flatten params
                for k, v in params.items():
                    result_entry[f'param_{k}'] = v
                
                # Merge all metrics
                result_entry.update(cv_results)
                result_entry.update(val_metrics)
                result_entry.update(test_metrics)
                
                self._save_progress(result_entry)
                
                cmae_val = cv_results.get('cv_cmae_deg_mean', float('nan'))
                self.logger.info(f"Evaluated {model_name} | CV CMAE: {cmae_val:.4f}")

        # 5. Finalize
        return self._finalize_results(results_dir)

    def _save_snapshot(self, snapshot_dir: Path, trial_id: int, split_name: str, 
                       indices, y_true_df, preds_raw):
        """
        [NEW] Saves a lightweight CSV snapshot of predictions for Global Tracking.
        """
        try:
            # Reconstruct angles to get meaningful errors immediately
            y_true_np = y_true_df.values
            true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
            pred_angle = reconstruct_angle(preds_raw[:, 0], preds_raw[:, 1])
            
            raw_diff = np.abs(true_angle - pred_angle)
            angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
            
            snapshot_df = pd.DataFrame({
                'row_index': indices,
                'pred_sin': preds_raw[:, 0],
                'pred_cos': preds_raw[:, 1],
                'pred_angle': pred_angle,
                'abs_error': angle_error
            })
            
            # Format: trial_001_val.csv
            filename = f"trial_{trial_id:03d}_{split_name}.csv"
            snapshot_df.to_csv(snapshot_dir / filename, index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to save snapshot for trial {trial_id} ({split_name}): {e}")

    def _evaluate_config_cv(self, model_name: str, params: dict, X, y, stratify_col) -> dict:
        """Perform K-Fold CV and calculate metrics."""
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
            
            return self._calculate_fold_metrics(y_val, preds, fold_idx + 1)

        if stratify_col is not None:
            splits = list(cv.split(X, stratify_col))
        else:
            splits = list(cv.split(X))
            
        n_jobs = self.config['execution'].get('n_jobs', -1)
        
        fold_results_list = Parallel(n_jobs=n_jobs)(
            delayed(run_fold)(i, train_idx, val_idx) for i, (train_idx, val_idx) in enumerate(splits)
        )
        
        aggregated_results = {}
        
        for fr in fold_results_list:
            aggregated_results.update(fr)
            
        if fold_results_list:
            metric_names = set()
            for k in fold_results_list[0].keys():
                parts = k.split('_')
                if len(parts) > 2:
                    metric_name = '_'.join(parts[2:])
                    metric_names.add(metric_name)
                
            for metric in metric_names:
                values = []
                for fr in fold_results_list:
                    for k, v in fr.items():
                        if k.endswith(metric) and not k.startswith("cv_"):
                            values.append(v)
                
                if values:
                    aggregated_results[f'cv_{metric}_mean'] = np.mean(values)
                    aggregated_results[f'cv_{metric}_std'] = np.std(values)
            
        return aggregated_results

    def _calculate_metrics_with_prefix(self, y_true, y_pred, prefix) -> dict:
        """Helper to attach prefix to all metrics."""
        metrics = self._calculate_metrics(y_true, y_pred)
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def _calculate_fold_metrics(self, y_true_df, y_pred_raw, fold_num) -> dict:
        """Calculate metrics for a single fold."""
        metrics = self._calculate_metrics(y_true_df, y_pred_raw)
        prefix = f"fold_{fold_num}_"
        return {f"{prefix}{k}": v for k, v in metrics.items()}

    def _calculate_metrics(self, y_true_df, y_pred_raw) -> dict:
        """Calculate all metrics (Full Original Suite)."""
        y_true_np = y_true_df.values
        true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
        pred_angle = reconstruct_angle(y_pred_raw[:, 0], y_pred_raw[:, 1])
        
        raw_diff = np.abs(true_angle - pred_angle)
        angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
        
        cmae = np.mean(angle_error)
        crmse = np.sqrt(np.mean(angle_error ** 2))
        max_err = np.max(angle_error)
        
        acc_0 = np.mean(angle_error <= 0.001) * 100
        acc_5 = np.mean(angle_error <= 5.0) * 100
        acc_10 = np.mean(angle_error <= 10.0) * 100
        
        mae_sin = mean_absolute_error(y_true_np[:, 0], y_pred_raw[:, 0])
        mae_cos = mean_absolute_error(y_true_np[:, 1], y_pred_raw[:, 1])
        rmse_sin = np.sqrt(mean_squared_error(y_true_np[:, 0], y_pred_raw[:, 0]))
        rmse_cos = np.sqrt(mean_squared_error(y_true_np[:, 1], y_pred_raw[:, 1]))
        
        r2_avg = (r2_score(y_true_np[:, 0], y_pred_raw[:, 0]) + 
                  r2_score(y_true_np[:, 1], y_pred_raw[:, 1])) / 2
        
        expl_var = (explained_variance_score(y_true_np[:, 0], y_pred_raw[:, 0]) + 
                    explained_variance_score(y_true_np[:, 1], y_pred_raw[:, 1])) / 2
        
        return {
            'cmae_deg': cmae,
            'crmse_deg': crmse,
            'max_error_deg': max_err,
            'accuracy_within_0deg': acc_0,
            'accuracy_within_5deg': acc_5,
            'accuracy_within_10deg': acc_10,
            'mae_sin': mae_sin,
            'mae_cos': mae_cos,
            'rmse_sin': rmse_sin,
            'rmse_cos': rmse_cos,
            'r2_score': r2_avg,
            'explained_variance': expl_var
        }

    def _finalize_results(self, output_dir: Path) -> dict:
        """Read progress, format, save, and return best config based on CV metrics."""
        if not self.progress_file.exists():
            return {}
            
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

        cols = list(df.columns)
        
        # Original complex sorting logic restored
        def col_sort_key(c):
            if c in ['config_id', 'config_hash', 'model_name', 'status', 'timestamp']:
                return (0, c)
            if c.startswith('param_'):
                return (1, c)
            if c.startswith('cv_'):
                if 'cmae' in c: return (2, 0, c)
                if 'crmse' in c: return (2, 1, c)
                return (2, 2, c)
            if c.startswith('val_'):
                return (3, c)
            if c.startswith('test_'):
                return (4, c)
            if c.startswith('fold_'):
                parts = c.split('_')
                try:
                    fold_num = int(parts[1])
                except:
                    fold_num = 999
                return (5, fold_num, c)
            return (6, c)

        sorted_cols = sorted(cols, key=col_sort_key)
        
        if 'params' in sorted_cols:
            sorted_cols.remove('params')
            sorted_cols.insert(5, 'params')
            
        final_df = df[sorted_cols]
        
        excel_path = output_dir / "all_configurations.xlsx"
        final_df.to_excel(excel_path, index=False)
        self.logger.info(f"Saved comprehensive results to {excel_path}")
        
        # CRITICAL: Select best based on CV metrics ONLY
        metric_col = 'cv_cmae_deg_mean'
        tie_breaker_col = 'cv_max_error_deg_mean'
        
        if metric_col in final_df.columns:
            # Sort by primary metric
            sorted_df = final_df.sort_values(metric_col, ascending=True)
            
            # Get best CMAE value
            best_cmae = sorted_df.iloc[0][metric_col]
            
            # Filter all configs with this exact CMAE (for tie-breaking)
            best_candidates = sorted_df[sorted_df[metric_col] == best_cmae]
            
            if len(best_candidates) > 1 and tie_breaker_col in final_df.columns:
                # Tie-breaker: use max_error
                best_row = best_candidates.sort_values(tie_breaker_col, ascending=True).iloc[0]
                self.logger.info(f"Tie-breaker applied: {len(best_candidates)} configs with CMAE={best_cmae:.4f}")
            else:
                best_row = sorted_df.iloc[0]
            
            params_val = best_row['params']
            if isinstance(params_val, str):
                params_val = json.loads(params_val)
                
            best_config = {
                'model': best_row['model_name'],
                'params': params_val,
                'cv_metrics': {
                    'cmae': best_row.get('cv_cmae_deg_mean', 0),
                    'crmse': best_row.get('cv_crmse_deg_mean', 0),
                    'max_error': best_row.get('cv_max_error_deg_mean', 0)
                },
                'val_metrics': {
                    'cmae': best_row.get('val_cmae_deg', 0),
                    'crmse': best_row.get('val_crmse_deg', 0)
                },
                'test_metrics': {
                    'cmae': best_row.get('test_cmae_deg', 0),
                    'crmse': best_row.get('test_crmse_deg', 0)
                },
                'selection_note': 'Selected based on cv_cmae_deg_mean with cv_max_error_deg_mean as tie-breaker'
            }
            
            with open(output_dir / "best_configuration.json", 'w') as f:
                json.dump(best_config, f, indent=2, cls=NumpyEncoder)
                
            self.logger.info(f"Best config: {best_row['model_name']} | CV CMAE: {best_cmae:.4f}")
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
        with open(self.progress_file, 'a') as f:
            f.write(json.dumps(result_entry, cls=NumpyEncoder) + "\n")