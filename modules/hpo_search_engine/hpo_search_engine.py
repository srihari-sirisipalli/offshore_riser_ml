import pandas as pd
import numpy as np
import json
import os
import time
import hashlib
import logging
import datetime
import shutil
import contextlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from joblib import Parallel, delayed

from sklearn.model_selection import StratifiedKFold, KFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from modules.model_factory import ModelFactory
from utils.circular_metrics import compute_cmae, compute_crmse, reconstruct_angle

# --- Helper: Safe File Locking ---
@contextlib.contextmanager
def file_lock(lock_file: Path, timeout: int = 60, poll_interval: float = 0.1):
    """
    A cross-platform file locking mechanism using a directory (atomic on most OS).
    Prevents race conditions when writing to the progress file.
    """
    lock_dir = lock_file.parent / (lock_file.name + ".lock")
    start_time = time.time()
    
    while True:
        try:
            lock_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            if time.time() - start_time > timeout:
                # Force break lock if timeout exceeded (stale lock assumption)
                logging.warning(f"Lock timeout expired for {lock_file}. Forcing release.")
                try:
                    shutil.rmtree(lock_dir)
                except OSError:
                    pass # Race condition on removal
            time.sleep(poll_interval)
        except OSError:
            time.sleep(poll_interval)
            
    try:
        yield
    finally:
        try:
            shutil.rmtree(lock_dir)
        except OSError:
            pass

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32)):
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
        
        Parameters:
            train_df, val_df, test_df: DataFrames from SplitEngine.
            run_id: Identifier for the current run.
        """
        if not self.hpo_config.get('enabled', True):
            self.logger.info("HPO disabled. Using default model configuration.")
            default_model = self.config.get('models', {}).get('native', [{}])[0]
            return {
                'model': default_model,
                'params': {}
            }

        self.logger.info("Starting Hyperparameter Optimization (HPO)...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "03_HYPERPARAMETER_OPTIMIZATION"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # [NEW] Setup Snapshot Directory for Global Tracking
        snapshot_dir = Path(base_dir) / "03_HPO_SEARCH" / "tracking_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        progress_dir = output_dir / "progress"
        results_dir = output_dir / "results"
        progress_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
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
                
                # Deterministic hash for resume capability
                config_signature = json.dumps({'model': model_name, 'params': params}, sort_keys=True, cls=NumpyEncoder)
                config_hash = hashlib.md5(config_signature.encode()).hexdigest()
                
                if config_hash in self.completed_hashes:
                    continue
                
                try:
                    # A. Perform CV (Selection Metrics)
                    cv_results = self._evaluate_config_cv(model_name, params, X_train, y_train, stratify_col)
                    
                    # B. Train Final Model on Full Train Set
                    model = ModelFactory.create(model_name, params)
                    model.fit(X_train, y_train)
                    
                    # C. Predict & Snapshot (Val)
                    preds_val = model.predict(X_val)
                    val_metrics = self._calculate_metrics_with_prefix(y_val, preds_val, prefix="val")
                    self._save_snapshot(snapshot_dir, config_counter, "val", val_df.index, y_val, preds_val)
                    
                    # D. Predict & Snapshot (Test)
                    preds_test = model.predict(X_test)
                    test_metrics = self._calculate_metrics_with_prefix(y_test, preds_test, prefix="test")
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
                
                # Flatten params for easy analysis
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
        """Saves a lightweight CSV snapshot of predictions for Global Tracking."""
        try:
            # Reconstruct angles for immediate error analysis
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
            
            filename = f"trial_{trial_id:03d}_{split_name}.csv"
            snapshot_df.to_csv(snapshot_dir / filename, index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to save snapshot for trial {trial_id} ({split_name}): {e}")

    def _evaluate_config_cv(self, model_name: str, params: dict, X, y, stratify_col) -> dict:
        """Perform K-Fold CV and calculate metrics."""
        n_splits = self.hpo_config.get('cv_folds', 5)
        
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            # Check if stratification is possible (requires at least n_splits members per class)
            next(cv.split(X, stratify_col))
        except (ValueError, Warning):
            # Fallback if stratification fails (e.g., extremely rare bins)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify_col = None 

        # Helper for Parallel Execution
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
            # Robust Metric Aggregation (Fix #80)
            # Keys are formatted like 'fold_1_cmae_deg'
            # We extract 'cmae_deg' by dropping 'fold_N_'
            base_metrics = set('_'.join(k.split('_')[2:]) for k in fold_results_list[0])

            for metric in base_metrics:
                values = [
                    v for fr in fold_results_list 
                    for k, v in fr.items() 
                    if '_'.join(k.split('_')[2:]) == metric
                ]
                
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
        # Reconstruct angles
        true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
        pred_angle = reconstruct_angle(y_pred_raw[:, 0], y_pred_raw[:, 1])
        
        raw_diff = np.abs(true_angle - pred_angle)
        angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
        
        cmae = np.mean(angle_error)
        crmse = np.sqrt(np.mean(angle_error ** 2))
        max_err = np.max(angle_error)
        
        # Accuracy Bands
        acc_0 = np.mean(angle_error <= 0.001) * 100
        acc_5 = np.mean(angle_error <= 5.0) * 100
        acc_10 = np.mean(angle_error <= 10.0) * 100
        
        # Component Metrics
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
        output_dir.mkdir(parents=True, exist_ok=True)
        if not self.progress_file or not self.progress_file.exists():
            return {}
            
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        chunks = []
        
        with open(self.progress_file, 'r') as f:
            chunk = []
            for i, line in enumerate(f):
                try:
                    chunk.append(json.loads(line))
                    if len(chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(chunk))
                        chunk = []
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Skipping corrupted line {i+1} in HPO progress file: {e}")
            
            if chunk:
                chunks.append(pd.DataFrame(chunk))
        
        if not chunks:
            return {}
            
        df = pd.concat(chunks, ignore_index=True)
        if df.empty:
            return {}

        cols = list(df.columns)
        
        # Helper for sorting columns nicely in the Excel output
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
                try: fold_num = int(parts[1])
                except: fold_num = 999
                return (5, fold_num, c)
            return (6, c)

        sorted_cols = sorted(cols, key=col_sort_key)
        
        # Ensure params column is visible
        if 'params' in sorted_cols:
            sorted_cols.remove('params')
            sorted_cols.insert(5, 'params')
            
        final_df = df[sorted_cols]
        
        excel_path = output_dir / "all_configurations.xlsx"
        final_df.to_excel(excel_path, index=False)
        self.logger.info(f"Saved comprehensive results to {excel_path}")
        
        # Select best based on CV CMAE
        metric_col = 'cv_cmae_deg_mean'
        tie_breaker_col = 'cv_max_error_deg_mean'
        
        if metric_col in final_df.columns:
            sorted_df = final_df.sort_values(metric_col, ascending=True)
            best_cmae = sorted_df.iloc[0][metric_col]
            
            # Tie-breaker logic
            best_candidates = sorted_df[sorted_df[metric_col] == best_cmae]
            
            if len(best_candidates) > 1 and tie_breaker_col in final_df.columns:
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
                'selection_note': 'Selected based on cv_cmae_deg_mean'
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
        """Append a single result as a JSON line with robust file locking."""
        # FIX #5: Use file lock to prevent race conditions from parallel processes
        with file_lock(self.progress_file):
            with open(self.progress_file, 'a') as f:
                f.write(json.dumps(result_entry, cls=NumpyEncoder) + "\n")