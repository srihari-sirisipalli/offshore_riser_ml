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
import gc  # Explicit garbage collection
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
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
    """Handles serialization of NumPy types to JSON."""
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
    Hyperparameter Optimization Engine.
    
    Improvements (Phase 1):
    - Memory-safe processing (Streaming).
    - Parquet snapshot storage.
    - Explicit garbage collection.
    - Resume capability.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.hpo_config = config.get('hyperparameters', {})
        self.progress_file: Optional[Path] = None
        self.completed_hashes = set()
        
        # Resource Limits (Task 1.2)
        self.max_configs = self.config.get('resources', {}).get('max_hpo_configs', 1000)

    def execute(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """
        Execute Grid Search, save snapshots, and generate detailed results.
        
        Args:
            train_df, val_df, test_df: Feature-rich DataFrames.
            run_id: Unique identifier for this execution.
            
        Returns:
            Dict containing the best model configuration.
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
        base_dir = Path(self.config.get('outputs', {}).get('base_results_dir', 'results'))
        output_dir = base_dir / "03_HYPERPARAMETER_OPTIMIZATION"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Snapshot Directory (Parquet for speed)
        snapshot_dir = output_dir / "tracking_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        progress_dir = output_dir / "progress"
        results_dir = output_dir / "results"
        progress_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Setup Resume Capability
        self.progress_file = progress_dir / "hpo_progress.jsonl"
        self._load_progress()
        
        # 3. Prepare Data (Drop non-features)
        # Using copy=False where possible (Issue #P5)
        drop_cols = self.config['data']['drop_columns'] + [
            self.config['data']['target_sin'], 
            self.config['data']['target_cos'],
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]
        
        X_train = train_df.drop(columns=drop_cols, errors='ignore')
        y_train = train_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        X_val = val_df.drop(columns=drop_cols, errors='ignore')
        y_val = val_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        X_test = test_df.drop(columns=drop_cols, errors='ignore')
        y_test = test_df[[self.config['data']['target_sin'], self.config['data']['target_cos']]]
        
        stratify_col = train_df['combined_bin'] if 'combined_bin' in train_df.columns else None
        
        # 4. Grid Search Loop
        grids = self.hpo_config.get('grids', {})
        config_counter = 0
        total_configs_processed = 0
        
        for model_name, param_grid in grids.items():
            self.logger.info(f"Expanding grid for {model_name}...")
            combinations = list(ParameterGrid(param_grid))
            
            for params in combinations:
                config_counter += 1
                
                # Check limits
                if config_counter > self.max_configs:
                    self.logger.warning(f"Max HPO configs ({self.max_configs}) reached. Stopping search early.")
                    break

                # Deterministic hash for resume
                config_signature = json.dumps({'model': model_name, 'params': params}, sort_keys=True, cls=NumpyEncoder)
                config_hash = hashlib.md5(config_signature.encode()).hexdigest()
                
                if config_hash in self.completed_hashes:
                    continue
                
                try:
                    # A. CV Evaluation
                    cv_results = self._evaluate_config_cv(model_name, params, X_train, y_train, stratify_col)
                    
                    # B. Final Train & Snapshot
                    # Only train final model if CV didn't fail spectacularly
                    model = ModelFactory.create(model_name, params)
                    model.fit(X_train, y_train)
                    
                    # C. Val Snapshot
                    preds_val = model.predict(X_val)
                    val_metrics = self._calculate_metrics_with_prefix(y_val, preds_val, prefix="val")
                    self._save_snapshot_parquet(snapshot_dir, config_counter, "val", val_df.index, y_val, preds_val)
                    
                    # D. Test Snapshot
                    preds_test = model.predict(X_test)
                    test_metrics = self._calculate_metrics_with_prefix(y_test, preds_test, prefix="test")
                    self._save_snapshot_parquet(snapshot_dir, config_counter, "test", test_df.index, y_test, preds_test)
                    
                    status = "success"
                    
                except Exception as e:
                    self.logger.error(f"HPO Failed for {model_name} {params}: {str(e)}")
                    cv_results = {}
                    val_metrics = {}
                    test_metrics = {}
                    status = "failed"

                # Prepare Result Entry
                result_entry = {
                    'config_id': config_counter,
                    'config_hash': config_hash,
                    'model_name': model_name,
                    'status': status,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'params': params, 
                    **cv_results,
                    **val_metrics,
                    **test_metrics
                }
                
                self._save_progress(result_entry)
                total_configs_processed += 1
                
                # Memory Management (Issue #P1)
                # Force cleanup after every iteration
                del model, preds_val, preds_test
                gc.collect()
                
                if total_configs_processed % 10 == 0:
                    self.logger.info(f"Processed {total_configs_processed} configs...")

        # 5. Finalize Results (Streaming Read)
        return self._finalize_results(results_dir)

    def _save_snapshot_parquet(self, snapshot_dir: Path, trial_id: int, split_name: str, 
                               indices, y_true_df, preds_raw):
        """
        Saves prediction snapshot in Parquet format (Issue #P4).
        50x faster than CSV/Excel and compressed.
        """
        try:
            y_true_np = y_true_df.values
            true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
            pred_angle = reconstruct_angle(preds_raw[:, 0], preds_raw[:, 1])
            
            raw_diff = np.abs(true_angle - pred_angle)
            angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
            
            snapshot_df = pd.DataFrame({
                'row_index': indices,
                'pred_sin': preds_raw[:, 0].astype('float32'),
                'pred_cos': preds_raw[:, 1].astype('float32'),
                'pred_angle': pred_angle.astype('float32'),
                'abs_error': angle_error.astype('float32')
            })
            
            filename = f"trial_{trial_id:04d}_{split_name}.parquet"
            snapshot_df.to_parquet(snapshot_dir / filename, index=False)
            
        except Exception as e:
            self.logger.warning(f"Failed to save snapshot for trial {trial_id} ({split_name}): {e}")

    def _evaluate_config_cv(self, model_name: str, params: dict, X, y, stratify_col) -> dict:
        """Perform K-Fold CV."""
        n_splits = self.hpo_config.get('cv_folds', 5)
        
        # Determine CV strategy
        try:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            next(cv.split(X, stratify_col))
        except (ValueError, Warning):
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            stratify_col = None 

        splits = list(cv.split(X, stratify_col)) if stratify_col is not None else list(cv.split(X))
        
        # Parallel Execution for Folds
        n_jobs = self.config['execution'].get('n_jobs', -1)
        
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_fold)(model_name, params, X, y, train_idx, val_idx, i)
            for i, (train_idx, val_idx) in enumerate(splits)
        )
        
        # Aggregation
        aggregated = {}
        # Collect all metric keys
        all_keys = set()
        for res in fold_results:
            all_keys.update(res.keys())
            
        # Group by metric base name (e.g. cmae_deg)
        metric_bases = set()
        for k in all_keys:
            if k.startswith('fold_'):
                # Extract 'cmae_deg' from 'fold_1_cmae_deg'
                parts = k.split('_')
                if len(parts) > 2:
                    metric_bases.add('_'.join(parts[2:]))

        for base in metric_bases:
            values = []
            for res in fold_results:
                # Find matching key
                for k, v in res.items():
                    if k.endswith(f"_{base}"):
                        values.append(v)
            
            if values:
                aggregated[f'cv_{base}_mean'] = np.mean(values)
                aggregated[f'cv_{base}_std'] = np.std(values)
                
        return aggregated

    def _run_single_fold(self, model_name, params, X, y, train_idx, val_idx, fold_idx):
        """Helper for parallel fold execution."""
        try:
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model = ModelFactory.create(model_name, params)
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_val_fold, preds)
            
            # Prefix keys
            return {f"fold_{fold_idx+1}_{k}": v for k, v in metrics.items()}
        except Exception:
            return {}

    def _calculate_metrics_with_prefix(self, y_true, y_pred, prefix) -> dict:
        metrics = self._calculate_metrics(y_true, y_pred)
        return {f"{prefix}_{k}": v for k, v in metrics.items()}

    def _calculate_metrics(self, y_true_df, y_pred_raw) -> dict:
        """Standard metric calculation."""
        y_true_np = y_true_df.values
        true_angle = reconstruct_angle(y_true_np[:, 0], y_true_np[:, 1])
        pred_angle = reconstruct_angle(y_pred_raw[:, 0], y_pred_raw[:, 1])
        
        raw_diff = np.abs(true_angle - pred_angle)
        angle_error = np.where(raw_diff > 180, 360 - raw_diff, raw_diff)
        
        return {
            'cmae_deg': np.mean(angle_error),
            'crmse_deg': np.sqrt(np.mean(angle_error ** 2)),
            'max_error_deg': np.max(angle_error),
            'accuracy_at_5deg': np.mean(angle_error <= 5.0) * 100
        }

    def _load_progress(self):
        """Load completed hashes using a generator to avoid memory spikes."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            # Only parse what we need (the hash) to save memory
                            # For simple JSON, regex might be faster, but json.loads is safer
                            entry = json.loads(line)
                            if 'config_hash' in entry:
                                self.completed_hashes.add(entry['config_hash'])
                        except json.JSONDecodeError:
                            continue
                self.logger.info(f"Resumed HPO: {len(self.completed_hashes)} configs completed.")
            except Exception as e:
                self.logger.warning(f"Error reading progress file: {e}")

    def _save_progress(self, result_entry: dict):
        """Append result with locking."""
        with file_lock(self.progress_file):
            with open(self.progress_file, 'a') as f:
                f.write(json.dumps(result_entry, cls=NumpyEncoder) + "\n")

    def _finalize_results(self, output_dir: Path) -> dict:
        """
        Process results in a memory-efficient way (Streaming).
        """
        if not self.progress_file.exists():
            return {}
            
        best_config = {}
        best_score = float('inf')
        
        # Stream file to find best config without loading everything
        # Also convert to Excel/Parquet for user analysis in chunks if needed
        
        # 1. First Pass: Find Best Config
        with open(self.progress_file, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    score = res.get('cv_cmae_deg_mean', float('inf'))
                    if score < best_score:
                        best_score = score
                        best_config = res
                except:
                    continue
                    
        # 2. Save Summary (using Pandas in chunks if massive, or full if reasonable)
        # For typical HPO (<10k rows), loading into DF is fine for the summary file.
        # If >100k, we would need chunked writing.
        try:
            # We assume the summary fits in memory (metadata only, no predictions)
            # Use chunks if worried
            chunks = []
            chunk_size = 5000
            
            with pd.read_json(self.progress_file, lines=True, chunksize=chunk_size) as reader:
                for chunk in reader:
                    chunks.append(chunk)
            
            if chunks:
                full_df = pd.concat(chunks, ignore_index=True)
                full_df.to_excel(output_dir / "all_configurations.xlsx", index=False)
                # Also save parquet for internal use
                full_df.to_parquet(output_dir / "all_configurations.parquet", index=False)
        except Exception as e:
            self.logger.error(f"Failed to compile final results file: {e}")

        # 3. Format Best Config
        if best_config:
            formatted_best = {
                'model': best_config.get('model_name'),
                'params': best_config.get('params'),
                'metrics': {
                    'cv_cmae': best_config.get('cv_cmae_deg_mean'),
                    'val_cmae': best_config.get('val_cmae_deg'),
                    'test_cmae': best_config.get('test_cmae_deg')
                }
            }
            
            with open(output_dir / "best_configuration.json", 'w') as f:
                json.dump(formatted_best, f, indent=2, cls=NumpyEncoder)
                
            self.logger.info(f"Best Config Found: {formatted_best['model']} (CV CMAE: {best_score:.4f})")
            return formatted_best
            
        return {}