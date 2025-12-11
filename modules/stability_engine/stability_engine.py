import pandas as pd
import numpy as np
import logging
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Import all core engines to orchestrate runs
from modules.data_manager import DataManager
from modules.split_engine import SplitEngine
from modules.hpo_search_engine import HPOSearchEngine
from modules.training_engine import TrainingEngine
from modules.prediction_engine import PredictionEngine
from modules.evaluation_engine import EvaluationEngine
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils.file_io import save_dataframe

class StabilityEngine(BaseEngine):
    """
    Orchestrates multiple full pipeline runs with varying random seeds 
    to assess model and metric stability.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.stab_config = config.get('stability', {})
        self.enabled = self.stab_config.get('enabled', False)

    def _get_engine_directory_name(self) -> str:
        return "11_STABILITY_ANALYSIS"
        
    @handle_engine_errors("Stability Analysis")
    def execute(self, raw_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """
        Execute N full pipeline runs to determine stability.
        
        Parameters:
            raw_df: The raw dataframe loaded by DataManager.
            run_id: Run identifier.
        """
        if not self.enabled:
            self.logger.info("Stability Analysis disabled in config.")
            return {}

        num_runs = self.stab_config.get('num_runs', 5)
        self.logger.info(f"Starting Stability Analysis: {num_runs} runs...")
        
        results_collection = []
        feature_sets = []
        
        # 2. Loop Runs
        # We use the master seed from config as a starting point
        base_seed = self.config['splitting']['seed']
        
        for i in range(num_runs):
            run_idx = i + 1
            # Generate a deterministic but different seed for each run
            current_seed = base_seed + (run_idx * 100)
            self.logger.info(f"--- Stability Run {run_idx}/{num_runs} (Seed: {current_seed}) ---")
            
            # FIX #51: Create a DEEP COPY of the config to prevent pollution across runs
            run_config = copy.deepcopy(self.config)
            
            # Update seeds in the isolated config
            run_config['splitting']['seed'] = current_seed
            if '_internal_seeds' in run_config:
                run_config['_internal_seeds']['split'] = current_seed
                run_config['_internal_seeds']['cv'] = current_seed + 1
                run_config['_internal_seeds']['model'] = current_seed + 2
            
            # FIX #53: Isolate output directories for this run to prevent file collisions
            stability_run_dir = self.output_dir / "runs" / f"run_{run_idx}"
            stability_run_dir.mkdir(parents=True, exist_ok=True)
            run_config['outputs']['base_results_dir'] = str(stability_run_dir)

            try:
                # FIX #51: Pass a DEEP COPY of raw_df. 
                split_engine = SplitEngine(run_config, self.logger)
                train_df, val_df, test_df = split_engine.execute(raw_df.copy(), run_id)
                
                # B. HPO / Model Selection
                if run_config['hyperparameters']['enabled']:
                    hpo_engine = HPOSearchEngine(run_config, self.logger)
                    best_config = hpo_engine.execute(train_df, val_df, test_df, run_id)
                else:
                    model_name = run_config.get('models', {}).get('native', [{}])[0]
                    if not isinstance(model_name, str):
                         model_name = 'ExtraTreesRegressor'
                    best_config = {'model': model_name, 'params': {}}
                    
                # C. Training
                training_engine = TrainingEngine(run_config, self.logger)
                model = training_engine.execute(train_df, best_config, run_id)
                
                # D. Evaluation
                pred_engine = PredictionEngine(run_config, self.logger)
                preds_test = pred_engine.execute(model, test_df, "stability_test", run_id)
                
                eval_engine = EvaluationEngine(run_config, self.logger)
                metrics = eval_engine.compute_metrics(preds_test)
                
                # Flatten and prefix keys
                flat_metrics = {
                    'test_cmae': metrics.get('cmae', float('nan')),
                    'test_crmse': metrics.get('crmse', float('nan')),
                    'test_accuracy_5': metrics.get('accuracy_at_5deg', float('nan'))
                }
                
                # Identify features used (exclude targets and metadata)
                drop_cols = run_config['data'].get('drop_columns', []) + [
                    run_config['data'].get('target_sin'), 
                    run_config['data'].get('target_cos'),
                    run_config['data'].get('hs_column', 'Hs'),
                    'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
                ]
                features = [c for c in train_df.columns if c not in drop_cols]
                
                results_collection.append({
                    'run': run_idx,
                    'seed': current_seed,
                    **flat_metrics
                })
                feature_sets.append(set(features))
                
            except Exception as e:
                self.logger.error(f"Stability Run {run_idx} failed: {e}", exc_info=True)
                continue
                
        if not results_collection:
            self.logger.error("All stability runs failed.")
            return {}

        # 3. Analyze Results
        results_df = pd.DataFrame(results_collection)
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        save_dataframe(results_df, self.output_dir / "stability_results_raw.parquet", excel_copy=excel_copy, index=False)
        
        metric_cols = [c for c in results_df.columns if c not in ['run', 'seed']]
        
        if not metric_cols:
            self.logger.warning("No metrics found to analyze stability.")
            return {}

        stats = results_df[metric_cols].describe().T[['mean', 'std', 'min', 'max']]
        stats['cv_pct'] = (stats['std'] / stats['mean'].replace(0, 1e-9)) * 100 
        save_dataframe(stats, self.output_dir / "stability_metrics_summary.parquet", excel_copy=excel_copy, index=False)
        
        # Feature Stability (Jaccard Index)
        jaccard_score = self._compute_feature_stability(feature_sets)
        
        summary = {
            'num_runs': num_runs,
            'successful_runs': len(results_df),
            'cmae_mean': stats.loc['test_cmae', 'mean'] if 'test_cmae' in stats.index else 0.0,
            'cmae_std': stats.loc['test_cmae', 'std'] if 'test_cmae' in stats.index else 0.0,
            'feature_stability_jaccard': jaccard_score
        }
        
        self.logger.info(f"Stability Complete. CMAE Mean: {summary['cmae_mean']:.4f} ± {summary['cmae_std']:.4f}")
        return summary

    def _compute_feature_stability(self, feature_sets: List[set]) -> float:
        """
        Compute average pairwise Jaccard index for selected features.
        """
        if not feature_sets:
            return 0.0
            
        scores = []
        n = len(feature_sets)
        if n < 2:
            return 1.0 
            
        for i in range(n):
            for j in range(i+1, n):
                a, b = feature_sets[i], feature_sets[j]
                denom = len(a | b)
                if denom == 0:
                    continue
                scores.append(len(a & b) / denom)
                
        return float(np.mean(scores)) if scores else 0.0
