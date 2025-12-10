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

class StabilityEngine:
    """
    Orchestrates multiple full pipeline runs with varying random seeds 
    to assess model and metric stability.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.stab_config = config.get('stability', {})
        self.enabled = self.stab_config.get('enabled', False)
        
    def run_stability_analysis(self, raw_df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
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
        
        # 1. Setup Main Output Directory for Stability Reports
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "11_ADVANCED_ANALYTICS" / "stability"
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            # (e.g., prevent Run 2 from overwriting Run 1's split files)
            stability_run_dir = output_dir / "runs" / f"run_{run_idx}"
            stability_run_dir.mkdir(parents=True, exist_ok=True)
            run_config['outputs']['base_results_dir'] = str(stability_run_dir)

            try:
                # FIX #51: Pass a DEEP COPY of raw_df. 
                # SplitEngine modifies df in-place (adding bin columns). 
                # Without deep copy, Run 2 receives polluted data from Run 1.
                metrics, feats = self._execute_pipeline_run(
                    raw_df.copy(), 
                    run_config, 
                    f"{run_id}_stab_{run_idx}", 
                    run_idx
                )
                
                results_collection.append({
                    'run': run_idx,
                    'seed': current_seed,
                    **metrics
                })
                feature_sets.append(set(feats))
                
            except Exception as e:
                self.logger.error(f"Stability Run {run_idx} failed: {e}", exc_info=True)
                continue
                
        if not results_collection:
            self.logger.error("All stability runs failed.")
            return {}

        # 3. Analyze Results
        results_df = pd.DataFrame(results_collection)
        results_df.to_excel(output_dir / "stability_results_raw.xlsx", index=False)
        
        # Metric Stability (Mean/Std)
        # FIX #98: Filter to only metric columns before calculating stats
        # Previously, it tried to calculate mean of 'run' index and 'seed'.
        metric_cols = [c for c in results_df.columns if c not in ['run', 'seed']]
        
        if not metric_cols:
            self.logger.warning("No metrics found to analyze stability.")
            return {}

        stats = results_df[metric_cols].describe().T[['mean', 'std', 'min', 'max']]
        # Calculate Coefficient of Variation (CV) safely
        stats['cv_pct'] = (stats['std'] / stats['mean'].replace(0, 1e-9)) * 100 
        stats.to_excel(output_dir / "stability_metrics_summary.xlsx")
        
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

    def _execute_pipeline_run(self, raw_df: pd.DataFrame, config: dict, run_id: str, run_idx: int) -> Tuple[Dict, List]:
        """
        Executes Data -> Split -> HPO -> Train -> Eval for a single seed.
        Returns (metrics_dict, selected_features_list).
        """
        # A. Split
        # SplitEngine will verify stratification and write new files to the isolated run dir
        split_engine = SplitEngine(config, self.logger)
        train_df, val_df, test_df = split_engine.execute(raw_df, run_id)
        
        # B. HPO / Model Selection
        if config['hyperparameters']['enabled']:
            hpo_engine = HPOSearchEngine(config, self.logger)
            # FIX #53: Updated call signature to match main pipeline (3 dfs)
            best_config = hpo_engine.execute(train_df, val_df, test_df, run_id)
        else:
            # Fallback if HPO is disabled
            model_name = config.get('models', {}).get('native', [{}])[0]
            if not isinstance(model_name, str):
                 model_name = 'ExtraTreesRegressor' # Default fallback
            best_config = {'model': model_name, 'params': {}}
            
        # C. Training
        training_engine = TrainingEngine(config, self.logger)
        model = training_engine.train(train_df, best_config, run_id)
        
        # D. Evaluation
        pred_engine = PredictionEngine(config, self.logger)
        preds_test = pred_engine.predict(model, test_df, "stability_test", run_id)
        
        eval_engine = EvaluationEngine(config, self.logger)
        # Compute metrics directly without saving Excel files for every single run
        metrics = eval_engine.compute_metrics(preds_test)
        
        # Flatten and prefix keys
        flat_metrics = {
            'test_cmae': metrics.get('cmae', float('nan')),
            'test_crmse': metrics.get('crmse', float('nan')),
            'test_accuracy_5': metrics.get('accuracy_at_5deg', float('nan'))
        }
        
        # Identify features used (exclude targets and metadata)
        drop_cols = config['data'].get('drop_columns', []) + [
            config['data'].get('target_sin'), 
            config['data'].get('target_cos'),
            config['data'].get('hs_column', 'Hs'),
            'angle_deg', 'angle_bin', 'hs_bin', 'combined_bin'
        ]
        features = [c for c in train_df.columns if c not in drop_cols]
        
        return flat_metrics, features

    def _compute_feature_stability(self, feature_sets: List[set]) -> float:
        """
        Compute average pairwise Jaccard index for selected features.
        J(A,B) = |A n B| / |A u B|
        """
        if not feature_sets:
            return 0.0
            
        scores = []
        n = len(feature_sets)
        if n < 2:
            return 1.0 # Only 1 run implies perfect stability with itself
            
        for i in range(n):
            for j in range(i + 1, n):
                set_a = feature_sets[i]
                set_b = feature_sets[j]
                
                intersection = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                
                if union == 0:
                    scores.append(0.0)
                else:
                    scores.append(intersection / union)
                    
        return float(np.mean(scores))