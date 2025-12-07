import pandas as pd
import numpy as np
import logging
import copy
from pathlib import Path
from typing import Dict, Any, List
from typing import Tuple, Optional

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
        Execute N full pipeline runs.
        
        Parameters:
            raw_df: The raw dataframe loaded by DataManager (to save reloading).
                    Note: We re-split this every time.
            run_id: Run identifier.
        """
        if not self.enabled:
            self.logger.info("Stability Analysis disabled in config.")
            return {}

        num_runs = self.stab_config.get('num_runs', 5)
        self.logger.info(f"Starting Stability Analysis: {num_runs} runs...")
        
        # 1. Setup Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "09_ADVANCED_ANALYTICS" / "stability"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_collection = []
        feature_sets = []
        
        # 2. Loop Runs
        base_seed = self.config['splitting']['seed']
        
        for i in range(num_runs):
            run_idx = i + 1
            current_seed = base_seed + (run_idx * 100)
            self.logger.info(f"--- Stability Run {run_idx}/{num_runs} (Seed: {current_seed}) ---")
            
            # Create a run-specific config copy
            run_config = copy.deepcopy(self.config)
            run_config['splitting']['seed'] = current_seed
            # Also update internal seeds if used
            if '_internal_seeds' in run_config:
                run_config['_internal_seeds']['split'] = current_seed
                run_config['_internal_seeds']['cv'] = current_seed + 1
                run_config['_internal_seeds']['model'] = current_seed + 2
                
            try:
                # Execute Pipeline for this seed
                metrics, feats = self._execute_pipeline_run(raw_df, run_config, run_id, run_idx)
                
                results_collection.append({
                    'run': run_idx,
                    'seed': current_seed,
                    **metrics
                })
                feature_sets.append(set(feats))
                
            except Exception as e:
                self.logger.error(f"Stability Run {run_idx} failed: {e}")
                continue
                
        if not results_collection:
            self.logger.error("All stability runs failed.")
            return {}

        # 3. Analyze Results
        results_df = pd.DataFrame(results_collection)
        results_df.to_excel(output_dir / "stability_results_raw.xlsx", index=False)
        
        # Metric Stability (Mean/Std)
        stats = results_df.describe().T[['mean', 'std', 'min', 'max']]
        stats['cv_pct'] = (stats['std'] / stats['mean']) * 100 # Coefficient of Variation
        stats.to_excel(output_dir / "stability_metrics_summary.xlsx")
        
        # Feature Stability (Jaccard)
        jaccard_score = self._compute_feature_stability(feature_sets)
        
        summary = {
            'num_runs': num_runs,
            'successful_runs': len(results_df),
            'cmae_mean': stats.loc['test_cmae', 'mean'],
            'cmae_std': stats.loc['test_cmae', 'std'],
            'feature_stability_jaccard': jaccard_score
        }
        
        self.logger.info(f"Stability Complete. CMAE Mean: {summary['cmae_mean']:.2f} ± {summary['cmae_std']:.2f}")
        return summary

    def _execute_pipeline_run(self, raw_df: pd.DataFrame, config: dict, run_id: str, run_idx: int) -> Tuple[Dict, List]:
        """
        Executes Data -> Split -> HPO -> Train -> Eval for a single seed.
        Returns (metrics_dict, selected_features_list).
        """
        # A. Split
        # We assume DataManager validation is deterministic/already done on raw_df
        # We just need to re-split.
        split_engine = SplitEngine(config, self.logger)
        # Note: SplitEngine writes files to disk. In stability runs, this might overwrite 
        # main run files if we don't change paths. Ideally, we should redirect output,
        # but for simplicity we rely on the in-memory dataframes here.
        train_df, val_df, test_df = split_engine.execute(raw_df, run_id)
        
        # B. HPO / Model Selection
        # To save time in stability runs, we might want to disable HPO and use the 
        # best config from the main run, OR re-run HPO to test stability of HPO itself.
        # Here we assume full re-run to test full pipeline stability.
        
        if config['hyperparameters']['enabled']:
            hpo_engine = HPOSearchEngine(config, self.logger)
            # This writes to 04_HYPERPARAMETER_SEARCH. 
            # Risk: Overwriting main run artifacts. 
            # Mitigation: In a robust sys, we'd change base_results_dir for stability runs.
            # For now, we proceed knowing it updates "latest".
            best_config = hpo_engine.execute(train_df, run_id)
        else:
            model_name = config['models']['native'][0]
            best_config = {'model': model_name, 'params': {}}
            
        # C. Training
        training_engine = TrainingEngine(config, self.logger)
        model = training_engine.train(train_df, best_config, run_id)
        
        # D. Evaluation
        pred_engine = PredictionEngine(config, self.logger)
        preds_test = pred_engine.predict(model, test_df, "stability_test", run_id)
        
        eval_engine = EvaluationEngine(config, self.logger)
        # We don't need to save Excel artifacts for every stability run, just get dict
        # But EvalEngine saves by default.
        metrics = eval_engine.compute_metrics(preds_test)
        
        # Prefix keys
        flat_metrics = {
            'test_cmae': metrics['cmae'],
            'test_crmse': metrics['crmse'],
            'test_accuracy_5': metrics['accuracy_at_5deg']
        }
        
        # Get features used
        # In current phase (no Iterative FS), all columns in X are features.
        drop_cols = config['data']['drop_columns'] + [
            config['data']['target_sin'], config['data']['target_cos'],
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
            return 1.0 # Only 1 run, stability is defined as 1
            
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
                    
        return np.mean(scores)