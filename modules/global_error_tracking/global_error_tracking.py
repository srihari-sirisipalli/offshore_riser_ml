import pandas as pd
import numpy as np
import logging
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

class GlobalErrorTrackingEngine:
    """
    Tracks sample-level error evolution.
    
    NEW CAPABILITY:
    - Compiles 'Crash-Proof' snapshots from HPO into a Master Failure Matrix.
    - Identifies 'Persistent Failures' (samples that fail >80% of the time).
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.track_config = config.get('global_tracking', {})
        
    def compile_tracking_data(self, val_df: pd.DataFrame, test_df: pd.DataFrame, run_id: str):
        """
        [PHASE 4 COMPILER]
        Reads all CSV snapshots from disk, merges them, calculates persistence, 
        and saves the Master Excel files for Val and Test sets.
        """
        self.logger.info("Starting Global Failure Tracking Compilation (Phase 4)...")
        
        base_dir = Path(self.config['outputs'].get('base_results_dir', 'results'))
        snapshot_dir = base_dir / "03_HPO_SEARCH" / "tracking_snapshots"
        output_dir = base_dir / "04_GLOBAL_FAILURE_TRACKING"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not snapshot_dir.exists():
            self.logger.warning("No HPO snapshots found. Skipping Global Tracking compilation.")
            return

        # Compile Validation Set
        self._process_set_compilation(val_df, snapshot_dir, output_dir, "val")
        
        # Compile Test Set
        self._process_set_compilation(test_df, snapshot_dir, output_dir, "test")
        
        self.logger.info(f"Global Tracking compilation complete. Results in: {output_dir}")

    def _process_set_compilation(self, base_df: pd.DataFrame, snapshot_dir: Path, output_dir: Path, set_name: str):
        """
        Internal method to aggregate snapshots for a specific dataset (val/test).
        """
        # 1. Setup Master DataFrame with Context Columns
        hs_col = self.config['data'].get('hs_column', 'sea_elevation_significant_height_Hs_m')
        
        # Safe retrieval of Hs (The Physics Context)
        if hs_col in base_df.columns:
            master_df = base_df[[hs_col]].copy()
            master_df.rename(columns={hs_col: 'Hs'}, inplace=True)
        else:
            # Fallback if Hs column name doesn't match
            master_df = pd.DataFrame(index=base_df.index)
            master_df['Hs'] = np.nan

        # Add Row Index and True Angle
        master_df['row_index'] = base_df.index
        
        # Try to get the Ground Truth Angle for context
        if 'angle_deg' in base_df.columns:
            master_df['True_Angle'] = base_df['angle_deg']
        else:
            # Reconstruct from sin/cos if 'angle_deg' is missing
            sin_col = self.config['data'].get('target_sin')
            cos_col = self.config['data'].get('target_cos')
            if sin_col in base_df.columns and cos_col in base_df.columns:
                from utils.circular_metrics import reconstruct_angle
                master_df['True_Angle'] = reconstruct_angle(base_df[sin_col].values, base_df[cos_col].values)
            else:
                master_df['True_Angle'] = 0.0

        # FIX #62: Set master_df index to 'row_index' for safe merging
        master_df = master_df.set_index('row_index')

        # 2. Find and Merge Snapshots
        pattern = str(snapshot_dir / f"*_{set_name}.csv")
        csv_files = sorted(glob.glob(pattern))
        
        if not csv_files:
            self.logger.warning(f"No snapshots found for {set_name} set.")
            return

        self.logger.info(f"Compiling {len(csv_files)} snapshots for {set_name} set...")

        error_cols = []
        num_files = len(csv_files)
        # FIX #92: Add progress indicator for compiling snapshots.
        log_interval = max(1, num_files // 10) # Log roughly 10 times.

        for i, f in enumerate(csv_files):
            if (i + 1) % log_interval == 0 or i == num_files - 1:
                self.logger.info(f"  ... processing snapshot {i+1}/{num_files} ({(i+1)/num_files:.0%})")
                
            try:
                # Filename format: trial_001_val.csv
                trial_id = Path(f).stem.replace(f"_{set_name}", "") 
                
                # Load snapshot
                # FIX #26: Specify encoding for pd.read_csv
                snap = pd.read_csv(f, encoding='utf-8')
                
                # We only need the error column, renamed to the Trial ID
                if 'row_index' in snap.columns and 'abs_error' in snap.columns:
                    snap_indexed = snap.set_index('row_index')
                    col_name = f"{trial_id}_Error"
                    
                    # FIX #62: Use reindex to safely merge and detect missing rows
                    merged_series = snap_indexed['abs_error'].reindex(master_df.index)
                    
                    nan_count = merged_series.isna().sum()
                    if nan_count > 0:
                        self.logger.warning(f"Snapshot {f} is missing {nan_count} rows from the base dataset for {set_name}. NaNs introduced in '{col_name}'.")

                    master_df[col_name] = merged_series
                    error_cols.append(col_name)
            except Exception as e:
                self.logger.warning(f"Skipping corrupt snapshot {f}: {e}")

        # 3. Calculate Persistence Logic (The "Smart" Columns)
        if error_cols:
            threshold = 10.0 # Error threshold in degrees (adjustable)
            
            # Count how many configs failed this specific row
            failures = (master_df[error_cols] > threshold).sum(axis=1)
            total_trials = len(error_cols)
            
            master_df['Failure_Rate_%'] = (failures / total_trials) * 100
            
            # The Verdict: Is this a "Persistent Failure"? (>80%)
            master_df['Is_Persistent_Failure'] = master_df['Failure_Rate_%'] > 80.0
            
            # Add Min/Mean stats for deeper analysis
            master_df['Min_Error'] = master_df[error_cols].min(axis=1)
            master_df['Mean_Error'] = master_df[error_cols].mean(axis=1)
        else:
            master_df['Failure_Rate_%'] = 0.0
            master_df['Is_Persistent_Failure'] = False

        # 4. Final Column Ordering
        # Order: Index, Hs, Angle, FLAGS, METRICS, RAW DATA
        # Reset index to make 'row_index' a column again for final Excel output
        master_df = master_df.reset_index()

        metadata_cols = ['row_index', 'Hs', 'True_Angle']
        flag_cols = ['Is_Persistent_Failure', 'Failure_Rate_%', 'Min_Error', 'Mean_Error']
        
        # Ensure columns exist before selecting
        final_cols = [c for c in (metadata_cols + flag_cols + error_cols) if c in master_df.columns]
        
        final_df = master_df[final_cols]

        # 5. Save to Excel
        save_path = output_dir / f"{set_name.capitalize()}_Set_Tracking.xlsx"
        try:
            final_df.to_excel(save_path, index=False)
            self.logger.info(f"Saved {set_name} Master Matrix: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save Excel matrix for {set_name}: {e}")

    # FIX #15, #6: Remove legacy/dead code methods that are not called and do nothing.
    # def track(self, round_predictions: List[pd.DataFrame], feature_history: List[Dict], run_id: str) -> Dict[str, Any]:
    #     """Legacy tracking method (optional use)."""
    #     if not self.track_config.get('enabled', True):
    #         return {}
    #     return {}

    # def _build_evolution_matrix(self, predictions_list: List[pd.DataFrame]) -> pd.DataFrame:
    #     # This method was also unused and served no purpose.
    #     pass