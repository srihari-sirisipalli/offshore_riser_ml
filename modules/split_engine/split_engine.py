"""
SplitEngine for the Offshore Riser ML Pipeline.

This module is responsible for splitting the validated and feature-enriched
dataset into training, validation, and test sets. It employs a smart
stratification strategy to ensure that the distribution of key data segments
is preserved across all splits, which is crucial for robust model evaluation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from modules.base.base_engine import BaseEngine
from utils.error_handling import handle_engine_errors
from utils.cache import ensure_cache_dir, fingerprint
from pandas.util import hash_pandas_object
from utils.file_io import save_dataframe
from utils import constants

class SplitEngine(BaseEngine):
    """
    Splits data into Train/Val/Test sets using a robust stratified sampling strategy.

    This engine first determines the best column for stratification by analyzing
    bin distributions. It falls back from a high-cardinality 'combined_bin' to
    broader bins like 'hs_bin' or 'angle_bin' if necessary. It also handles
    rare data points (from bins with few members) by either moving them to the
    training set or dropping them, based on the configuration. This ensures
    that the splits are balanced and representative of the overall dataset.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.cache_enabled = self.config.get('execution', {}).get('enable_cache', True)
        self.cache_dir = ensure_cache_dir(self.base_dir, "split_engine")
        
    def _get_engine_directory_name(self) -> str:
        return constants.MASTER_SPLITS_DIR

    @handle_engine_errors("Data Splitting")
    def execute(self, df: pd.DataFrame, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the splitting workflow.
        
        Returns:
            train, val, test DataFrames
        """
        self.logger.info("Starting Split Engine execution...")
        
        # 1. Determine Stratification Column
        # Your data has 3456 unique bins for 3888 rows (High Cardinality).
        # Direct combined_bin stratification will fail. We need to check feasibility.
        stratify_col = self._determine_stratification_strategy(df)

        signature = self._compute_signature(df)
        if self.cache_enabled and signature:
            cached = self._try_load_cached_split(signature)
            if cached:
                train, val, test = cached
                self.logger.info(f"Loaded cached split (signature={signature}).")
                self._save_splits(train, val, test)
                return train, val, test
        
        # 2. Perform Split
        train, val, test = self._perform_split(df, stratify_col)
        
        # 3. Verify Balance
        self._generate_balance_report(train, val, test, stratify_col, self.output_dir)
        self._generate_split_plots(train, val, test, self.output_dir)
        if self.standard_output_dir != self.output_dir:
            self._generate_balance_report(train, val, test, stratify_col, self.standard_output_dir)
            self._generate_split_plots(train, val, test, self.standard_output_dir)
        
        # 4. Save Splits
        self._save_splits(train, val, test)

        if self.cache_enabled and signature:
            self._store_cached_split(signature, train, val, test)
        
        self.logger.info(f"Splits saved: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return train, val, test

    def _compute_signature(self, df: pd.DataFrame) -> Optional[str]:
        """
        Build a lightweight hash based on stratification columns to reuse splits for identical data/seeds.
        """
        candidate_cols = [c for c in ['combined_bin', 'hs_bin', 'angle_bin'] if c in df.columns]
        if not candidate_cols:
            return None
        hashed = hash_pandas_object(df[candidate_cols], index=True).sum()
        key = f"{len(df)}_{hashed}"
        return fingerprint(key)

    def _try_load_cached_split(self, signature: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        train_p = self.cache_dir / f"{signature}_train.parquet"
        val_p = self.cache_dir / f"{signature}_val.parquet"
        test_p = self.cache_dir / f"{signature}_test.parquet"
        if train_p.exists() and val_p.exists() and test_p.exists():
            try:
                return pd.read_parquet(train_p), pd.read_parquet(val_p), pd.read_parquet(test_p)
            except Exception:
                return None
        return None

    def _store_cached_split(self, signature: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        train.to_parquet(self.cache_dir / f"{signature}_train.parquet", index=True)
        val.to_parquet(self.cache_dir / f"{signature}_val.parquet", index=True)
        test.to_parquet(self.cache_dir / f"{signature}_test.parquet", index=True)

    def _save_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        save_dataframe(train, self.output_dir / "train.parquet", excel_copy=excel_copy, index=True)
        save_dataframe(val, self.output_dir / "val.parquet", excel_copy=excel_copy, index=True)
        save_dataframe(test, self.output_dir / "test.parquet", excel_copy=excel_copy, index=True)
        if self.standard_output_dir != self.output_dir:
            save_dataframe(train, self.standard_output_dir / "train.parquet", excel_copy=excel_copy, index=True)
            save_dataframe(val, self.standard_output_dir / "val.parquet", excel_copy=excel_copy, index=True)
            save_dataframe(test, self.standard_output_dir / "test.parquet", excel_copy=excel_copy, index=True)

    def _determine_stratification_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check if combined_bin is viable for stratification.
        If too many singletons, fallback to hs_bin or angle_bin.
        """
        # FIX #52: To survive two splits (train/val/test), we need at least 3 samples per bin.
        min_samples = 3
        
        # Check Combined Bin
        counts = df['combined_bin'].value_counts()
        singletons = (counts < min_samples).sum()
        
        if not len(counts):
             self.logger.error("Stratification column 'combined_bin' has no data.")
             return None

        singleton_pct = singletons / len(counts) * 100
        
        self.logger.info(f"Bin Analysis (combined_bin): {singletons} bins have < {min_samples} samples ({singleton_pct:.1f}%)")
        
        if singleton_pct < 20: # Arbitrary threshold: if <20% bins are issues, try combined
            self.logger.info("Using 'combined_bin' for stratification.")
            return 'combined_bin'
            
        # Fallback 1: Hs Bin (Wave height is critical for risers)
        counts_hs = df['hs_bin'].value_counts()
        singletons_hs = (counts_hs < min_samples).sum()
        
        if singletons_hs == 0:
            self.logger.warning("Too many rare combined bins. Falling back to 'hs_bin' stratification.")
            return 'hs_bin'
        else:
            # FIX #20: Add logging for why fallback is skipped.
            self.logger.warning(f"Could not use 'hs_bin' for stratification, it has {singletons_hs} bins with < {min_samples} samples.")

        # Fallback 2: Angle Bin
        counts_ang = df['angle_bin'].value_counts()
        if (counts_ang < min_samples).sum() == 0:
            self.logger.warning("Falling back to 'angle_bin' stratification.")
            return 'angle_bin'
        else:
            self.logger.warning(f"Could not use 'angle_bin' for stratification, it also has rare bins.")

        # Fallback 3: No Stratification (Random)
        self.logger.warning("Cannot stratify safely. Using random splitting.")
        return None

    def _perform_split(self, df: pd.DataFrame, strat_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform 2-stage split: 
        1. Full -> TrainVal + Test
        2. TrainVal -> Train + Val
        """
        # FIX #67: Use the propagated seed for splitting from _internal_seeds.
        seed = self.config.get('_internal_seeds', {}).get('split', self.config['splitting']['seed'])
        test_size = self.config['splitting']['test_size']
        val_size = self.config['splitting']['val_size']
        
        # FIX #52: Use a more robust minimum sample count for stratification.
        # A class needs at least one sample for each set: train, val, test.
        min_samples_for_stratify = 3
        drop_incomplete_bins = self.config['splitting'].get('drop_incomplete_bins', False)

        # Pre-filter for valid stratification if strat_col exists
        if strat_col:
            counts = df[strat_col].value_counts()
            valid_bins = counts[counts >= min_samples_for_stratify].index
            
            # If we have rows that can't be stratified
            rare_mask = ~df[strat_col].isin(valid_bins)
            if rare_mask.sum() > 0:
                # FIX #83: Implement drop_incomplete_bins logic.
                if drop_incomplete_bins:
                    self.logger.warning(f"{rare_mask.sum()} samples belong to incomplete bins and will be DROPPED.")
                    strat_data = df[~rare_mask]
                    rare_data = pd.DataFrame() # Ensure rare_data is empty
                else:
                    self.logger.warning(f"{rare_mask.sum()} samples belong to bins with < {min_samples_for_stratify} members. They will be put into Train set automatically.")
                    # Separate rare data (forced to train) from stratifiable data
                    rare_data = df[rare_mask]
                    strat_data = df[~rare_mask]
            else:
                rare_data = pd.DataFrame()
                strat_data = df
        else:
            rare_data = pd.DataFrame()
            strat_data = df

        # --- SPLIT 1: Separate Test ---
        # Stratify if possible
        stratify_target = strat_data[strat_col] if strat_col and not strat_data.empty else None
        
        try:
            # Handle case where strat_data might be empty after filtering
            if strat_data.empty:
                self.logger.warning("No data left for stratification after filtering rare bins. Performing random split on original data.")
                train_val_main, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
                strat_col = None # Ensure no further stratification is attempted
            else:
                train_val_main, test = train_test_split(
                    strat_data,
                    test_size=test_size,
                    random_state=seed,
                    shuffle=True,
                    stratify=stratify_target
                )
        except ValueError as e:
            self.logger.error(f"Stratification failed: {e}. Falling back to random split.")
            train_val_main, test = train_test_split(
                strat_data,
                test_size=test_size,
                random_state=seed,
                shuffle=True
            )
            strat_col = None # Ensure no further stratification is attempted

        # --- SPLIT 2: Separate Validation ---
        # Adjust val_size to be relative to train_val
        val_ratio_adjusted = val_size / (1.0 - test_size)
        
        # Only stratify if we could do it in the first stage and data exists
        stratify_target_tv = train_val_main[strat_col] if strat_col and not train_val_main.empty else None
        
        try:
            if train_val_main.empty:
                 # This case can happen if test_size is very large
                 train_main, val = pd.DataFrame(), pd.DataFrame()
            else:
                train_main, val = train_test_split(
                    train_val_main,
                    test_size=val_ratio_adjusted,
                    random_state=seed,
                    shuffle=True,
                    stratify=stratify_target_tv
                )
        except ValueError:
             self.logger.warning("Secondary stratification failed. Using random split for Validation.")
             train_main, val = train_test_split(
                train_val_main,
                test_size=val_ratio_adjusted,
                random_state=seed,
                shuffle=True
            )

        # Add rare data back to Train
        if not rare_data.empty:
            train = pd.concat([train_main, rare_data])
        else:
            train = train_main
            
        return train, val, test

    def _generate_balance_report(self, train, val, test, col, output_dir):
        """Save a report showing distribution of the stratification key."""
        if col is None:
            col = 'hs_bin' # Default for reporting if random split
            
        all_bins = set(train[col].unique()) | set(val[col].unique()) | set(test[col].unique())
        
        report = []
        for b in all_bins:
            c_train = len(train[train[col] == b])
            c_val = len(val[val[col] == b])
            c_test = len(test[test[col] == b])
            total = c_train + c_val + c_test
            
            report.append({
                'bin': b,
                'total': total,
                'train%': round(c_train/total, 2),
                'val%': round(c_val/total, 2),
                'test%': round(c_test/total, 2)
            })
            
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)
        save_dataframe(pd.DataFrame(report), output_dir / "split_balance_report.parquet", excel_copy=excel_copy, index=False)

    def _generate_split_plots(self, train, val, test, output_dir):
        """Plot Hs and Angle distributions overlaid."""
        plt.switch_backend('Agg')
        # FIX: Use .get() to safely retrieve 'hs_column' from config with a default of 'Hs'
        hs_col = self.config.get('data', {}).get('hs_column', 'Hs')

        # Plot HS Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(train[hs_col], bins=30, alpha=0.5, label='Train', density=True)
        plt.hist(val[hs_col], bins=30, alpha=0.5, label='Val', density=True)
        plt.hist(test[hs_col], bins=30, alpha=0.5, label='Test', density=True)
        plt.legend()
        plt.title("Hs Distribution per Split (Normalized)")
        plt.savefig(output_dir / "split_hs_dist.png")
        plt.close()
