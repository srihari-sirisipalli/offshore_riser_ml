import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

class SplitEngine:
    """
    Splits data into Train/Val/Test sets using Stratified Sampling.
    Handles rare bins by falling back to broader stratification keys if necessary.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def execute(self, df: pd.DataFrame, run_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the splitting workflow.
        
        Returns:
            train, val, test DataFrames
        """
        self.logger.info("Starting Split Engine execution...")
        
        # 1. Output Directory
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "02_SMART_SPLIT"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Determine Stratification Column
        # Your data has 3456 unique bins for 3888 rows (High Cardinality).
        # Direct combined_bin stratification will fail. We need to check feasibility.
        stratify_col = self._determine_stratification_strategy(df)
        
        # 3. Perform Split
        train, val, test = self._perform_split(df, stratify_col)
        
        # 4. Verify Balance
        self._generate_balance_report(train, val, test, stratify_col, output_dir)
        self._generate_split_plots(train, val, test, output_dir)
        
        # 5. Save Splits
        train.to_excel(output_dir / "train.xlsx", index=False)
        val.to_excel(output_dir / "val.xlsx", index=False)
        test.to_excel(output_dir / "test.xlsx", index=False)
        
        self.logger.info(f"Splits saved: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        return train, val, test

    def _determine_stratification_strategy(self, df: pd.DataFrame) -> Optional[str]:
        """
        Check if combined_bin is viable for stratification.
        If too many singletons, fallback to hs_bin or angle_bin.
        """
        min_samples = 2 # Need at least 2 samples to split (1 train, 1 test)
        
        # Check Combined Bin
        counts = df['combined_bin'].value_counts()
        singletons = (counts < min_samples).sum()
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
            
        # Fallback 2: Angle Bin
        counts_ang = df['angle_bin'].value_counts()
        if (counts_ang < min_samples).sum() == 0:
            self.logger.warning("Falling back to 'angle_bin' stratification.")
            return 'angle_bin'
            
        # Fallback 3: No Stratification (Random)
        self.logger.warning("Cannot stratify safely. Using random splitting.")
        return None

    def _perform_split(self, df: pd.DataFrame, strat_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform 2-stage split: 
        1. Full -> TrainVal + Test
        2. TrainVal -> Train + Val
        """
        seed = self.config['splitting']['seed']
        test_size = self.config['splitting']['test_size']
        val_size = self.config['splitting']['val_size']
        
        # If stratification column has any classes with only 1 member, 
        # train_test_split will error. We filter them out for the split logic 
        # or handle them. Here we use the 'try/except' approach or pre-filtering.
        
        # Pre-filter for valid stratification if strat_col exists
        if strat_col:
            # Check for bins with only 1 sample
            counts = df[strat_col].value_counts()
            valid_bins = counts[counts >= 2].index
            
            # If we have rows that can't be stratified
            rare_mask = ~df[strat_col].isin(valid_bins)
            if rare_mask.sum() > 0:
                self.logger.warning(f"{rare_mask.sum()} samples belong to singleton bins. They will be put into Train set automatically.")
                
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
        stratify_target = strat_data[strat_col] if strat_col else None
        
        try:
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

        # --- SPLIT 2: Separate Validation ---
        # Adjust val_size to be relative to train_val
        # If Total=100, Test=10, TrainVal=90. We want Val=10.
        # So val_ratio = 10 / 90 = 0.111...
        val_ratio_adjusted = val_size / (1.0 - test_size)
        
        stratify_target_tv = train_val_main[strat_col] if strat_col else None
        
        try:
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
            
        pd.DataFrame(report).to_excel(output_dir / "split_balance_report.xlsx", index=False)

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