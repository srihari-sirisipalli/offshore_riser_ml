import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Tuple, Optional

from utils.exceptions import DataValidationError

class DataManager:
    """
    Load, validate, and prepare raw input data.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data: Optional[pd.DataFrame] = None
        
    def execute(self, run_id: str) -> pd.DataFrame:
        """
        Execute complete data loading and validation workflow.
        """
        self.logger.info("Starting Data Manager execution...")
        
        # 1. Determine Output Directory (FIXED)
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        output_dir = Path(base_dir) / "01_DATA_VALIDATION"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Load Data
        self.load_data()
        
        # 3. Validation
        self.validate_columns()
        stats_df = self.validate_nan_inf()
        
        # 4. Circle Validation
        if self.config['data'].get('validate_sin_cos_circle', True):
            validation_df = self.validate_circle_constraint()
            validation_df.to_excel(output_dir / "sin_cos_validation.xlsx", index=False)
        
        # 5. Compute Derived Columns
        self.compute_derived_columns()
        
        # 6. Generate Reports & Save
        self.generate_reports(output_dir)
        
        # Save validated data
        save_path = output_dir / "validated_data.xlsx"
        self.data.to_excel(save_path, index=False)
        self.logger.info(f"Saved validated data to {save_path}")
        
        # Save stats
        stats_df.to_excel(output_dir / "column_stats.xlsx", index=False)
        
        return self.data

    def load_data(self) -> pd.DataFrame:
        """Load data from file path specified in config."""
        file_path = self.config['data']['file_path']
        precision = self.config['data'].get('precision', 'float32')
        
        if not os.path.exists(file_path):
            raise DataValidationError(f"Data file not found: {file_path}")
            
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            ext = Path(file_path).suffix.lower()
            if ext == '.xlsx':
                self.data = pd.read_excel(file_path)
            elif ext == '.csv':
                self.data = pd.read_csv(file_path)
            else:
                raise DataValidationError(f"Unsupported file extension: {ext}")
                
            # Cast numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].astype(precision)
            
            self.logger.info(f"Loaded {len(self.data)} rows, {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            raise DataValidationError(f"Failed to load data: {str(e)}")

    def validate_columns(self) -> None:
        """Ensure all required columns from config exist."""
        required = [
            self.config['data']['target_sin'],
            self.config['data']['target_cos'],
            self.config['data']['hs_column']
        ]
        
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
            
        self.logger.info("Required columns validated successfully.")

    def validate_nan_inf(self) -> pd.DataFrame:
        """Check for NaN and Inf values."""
        stats = []
        for col in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[col]):
                nan_count = self.data[col].isna().sum()
                inf_count = np.isinf(self.data[col]).sum()
                
                stats.append({
                    'column': col,
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                })
                
                if nan_count > 0:
                    self.logger.warning(f"Column '{col}' has {nan_count} NaNs")
                if inf_count > 0:
                    self.logger.warning(f"Column '{col}' has {inf_count} Infs")
                    
        return pd.DataFrame(stats)

    def validate_circle_constraint(self) -> pd.DataFrame:
        """Validate sin^2 + cos^2 ~= 1."""
        sin_col = self.config['data']['target_sin']
        cos_col = self.config['data']['target_cos']
        tolerance = self.config['data'].get('circle_tolerance', 0.01)
        
        sin_vals = self.data[sin_col].values
        cos_vals = self.data[cos_col].values
        
        magnitude = np.sqrt(sin_vals**2 + cos_vals**2)
        errors = np.abs(magnitude - 1.0)
        
        status = np.where(errors < 0.001, 'GOOD',
                 np.where(errors < tolerance, 'WARNING', 'BAD'))
        
        val_df = pd.DataFrame({
            'row_index': self.data.index,
            'sin': sin_vals,
            'cos': cos_vals,
            'error': errors,
            'status': status
        })
        
        bad_count = (status == 'BAD').sum()
        total_count = len(self.data)
        
        self.logger.info(f"Circle Check: GOOD={(status=='GOOD').sum()}, WARNING={(status=='WARNING').sum()}, BAD={bad_count}")
        
        if bad_count > total_count * 0.01:
            msg = f"Too many rows ({bad_count}) violate circle constraint (BAD > 1%)"
            self.logger.error(msg)
            
        return val_df

    def compute_derived_columns(self) -> None:
        """Compute angle_deg, angle_bin, hs_bin, combined_bin."""
        self.data = self.data.copy()
        sin_col = self.config['data']['target_sin']
        cos_col = self.config['data']['target_cos']
        hs_col = self.config['data']['hs_column']
        
        # 1. Compute Angle (Degrees)
        angle_rad = np.arctan2(self.data[sin_col], self.data[cos_col])
        angle_deg = np.degrees(angle_rad)
        self.data['angle_deg'] = angle_deg % 360.0
        
        # 2. Compute Angle Bins
        n_angle_bins = self.config['splitting']['angle_bins']
        bin_width = 360.0 / n_angle_bins
        self.data['angle_bin'] = np.floor(self.data['angle_deg'] / bin_width).astype(int) % n_angle_bins
        
        # 3. Compute Hs Bins
        n_hs_bins = self.config['splitting']['hs_bins']
        method = self.config['splitting'].get('hs_binning_method', 'quantile')
        
        if method == 'quantile':
            self.data['hs_bin'], _ = pd.qcut(
                self.data[hs_col], q=n_hs_bins, labels=False, duplicates='drop', retbins=True
            )
        else:
            self.data['hs_bin'], _ = pd.cut(
                self.data[hs_col], bins=n_hs_bins, labels=False, retbins=True
            )
            
        # 4. Combined Bin
        self.data['combined_bin'] = self.data['angle_bin'] * n_hs_bins + self.data['hs_bin']
        
        self.logger.info(f"Derived columns computed. Unique combined bins: {self.data['combined_bin'].nunique()}")

    def generate_reports(self, output_dir: Path) -> None:
        """Generate histograms for Angle and Hs."""
        plt.switch_backend('Agg')
        
        # Angle Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['angle_deg'], bins=72, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Angle Distribution')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "angle_distribution.png")
        plt.close()
        
        # Hs Distribution
        plt.figure(figsize=(10, 6))
        hs_col = self.config['data']['hs_column']
        plt.hist(self.data[hs_col], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title('Significant Wave Height (Hs) Distribution')
        plt.xlabel('Hs (m)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "hs_distribution.png")
        plt.close()
        
        self.logger.info("Distribution plots generated.")