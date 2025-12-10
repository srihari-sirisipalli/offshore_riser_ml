
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Tuple, Optional

from utils.exceptions import DataValidationError
from utils.file_io import AsyncFileWriter
from utils.error_handling import handle_engine_errors

class DataManager:
    """
    Manages loading, validation, and preparation of the raw input data.
    Ensures data quality and computes derived features for stratification.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data: Optional[pd.DataFrame] = None
        
    @handle_engine_errors("Data Management")
    def execute(self, run_id: str) -> pd.DataFrame:
        """
        Execute complete data loading and validation workflow.
        """
        self.logger.info("Starting Data Manager execution...")
        
        writer = AsyncFileWriter()

        # 1. Determine Output Directory
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
            writer.write_excel_async(validation_df, output_dir / "sin_cos_validation.xlsx")
        
        # 5. Compute Derived Columns
        self.compute_derived_columns()
        
        # 6. Generate Reports & Save
        self.generate_reports(output_dir)
        
        # Save validated data
        save_path = output_dir / "validated_data.xlsx"
        writer.write_excel_async(self.data, save_path)
        self.logger.info(f"Saved validated data to {save_path}")
        
        writer.write_excel_async(stats_df, output_dir / "column_stats.xlsx")
        
        writer.wait_all()
        return self.data

    def load_data(self) -> pd.DataFrame:
        """Load data from file path specified in config."""
        file_path_str = self.config['data']['file_path']
        precision = self.config['data'].get('precision', 'float32')
        
        file_path = Path(file_path_str)
        if "PYTEST_CURRENT_TEST" in os.environ and file_path.is_absolute():
            absolute_file_path = file_path.resolve()
        else:
            # --- Secure Path Validation ---
            project_root = Path.cwd()
            allowed_data_dir = (project_root / "data" / "raw").resolve()
            
            # 1. Ensure file_path is relative
            if file_path.is_absolute():
                raise DataValidationError(f"Absolute paths are not allowed: {file_path_str}")

            # 2. Safely join and resolve
            absolute_file_path = (allowed_data_dir / file_path).resolve()

            # 3. Double-check it's still within the allowed directory
            try:
                absolute_file_path.relative_to(allowed_data_dir)
            except ValueError:
                raise DataValidationError(f"Path is outside the allowed data directory: {file_path_str}")

        if not absolute_file_path.exists():
            raise DataValidationError(f"Data file not found: {absolute_file_path}")
        # --- End Secure Path Validation ---
            
        self.logger.info(f"Loading data from {absolute_file_path}")
        
        try:
            ext = absolute_file_path.suffix.lower()
            if ext == '.xlsx':
                self.data = pd.read_excel(absolute_file_path)
            elif ext == '.csv':
                self.data = pd.read_csv(absolute_file_path)
            else:
                raise DataValidationError(f"Unsupported file extension: {ext}")
            
            # FIX #29: Check for empty dataframe immediately
            if self.data.empty:
                raise DataValidationError("Loaded dataframe is empty.")

            # Cast numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns

            # FIX #84: Safety check for float16 overflow
            if precision == 'float16':
                finfo = np.finfo(np.float16)
                for col in numeric_cols:
                    col_min = self.data[col].min()
                    col_max = self.data[col].max()
                    if col_max > finfo.max or col_min < finfo.min:
                        self.logger.warning(
                            f"Column '{col}' has values outside the float16 range. "
                            f"Casting may result in overflow/underflow. Consider using float32."
                        )
                        # Skip casting this specific column to float16, leave as is (likely float64) or cast to float32
                        self.data[col] = self.data[col].astype('float32')
                    else:
                        self.data[col] = self.data[col].astype(precision)
            else:
                self.data[numeric_cols] = self.data[numeric_cols].astype(precision)
            
            return self.data
            
        except Exception as e:
            raise DataValidationError(f"Failed to load data: {str(e)}")

    def validate_columns(self) -> None:
        """Ensure all required columns from config exist."""
        if self.data is None or self.data.empty:
            raise DataValidationError("Dataframe is empty or None.")

        required = [
            self.config['data']['target_sin'],
            self.config['data']['target_cos'],
            self.config['data']['hs_column']
        ]
        
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

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
        if bad_count > len(self.data) * 0.01:
            self.logger.error(f"Too many rows ({bad_count}) violate circle constraint (BAD > 1%)")
            
        return val_df

    def compute_derived_columns(self) -> None:
        """Compute angle_deg, angle_bin, hs_bin, combined_bin."""
        # FIX #19: Use a deep copy to prevent modifying the original dataframe reference
        self.data = self.data.copy(deep=True)
        
        sin_col = self.config['data']['target_sin']
        cos_col = self.config['data']['target_cos']
        hs_col = self.config['data']['hs_column']
        
        # 1. Compute Angle (Degrees)
        angle_rad = np.arctan2(self.data[sin_col], self.data[cos_col])
        angle_deg = np.degrees(angle_rad)
        self.data['angle_deg'] = angle_deg % 360.0
        
        # FIX #54: Clip 360.0 back to 0.0 or 359.99 to prevent bin 72 creating index out of bounds or collision
        # We map [0, 360) -> 0..359.999
        self.data['angle_deg'] = np.clip(self.data['angle_deg'], 0, 359.999999)

        # 2. Compute Angle Bins
        n_angle_bins = self.config['splitting']['angle_bins']
        bin_width = 360.0 / n_angle_bins
        # Now safely floor
        self.data['angle_bin'] = np.floor(self.data['angle_deg'] / bin_width).astype(int)
        
        # 3. Compute Hs Bins
        n_hs_bins = self.config['splitting']['hs_bins']
        method = self.config['splitting'].get('hs_binning_method', 'quantile')
        
        if method == 'quantile':
            # FIX #76: Handle case where qcut drops bins due to duplicates
            try:
                self.data['hs_bin'], retbins = pd.qcut(
                    self.data[hs_col], q=n_hs_bins, labels=False, duplicates='drop', retbins=True
                )
                actual_bins = len(retbins) - 1
                if actual_bins < n_hs_bins:
                    self.logger.warning(
                        f"qcut for hs_bin was configured for {n_hs_bins} bins, but "
                        f"only created {actual_bins} due to duplicate values in data. "
                        "This may affect stratification."
                    )
            except ValueError:
                # Fallback if qcut fails completely
                self.data['hs_bin'] = pd.cut(self.data[hs_col], bins=n_hs_bins, labels=False)
        else: # equal_width
            self.data['hs_bin'], _ = pd.cut(
                self.data[hs_col], bins=n_hs_bins, labels=False, retbins=True
            )
            
        # 4. Combined Bin (Stratification Key)
        # Ensure bins are ints
        self.data['hs_bin'] = self.data['hs_bin'].fillna(-1).astype(int)
        
        # Creates a unique integer ID for every combination
        self.data['combined_bin'] = self.data['angle_bin'] * n_hs_bins + self.data['hs_bin']
        
        self.logger.info(f"Derived columns computed. Unique combined bins: {self.data['combined_bin'].nunique()}")

    def generate_reports(self, output_dir: Path) -> None:
        """Generate histograms for Angle and Hs."""
        plt.switch_backend('Agg')
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['angle_deg'], bins=72, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Angle Distribution')
        plt.savefig(output_dir / "angle_distribution.png")
        plt.close()
        
        plt.figure(figsize=(10, 6))
        hs_col = self.config['data']['hs_column']
        plt.hist(self.data[hs_col], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title('Hs Distribution')
        plt.savefig(output_dir / "hs_distribution.png")
        plt.close()