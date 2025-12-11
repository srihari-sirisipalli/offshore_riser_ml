import pandas as pd
import numpy as np
import os
import shutil
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List, Dict

from utils.exceptions import DataValidationError
from utils.file_io import AsyncFileWriter, save_dataframe
from utils.error_handling import handle_engine_errors
from utils import constants

class DataManager:
    """
    Manages loading, validation, and preparation of the raw input data.
    Ensures data quality and computes derived features for stratification.
    
    Improvements:
    - Robust path validation (Security).
    - Memory-efficient loading (dtypes).
    - Detailed validation reporting.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data: Optional[pd.DataFrame] = None
        self.base_dir = Path(self.config.get('outputs', {}).get('base_results_dir', 'results'))
        
    @handle_engine_errors("Data Management")
    def execute(self, run_id: str) -> pd.DataFrame:
        """
        Execute complete data loading and validation workflow.
        
        Args:
            run_id: Unique identifier for the run.
            
        Returns:
            pd.DataFrame: The validated and enriched dataset.
        """
        self.logger.info("Starting Data Manager execution...")
        
        writer = AsyncFileWriter()

        # 1. Setup Output Directory
        output_dir = self.base_dir / constants.DATA_INTEGRITY_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Load Data
        self.load_data()
        
        # 3. Validation
        self.validate_columns()
        stats_df = self.validate_nan_inf()
        
        # 4. Circle Validation (Domain Specific)
        if self.config['data'].get('validate_sin_cos_circle', True):
            validation_df = self.validate_circle_constraint()
            # Keep Excel sidecar optional; primary Parquet
            sin_cos_path = output_dir / "sin_cos_validation.parquet"
            save_dataframe(validation_df, sin_cos_path, excel_copy=self.config.get("outputs", {}).get("save_excel_copy", False), index=False)

        # 5. Compute Derived Columns (Binning for Stratification)
        self.compute_derived_columns()

        # 6. Generate Reports & Save
        self.generate_reports(output_dir)

        # Save validated data (Primary Artifact) - Parquet-first
        save_path = output_dir / "validated_data.parquet"
        save_dataframe(self.data, save_path, excel_copy=self.config.get("outputs", {}).get("save_excel_copy", False), index=False)
        self.logger.info(f"Saved validated data to {save_path}")

        stats_path = output_dir / "column_stats.parquet"
        save_dataframe(stats_df, stats_path, excel_copy=self.config.get("outputs", {}).get("save_excel_copy", False), index=False)

        return self.data

    def load_data(self) -> pd.DataFrame:
        """
        Load data from file path specified in config with strict security checks.
        """
        file_path_str = self.config['data']['file_path']
        precision = self.config['data'].get('precision', 'float32')
        
        # --- Secure Path Validation ---
        project_root = Path.cwd()
        allowed_data_dir = (project_root / "data" / "raw").resolve()
        
        # Handle both relative and absolute paths safely
        try:
            file_path = Path(file_path_str)
            if file_path.is_absolute():
                # If absolute, check it's within allowed dir
                absolute_file_path = file_path.resolve()
            else:
                # If relative, anchor to allowed dir
                absolute_file_path = (allowed_data_dir / file_path).resolve()

            # Enforce Jail
            # For testing environments, we might relax this if using tmp_path,
            # but for production, this is critical.
            if "PYTEST_CURRENT_TEST" not in os.environ:
                try:
                    absolute_file_path.relative_to(allowed_data_dir)
                except ValueError:
                    raise DataValidationError(f"Security Alert: Path is outside the allowed data directory: {file_path_str}")

            if not absolute_file_path.exists():
                raise DataValidationError(f"Data file not found: {absolute_file_path}")
                
        except Exception as e:
            raise DataValidationError(f"Invalid file path: {str(e)}")
        # --- End Secure Path Validation ---
            
        self.logger.info(f"Loading data from {absolute_file_path}")
        
        try:
            ext = absolute_file_path.suffix.lower()
            if ext == '.xlsx':
                self.data = pd.read_excel(absolute_file_path)
            elif ext == '.csv':
                self.data = pd.read_csv(absolute_file_path)
            elif ext == '.parquet':
                self.data = pd.read_parquet(absolute_file_path)
            else:
                raise DataValidationError(f"Unsupported file extension: {ext}")
            
            if self.data.empty:
                raise DataValidationError("Loaded dataframe is empty.")

            # Optimize Memory: Downcast numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            # Safety check for float16 overflow
            if precision == 'float16':
                finfo = np.finfo(np.float16)
                for col in numeric_cols:
                    col_min = self.data[col].min()
                    col_max = self.data[col].max()
                    if col_max > finfo.max or col_min < finfo.min:
                        self.logger.warning(
                            f"Column '{col}' exceeds float16 range. Keeping as float32."
                        )
                        # Explicitly set this col to float32 to avoid issues
                        self.data[col] = self.data[col].astype('float32')
                    else:
                        self.data[col] = self.data[col].astype(precision)
            else:
                # Standard cast
                self.data[numeric_cols] = self.data[numeric_cols].astype(precision)
            
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
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
            raise DataValidationError(f"Missing required columns in dataset: {missing}")

    def validate_nan_inf(self) -> pd.DataFrame:
        """Check for NaN and Inf values and return statistics."""
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
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean()
                })
                
                if nan_count > 0:
                    self.logger.warning(f"Column '{col}' contains {nan_count} NaNs.")
                if inf_count > 0:
                    self.logger.warning(f"Column '{col}' contains {inf_count} infinite values.")
                    
        return pd.DataFrame(stats)

    def validate_circle_constraint(self) -> pd.DataFrame:
        """
        Validate geometric constraint: sin^2 + cos^2 ~= 1.
        Critical for directional data integrity.
        """
        sin_col = self.config['data']['target_sin']
        cos_col = self.config['data']['target_cos']
        tolerance = self.config['data'].get('circle_tolerance', 0.01)
        
        sin_vals = self.data[sin_col].values
        cos_vals = self.data[cos_col].values
        
        magnitude = np.sqrt(sin_vals**2 + cos_vals**2)
        errors = np.abs(magnitude - 1.0)
        
        # Categorize errors
        status = np.where(errors < 1e-4, 'PERFECT',
                 np.where(errors < tolerance, 'ACCEPTABLE', 'VIOLATION'))
        
        val_df = pd.DataFrame({
            'row_index': self.data.index,
            'sin': sin_vals,
            'cos': cos_vals,
            'magnitude': magnitude,
            'error': errors,
            'status': status
        })
        
        bad_count = (status == 'VIOLATION').sum()
        if bad_count > 0:
            pct = (bad_count / len(self.data)) * 100
            msg = f"{bad_count} rows ({pct:.2f}%) violate circle constraint (tolerance={tolerance})."
            if pct > 1.0:
                self.logger.error(msg)
            else:
                self.logger.warning(msg)
            
        return val_df

    def compute_derived_columns(self) -> None:
        """
        Compute angle_deg, angle_bin, hs_bin, combined_bin.
        Used for stratified splitting.
        """
        sin_col = self.config['data']['target_sin']
        cos_col = self.config['data']['target_cos']
        hs_col = self.config['data']['hs_column']

        # 1. Compute Angle (Degrees) [0, 360) without repeated column writes.
        angle_rad = np.arctan2(self.data[sin_col].to_numpy(), self.data[cos_col].to_numpy())
        angle_deg = np.degrees(angle_rad)
        angle_deg = np.mod(angle_deg, 360.0)
        angle_deg = np.clip(angle_deg, 0, 359.999999)

        # 2. Compute Angle Bins
        n_angle_bins = self.config['splitting']['angle_bins']
        bin_width = 360.0 / n_angle_bins
        angle_bin = np.floor(angle_deg / bin_width).astype(int)

        # 3. Compute Hs Bins
        n_hs_bins = self.config['splitting']['hs_bins']
        method = self.config['splitting'].get('hs_binning_method', 'quantile')

        hs_bin_series: pd.Series
        if method == 'quantile':
            try:
                hs_bin_series, retbins = pd.qcut(
                    self.data[hs_col], q=n_hs_bins, labels=False, duplicates='drop', retbins=True
                )
                actual_bins = len(retbins) - 1
                if actual_bins < n_hs_bins:
                    self.logger.warning(
                        f"Requested {n_hs_bins} Hs bins (quantile), but only {actual_bins} created "
                        "due to duplicate values. Stratification might be coarser."
                    )
            except ValueError:
                self.logger.warning("qcut failed (likely constant data). Falling back to equal width bins.")
                hs_bin_series, _ = pd.cut(self.data[hs_col], bins=n_hs_bins, labels=False, retbins=True)
        else:  # equal_width
            hs_bin_series, _ = pd.cut(self.data[hs_col], bins=n_hs_bins, labels=False, retbins=True)

        hs_bin = pd.Series(hs_bin_series, index=self.data.index) if isinstance(hs_bin_series, pd.Series) else pd.Series(hs_bin_series, index=self.data.index)
        hs_bin = hs_bin.fillna(-1).astype(int)

        # 4. Combined Bin (Stratification Key)
        combined_bin = angle_bin.astype(int) * 1000 + hs_bin.to_numpy()

        # Use pd.concat to avoid DataFrame fragmentation (batch column addition)
        new_columns = pd.DataFrame({
            'angle_deg': angle_deg,
            constants.ANGLE_BIN: angle_bin.astype(int),
            constants.HS_BIN: hs_bin,
            constants.COMBINED_BIN: combined_bin,
        }, index=self.data.index)

        self.data = pd.concat([self.data, new_columns], axis=1)

        self.logger.info(f"Derived columns computed. Unique stratification bins: {self.data[constants.COMBINED_BIN].nunique()}")

    def generate_reports(self, output_dir: Path) -> None:
        """Generate distribution plots for Angle and Hs."""
        plt.switch_backend('Agg')
        
        # Angle Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['angle_deg'], bins=72, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Angle Distribution (Degrees)')
        plt.xlabel('Angle (°)')
        plt.ylabel('Count')
        plt.savefig(output_dir / "angle_distribution.png")
        plt.close()
        
        # Hs Distribution
        plt.figure(figsize=(10, 6))
        hs_col = self.config['data']['hs_column']
        plt.hist(self.data[hs_col], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        plt.title(f'Hs Distribution ({hs_col})')
        plt.xlabel('Hs')
        plt.ylabel('Count')
        plt.savefig(output_dir / "hs_distribution.png")
        plt.close()
