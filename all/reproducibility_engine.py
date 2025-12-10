import logging
import os
import sys
import shutil
import subprocess
import json
import platform
import time # Added for FIX #36
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class ReproducibilityEngine:
    """
    Creates a self-contained reproducibility package containing all essential
    artifacts, configuration, environment details, and documentation.
    Acts as a 'Time Capsule' for the experiment.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.enabled = self.config.get('outputs', {}).get('archive_reproducibility_package', True)
        
    def package(self, run_id: str) -> str:
        """
        Execute the packaging workflow.
        
        Parameters:
            run_id: Run identifier (used for metadata/documentation).
            
        Returns:
            Path to the created package directory.
        """
        if not self.enabled:
            self.logger.info("Reproducibility packaging disabled in config.")
            return ""

        self.logger.info("Creating Reproducibility Package...")
        
        # 1. Determine Paths
        base_dir = self.config.get('outputs', {}).get('base_results_dir', 'results')
        source_root = Path(base_dir)
        
        # The package goes inside the results folder
        package_dir = source_root / "13_REPRODUCIBILITY_PACKAGE"
        
        # 2. Prepare Directory
        try:
            if package_dir.exists():
                # FIX #36: Add retry logic for rmtree on Windows to handle file locks.
                if platform.system() == "Windows":
                    for _ in range(5): # Retry a few times
                        try:
                            shutil.rmtree(package_dir)
                            break
                        except OSError as e:
                            self.logger.warning(f"Failed to remove directory {package_dir} due to OS error: {e}. Retrying in 0.5s...")
                            time.sleep(0.5) # Wait a bit before retrying
                    else: # If loop completes without break
                        raise OSError(f"Failed to remove directory {package_dir} after multiple retries.")
                else: # Non-Windows or if retry successful
                    shutil.rmtree(package_dir)
            package_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create package directory: {e}")
            return ""
        
        # 3. Copy Essential Artifacts
        self._copy_artifacts(source_root, package_dir)
        
        # 4. Capture Environment (Pip freeze, OS info)
        self._capture_environment(package_dir)
        
        # 5. Generate Documentation (README)
        self._generate_readme(package_dir, run_id)
        
        self.logger.info(f"Reproducibility Package successfully created at: {package_dir.absolute()}")
        return str(package_dir)

    def _copy_artifacts(self, source_root: Path, dest_root: Path):
        """
        Copy key files from the scattered results folders into the package.
        Flattens the structure for easier access.
        """
        
        artifacts_map = {
            "00_CONFIG/config_used.json": "config.json",
            "00_CONFIG/run_metadata.json": "run_metadata.json",
            "05_FINAL_MODEL/final_model.pkl": "final_model.pkl",
            "05_FINAL_MODEL/training_metadata.json": "training_metadata.json",
            "06_PREDICTIONS/predictions_test.xlsx": "predictions_test.xlsx",
            "06_PREDICTIONS/predictions_val.xlsx": "predictions_val.xlsx",
            "07_EVALUATION/metrics_test.xlsx": "metrics_test.xlsx",
            "07_EVALUATION/metrics_val.xlsx": "metrics_val.xlsx",
            "10_REPORT/final_report.pdf": "final_report.pdf"
        }
        
        for src_rel, dst_name in artifacts_map.items():
            src_path = source_root / src_rel
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dest_root / dst_name)
                except Exception as e:
                    self.logger.warning(f"Failed to copy {src_rel}: {e}")
            else:
                self.logger.debug(f"Optional artifact not found: {src_rel}")

        fs_path = source_root / "03_ITERATIVE_FS" / "FINAL_SUMMARY" / "optimal_feature_set.json"
        if fs_path.exists():
            shutil.copy2(fs_path, dest_root / "optimal_feature_set.json")

        diag_dest = dest_root / "diagnostics"
        diag_dest.mkdir(exist_ok=True)
        
        key_plots = [
            "scatter_plots/actual_vs_pred_test.png",
            "distribution_plots/error_hist_test.png",
            "per_hs_plots/error_vs_hs_scatter_test.png"
        ]
        
        for plot_rel in key_plots:
            src = source_root / "08_DIAGNOSTICS" / plot_rel
            if src.exists():
                shutil.copy2(src, diag_dest / src.name)

    def _capture_environment(self, dest_root: Path):
        sys_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": sys.version,
            "processor": platform.processor(),
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(dest_root / "system_info.json", 'w') as f:
                json.dump(sys_info, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Could not save system_info.json: {e}")
            
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'freeze'],
                capture_output=True,
                text=True,
                check=False,
                timeout=60 # FIX #86: Add a timeout to prevent hanging.
            )
            
            if result.returncode == 0:
                with open(dest_root / "requirements_frozen.txt", 'w') as f:
                    f.write(result.stdout)
            else:
                self.logger.warning(f"Pip freeze failed (code {result.returncode}). Stderr: {result.stderr}")
                with open(dest_root / "requirements_frozen.txt", 'w') as f:
                    f.write(f"# Pip freeze failed.\n# Error: {result.stderr}")
                    
        except Exception as e:
            self.logger.warning(f"Could not capture requirements: {e}")

    def _generate_readme(self, dest_root: Path, run_id: str):
        readme_content = f"""# AI-Driven Riser Prediction - Reproducibility Package

**Run ID:** {run_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Package Contents
This folder contains the complete artifacts required to reproduce, audit, or deploy the model from this run.

### 1. Configuration & Metadata
- `config.json`: The exact configuration parameters used for this run.
- `run_metadata.json`: Timestamp, user, and execution details.
- `system_info.json`: OS and Python version details.
- `requirements_frozen.txt`: Exact library versions (`pip install -r requirements_frozen.txt`).

### 2. Model Assets
- `final_model.pkl`: The trained model object. Load using `joblib`.
- `training_metadata.json`: Details on input shape, features used, and training time.
- `optimal_feature_set.json`: (If FS enabled) The list of features used.

### 3. Results & Evaluation
- `final_report.pdf`: The comprehensive human-readable report.
- `metrics_test.xlsx`: Final performance metrics on the Test set.
- `predictions_test.xlsx`: Row-by-row predictions, ground truth, and errors.
- `diagnostics/`: Folder containing key visual plots (Scatter, Histograms).

## How to Load the Model
```python
import joblib
import pandas as pd
import json
from pathlib import Path
from sklearn.base import BaseEstimator
from utils.exceptions import ModelTrainingError

def safe_load_model(path: Path) -> BaseEstimator:
    \"\"\"Safely load model with validation.\"\"\"
    try:
        model = joblib.load(path)
        if not isinstance(model, BaseEstimator):
            raise ValueError("Invalid model type")
        return model
    except Exception as e:
        raise ModelTrainingError(f"Failed to load model: {{e}}")

model = safe_load_model(Path('final_model.pkl'))
# X_new = ...
# preds = model.predict(X_new)
```
"""
        try:
            with open(dest_root / "README.md", 'w') as f:
                f.write(readme_content)
        except Exception as e:
            self.logger.warning(f"Could not write README.md: {e}")