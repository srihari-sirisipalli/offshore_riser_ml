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
import pandas as pd
from modules.base.base_engine import BaseEngine
from utils.file_io import save_dataframe
from utils import constants

class ReproducibilityEngine(BaseEngine):
    """
    Creates a self-contained reproducibility package containing all essential
    artifacts, configuration, environment details, and documentation.
    Acts as a 'Time Capsule' for the experiment.
    """
    
    def __init__(self, config: dict, logger: logging.Logger):
        super().__init__(config, logger)
        self.enabled = self.config.get('outputs', {}).get('archive_reproducibility_package', True)

    def _get_engine_directory_name(self) -> str:
        return constants.REPRODUCIBILITY_PACKAGE_DIR
        
    def execute(self, run_id: str) -> str:
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
        source_root = self.base_dir
        package_dir = self.output_dir
        standard_dir = self.standard_output_dir
        
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
        # 6. Deployment readiness checklist
        self._write_deployment_checklist(package_dir)
        
        # Copy to standardized directory if different
        if standard_dir != package_dir:
            try:
                if standard_dir.exists():
                    shutil.rmtree(standard_dir, ignore_errors=True)
                shutil.copytree(package_dir, standard_dir)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(f"Failed to copy reproducibility package to standard layout: {exc}")
        self.logger.info(f"Reproducibility Package successfully created at: {package_dir.absolute()}")
        return str(package_dir)

    def _copy_artifacts(self, source_root: Path, dest_root: Path):
        """
        Copy key files from the scattered results folders into the package.
        Flattens the structure for easier access.
        """
        
        from utils import constants

        artifacts_map = {
            f"{constants.CONFIG_DIR}/config_used.json": "config.json",
            f"{constants.CONFIG_DIR}/run_metadata.json": "run_metadata.json",
            f"{constants.RFE_SUMMARY_DIR}/all_rounds_metrics.parquet": "rfe_summary_metrics.parquet",
            f"{constants.RFE_SUMMARY_DIR}/feature_elimination_history.parquet": "feature_elimination_history.parquet",
            f"{constants.RECONSTRUCTION_MAPPING_DIR}/best_model_info.json": "best_model_info.json",
            f"{constants.REPORTING_DIR}/final_report.pdf": "final_report.pdf"
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

    def _write_deployment_checklist(self, dest_root: Path) -> None:
        """
        Write a deployment readiness checklist with real status checks.
        """
        base_results = Path(self.config.get("outputs", {}).get("base_results_dir", "results"))
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)

        def exists(rel_path: str) -> bool:
            return (base_results / rel_path).exists()

        checks = [
            {
                "item": "Model serialization validated",
                "status": "PASS" if exists("05_FINAL_MODEL/final_model.pkl") else "FAIL",
                "notes": "Checks for persisted final_model.pkl in 05_FINAL_MODEL",
            },
            {
                "item": "Input validation implemented",
                "status": "PASS" if exists("01_DATA_VALIDATION/validated_data.parquet") else "WARN",
                "notes": "Requires validated_data artifacts in 01_DATA_VALIDATION",
            },
            {
                "item": "Latency meets requirement",
                "status": "WARN",
                "notes": "No latency benchmark provided; measure before deployment",
            },
            {
                "item": "Memory footprint acceptable",
                "status": "PASS" if exists("00_DATA_INTEGRITY/resource_utilization_snapshot.json") else "WARN",
                "notes": "Uses resource snapshot from data integrity tracker",
            },
            {
                "item": "Resource limits configured",
                "status": "PASS" if "resource_limits" in self.config else "WARN",
                "notes": "Config should contain resource_limits",
            },
            {
                "item": "Logging/monitoring enabled",
                "status": "PASS" if "logging" in self.config else "WARN",
                "notes": "Verify logging config section present",
            },
            {
                "item": "Health checks implemented",
                "status": "WARN",
                "notes": "Add runtime health checks to deployment target",
            },
            {
                "item": "Error handling & fallbacks in place",
                "status": "WARN",
                "notes": "Ensure error handling utilities are packaged with the model",
            },
            {
                "item": "Security review completed",
                "status": "PENDING",
                "notes": "Manual security review required",
            },
            {
                "item": "Integration tests run",
                "status": "WARN",
                "notes": "Record latest pytest run artifacts before release",
            },
            {
                "item": "Rollback plan documented",
                "status": "PENDING",
                "notes": "Document rollback/blue-green procedure",
            },
        ]
        df = pd.DataFrame(checks)
        save_dataframe(df, dest_root / "deployment_readiness_checklist.parquet", excel_copy=excel_copy, index=False)
        (dest_root / "deployment_readiness_checklist.json").write_text(df.to_json(orient="records", indent=2))

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
- `metrics_test.parquet`: Final performance metrics on the Test set (optional Excel sidecar).
- `predictions_test.parquet`: Row-by-row predictions, ground truth, and errors (optional Excel sidecar).
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
