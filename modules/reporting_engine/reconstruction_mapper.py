import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from utils.file_io import save_dataframe, read_dataframe


class ReconstructionMapper:
    """
    Generates reconstruction mapping artifacts (Parquet-first, optional Excel copies).
    Provides an audit trail to recreate any model from RFE history.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.base_dir = Path(self.config.get("outputs", {}).get("base_results_dir", "results"))

    def generate_mapping(
        self,
        rounds_history: List[Dict[str, Any]],
        data_files_info: Dict[str, str],
        output_dir: Path,
    ):
        """
        Build and save reconstruction mapping artifacts.

        Args:
            rounds_history: List of round summary dictionaries (from RFEController).
            data_files_info: Paths/hashes of initial train/val/test files.
            output_dir: Directory to save artifacts.
        """
        self.logger.info("Generating Model Reconstruction Mapping...")
        index_path = output_dir / "model_reconstruction_mapping.parquet"
        excel_copy = self.config.get("outputs", {}).get("save_excel_copy", False)

        # Prepare DataFrames for each sheet-equivalent
        df_summary = self._build_summary_sheet(rounds_history)
        df_params = self._build_params_sheet(rounds_history)
        df_features = self._build_features_sheet(rounds_history)
        df_files = self._build_files_sheet(rounds_history, data_files_info)
        df_code = self._build_code_sheet(rounds_history, data_files_info)

        try:
            save_dataframe(df_summary, output_dir / "model_reconstruction_summary.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(df_params, output_dir / "model_reconstruction_hyperparameters.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(df_features, output_dir / "model_reconstruction_features.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(df_files, output_dir / "model_reconstruction_data_files.parquet", excel_copy=excel_copy, index=False)
            save_dataframe(df_code, output_dir / "model_reconstruction_code.parquet", excel_copy=excel_copy, index=False)

            artifact_index = pd.DataFrame(
                {
                    "artifact": [
                        "Rounds_Summary",
                        "Hyperparameters",
                        "Features_Per_Round",
                        "Data_Files",
                        "Recreation_Code",
                    ],
                    "path": [
                        "model_reconstruction_summary.parquet",
                        "model_reconstruction_hyperparameters.parquet",
                        "model_reconstruction_features.parquet",
                        "model_reconstruction_data_files.parquet",
                        "model_reconstruction_code.parquet",
                    ],
                }
            )
            save_dataframe(artifact_index, index_path, excel_copy=excel_copy, index=False)

            self.logger.info(f"Reconstruction mapping saved to: {output_dir}")

            # Copy to standardized location (97_RECONSTRUCTION_MAPPING)
            std_dir = self.base_dir / "97_RECONSTRUCTION_MAPPING"
            std_dir.mkdir(parents=True, exist_ok=True)
            for fname in [
                "model_reconstruction_mapping.parquet",
                "model_reconstruction_summary.parquet",
                "model_reconstruction_hyperparameters.parquet",
                "model_reconstruction_features.parquet",
                "model_reconstruction_data_files.parquet",
                "model_reconstruction_code.parquet",
            ]:
                src = output_dir / fname
                df = read_dataframe(src)
                save_dataframe(df, std_dir / fname, excel_copy=excel_copy, index=False)
        except Exception as e:
            self.logger.error(f"Failed to write reconstruction mapping: {e}")
            raise

    def _build_summary_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            metrics = r.get("metrics", {})
            data.append(
                {
                    "Round": r["round"],
                    "N_Features": r["n_features"],
                    "Dropped_Feature": r.get("dropped_feature", "None"),
                    "Val_CMAE": metrics.get("cmae", metrics.get("val_cmae", 0)),
                    "Val_Accuracy_5deg": metrics.get("accuracy_at_5deg", 0),
                    "Stop_Reason": r.get("stopping_reason", ""),
                }
            )
        return pd.DataFrame(data)

    def _build_params_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            row = {"Round": r["round"]}
            params = r.get("hyperparameters", {})
            for k, v in params.items():
                row[k] = v
            data.append(row)
        return pd.DataFrame(data)

    def _build_features_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            feats = r.get("active_features_list", [])
            feat_str = ", ".join(feats) if isinstance(feats, list) else "Check feature_list.json in round folder"

            data.append(
                {
                    "Round": r["round"],
                    "N_Features": r["n_features"],
                    "Active_Features": feat_str,
                }
            )
        return pd.DataFrame(data)

    def _build_files_sheet(self, history: List[Dict], data_info: Dict) -> pd.DataFrame:
        data = []
        for r in history:
            data.append(
                {
                    "Round": r["round"],
                    "Train_File": data_info.get("train_path", "train.parquet"),
                    "Val_File": data_info.get("val_path", "val.parquet"),
                    "Input_Hash": data_info.get("input_hash", "N/A"),
                }
            )
        return pd.DataFrame(data)

    def _build_code_sheet(self, history: List[Dict], data_info: Dict) -> pd.DataFrame:
        data = []
        for r in history:
            code = self._generate_python_snippet(r, data_info)
            data.append({"Round": r["round"], "Recreation_Code": code})
        return pd.DataFrame(data)

    def _generate_python_snippet(self, round_data: Dict, data_info: Dict) -> str:
        """Construct a string of valid Python code to train the model."""
        rnd = round_data["round"]
        params = round_data.get("hyperparameters", {})
        model_name = params.get("model_name", "ExtraTreesRegressor")

        train_params = {k: v for k, v in params.items() if k not in ["round_tuned", "model_name"]}
        param_str = ",\n    ".join([f"{k}={repr(v)}" for k, v in train_params.items()])

        code = f"""
# ============================================
# RECREATION CODE FOR ROUND {rnd}
# ============================================
import pandas as pd
import joblib
from sklearn.ensemble import {model_name}

# 1. Load Data
train_path = r"{data_info.get('train_path', 'train.parquet')}"
df_train = pd.read_parquet(train_path) if train_path.endswith('.parquet') else pd.read_excel(train_path)

# 2. Select Features (As used in Round {rnd})
# (Load from specific JSON if list is too long for this snippet)
features = {round_data.get('active_features_list', [])}

X = df_train[features]
y = df_train[['{self.config['data']['target_sin']}', '{self.config['data']['target_cos']}']]

# 3. Configure Model
model = {model_name}(
    {param_str},
    random_state=456
)

# 4. Train
print(f"Training {model_name} for Round {rnd}...")
model.fit(X, y)

# 5. Save
joblib.dump(model, 'recreated_model_round_{rnd:03d}.pkl')
print("Done.")
"""
        return code
