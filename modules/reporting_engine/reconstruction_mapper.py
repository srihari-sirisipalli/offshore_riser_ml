import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any

class ReconstructionMapper:
    """
    Generates the 'model_reconstruction_mapping.xlsx' artifact.
    
    This file serves as a complete audit trail and instruction manual for 
    recreating any model from the RFE history.
    
    Sheets Generated:
    1. Rounds_Summary: High-level metrics and dropped features per round.
    2. Hyperparameters: Exact parameters used for the model in each round.
    3. Features_Per_Round: The specific subset of features active in that round.
    4. Data_Files: Reference paths to the datasets used.
    5. Recreation_Code: Copy-pasteable Python code to train the model.
    """

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate_mapping(self, 
                         rounds_history: List[Dict[str, Any]], 
                         data_files_info: Dict[str, str], 
                         output_dir: Path):
        """
        Builds and saves the reconstruction mapping Excel file.
        
        Args:
            rounds_history: List of round summary dictionaries (from RFEController).
            data_files_info: Dict containing paths/hashes of initial train/val/test files.
            output_dir: Directory where the Excel file should be saved.
        """
        self.logger.info("Generating Model Reconstruction Mapping...")
        output_path = output_dir / "model_reconstruction_mapping.xlsx"
        
        # Prepare DataFrames for each sheet
        df_summary = self._build_summary_sheet(rounds_history)
        df_params = self._build_params_sheet(rounds_history)
        df_features = self._build_features_sheet(rounds_history)
        df_files = self._build_files_sheet(rounds_history, data_files_info)
        df_code = self._build_code_sheet(rounds_history, data_files_info)

        # Write to Excel with formatting
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_summary.to_excel(writer, sheet_name='Rounds_Summary', index=False)
                df_params.to_excel(writer, sheet_name='Hyperparameters', index=False)
                df_features.to_excel(writer, sheet_name='Features_Per_Round', index=False)
                df_files.to_excel(writer, sheet_name='Data_Files', index=False)
                df_code.to_excel(writer, sheet_name='Recreation_Code', index=False)
                
                self._auto_adjust_columns(writer)
                
            self.logger.info(f"✓ Reconstruction mapping saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to write reconstruction mapping: {e}")
            raise

    def _build_summary_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            metrics = r.get('metrics', {})
            data.append({
                'Round': r['round'],
                'N_Features': r['n_features'],
                'Dropped_Feature': r.get('dropped_feature', 'None'),
                'Val_CMAE': metrics.get('cmae', metrics.get('val_cmae', 0)),
                'Val_Accuracy_5deg': metrics.get('accuracy_at_5deg', 0),
                'Stop_Reason': r.get('stopping_reason', '')
            })
        return pd.DataFrame(data)

    def _build_params_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            row = {'Round': r['round']}
            # Flatten params
            params = r.get('hyperparameters', {})
            for k, v in params.items():
                row[k] = v
            data.append(row)
        return pd.DataFrame(data)

    def _build_features_sheet(self, history: List[Dict]) -> pd.DataFrame:
        data = []
        for r in history:
            # We assume RFEController might save the full list, or we reconstruct it.
            # In rounds_history from RFEController, we usually store 'n_features'.
            # Ideally, RFEController should pass the list of ACTIVE features in the history object.
            # If strictly list is needed, ensure RFEController populates 'features_list'.
            
            feats = r.get('active_features_list', []) # Expecting list of strings
            feat_str = ", ".join(feats) if isinstance(feats, list) else "Check feature_list.json in round folder"
            
            data.append({
                'Round': r['round'],
                'N_Features': r['n_features'],
                'Active_Features': feat_str
            })
        return pd.DataFrame(data)

    def _build_files_sheet(self, history: List[Dict], data_info: Dict) -> pd.DataFrame:
        data = []
        for r in history:
            data.append({
                'Round': r['round'],
                'Train_File': data_info.get('train_path', 'train.xlsx'),
                'Val_File': data_info.get('val_path', 'val.xlsx'),
                'Input_Hash': data_info.get('input_hash', 'N/A')
            })
        return pd.DataFrame(data)

    def _build_code_sheet(self, history: List[Dict], data_info: Dict) -> pd.DataFrame:
        data = []
        for r in history:
            code = self._generate_python_snippet(r, data_info)
            data.append({
                'Round': r['round'],
                'Recreation_Code': code
            })
        return pd.DataFrame(data)

    def _generate_python_snippet(self, round_data: Dict, data_info: Dict) -> str:
        """Constructs a string of valid Python code to train the model."""
        rnd = round_data['round']
        params = round_data.get('hyperparameters', {})
        model_name = params.get('model_name', 'ExtraTreesRegressor') # Default or extracted
        
        # Clean params for code generation (remove metadata keys)
        train_params = {k:v for k,v in params.items() if k not in ['round_tuned', 'model_name']}
        
        # Indented params string
        param_str = ",\n    ".join([f"{k}={repr(v)}" for k,v in train_params.items()])
        
        code = f"""
# ============================================
# RECREATION CODE FOR ROUND {rnd}
# ============================================
import pandas as pd
import joblib
from sklearn.ensemble import {model_name}

# 1. Load Data
train_path = r"{data_info.get('train_path', 'train.xlsx')}"
df_train = pd.read_excel(train_path)

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

    def _auto_adjust_columns(self, writer):
        """Helper to adjust column widths in Excel."""
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                try:
                    max_len = max(len(str(cell.value)) for cell in column if cell.value)
                    adj_width = min(max_len + 2, 100) # Cap width
                    worksheet.column_dimensions[column[0].column_letter].width = adj_width
                except:
                    pass