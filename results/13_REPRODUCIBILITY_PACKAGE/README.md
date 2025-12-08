# AI-Driven Riser Prediction - Reproducibility Package

**Run ID:** 20251208_145702
**Generated:** 2025-12-08 15:04:49

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

model = joblib.load('final_model.pkl')
# X_new = ...
# preds = model.predict(X_new)
```
