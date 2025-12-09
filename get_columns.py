
import pandas as pd
import sys

def get_excel_columns(file_path):
    """Reads an Excel file and prints its column names."""
    try:
        df = pd.read_excel(file_path)
        print("Columns found in the Excel file:")
        print(list(df.columns))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Path to one of the prediction files
    target_file = "results_RiserAngle_ExtraTrees_RFE_167_to_1/ROUND_000/03_BASE_MODEL_RESULTS/baseline_predictions_test.xlsx"
    get_excel_columns(target_file)
