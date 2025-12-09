
import pandas as pd
import glob
import os

def combine_baseline_metrics(base_path):
    """
    Reads test and validation baseline metrics from all rounds and combines
    them into a single XLSX file.
    """
    all_metrics_data = []
    
    round_folders = sorted(glob.glob(os.path.join(base_path, "ROUND_*")))
    
    if not round_folders:
        print(f"No 'ROUND_*' folders found in '{base_path}'")
        return

    print(f"Found {len(round_folders)} round folders to process.")

    for round_folder in round_folders:
        round_name = os.path.basename(round_folder)
        metrics_dir = os.path.join(round_folder, "03_BASE_MODEL_RESULTS")

        for metric_type in ["test", "val"]:
            file_name = f"baseline_metrics_{metric_type}.xlsx"
            file_path = os.path.join(metrics_dir, file_name)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    df['round'] = round_name
                    df['set'] = metric_type
                    all_metrics_data.append(df)
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")

    if not all_metrics_data:
        print("No baseline metrics files were found.")
        return

    # Reorder columns to have 'round' and 'set' first
    final_df = pd.concat(all_metrics_data, ignore_index=True)
    cols = ['round', 'set'] + [col for col in final_df.columns if col not in ['round', 'set']]
    final_df = final_df[cols]
    
    output_filename = "combined_baseline_metrics.xlsx"
    final_df.to_excel(output_filename, index=False)
    
    print(f"\nSuccessfully combined metrics from {len(all_metrics_data)} files into '{output_filename}'")
    print(f"The combined file has {len(final_df)} rows.")

if __name__ == "__main__":
    base_results_path = "results_RiserAngle_ExtraTrees_RFE_167_to_1"
    combine_baseline_metrics(base_results_path)
