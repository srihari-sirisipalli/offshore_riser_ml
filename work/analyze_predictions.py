import pandas as pd
import glob
import os

def analyze_base_model_predictions(base_path):
    """
    Reads test and validation baseline predictions from all rounds, 
    filters for high errors, and compiles them into a single XLSX file. 
    It also identifies consistently poorly predicted indices.
    """
    high_error_data = []
    
    round_folders = sorted(glob.glob(os.path.join(base_path, "ROUND_*")))
    
    if not round_folders:
        print(f"No 'ROUND_*' folders found in '{base_path}'")
        return

    print(f"Found {len(round_folders)} round folders to analyze.")

    for round_folder in round_folders:
        round_name = os.path.basename(round_folder)
        predictions_dir = os.path.join(round_folder, "03_BASE_MODEL_RESULTS")

        for pred_type in ["test", "val"]:
            file_name = f"baseline_predictions_{pred_type}.xlsx"
            file_path = os.path.join(predictions_dir, file_name)
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path)
                    
                    # Using the correct column names discovered: 'abs_error', 'true_angle', 'pred_angle', 'row_index'
                    high_error_df = df[df['abs_error'] > 10].copy()
                    
                    if not high_error_df.empty:
                        high_error_df['round'] = round_name
                        high_error_df['set'] = pred_type
                        
                        # Rename 'pred_angle' to 'predicted_angle' for the final report to match user's request
                        high_error_df.rename(columns={'pred_angle': 'predicted_angle', 'row_index': 'original_index'}, inplace=True)
                        
                        high_error_data.append(high_error_df[['round', 'set', 'original_index', 'true_angle', 'predicted_angle', 'abs_error']])

                except KeyError as e:
                    print(f"Column not found in {file_path}: {e}. This may happen if the file structure is inconsistent.")
                    print("Skipping this file.")
                    continue
                except Exception as e:
                    print(f"Could not process {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")

    if not high_error_data:
        print("No baseline predictions with absolute error greater than 10 degrees were found.")
        return

    final_df = pd.concat(high_error_data, ignore_index=True)
    
    output_filename = "high_error_baseline_predictions.xlsx"
    final_df.to_excel(output_filename, index=False)
    print(f"\nAnalysis complete. High-error predictions saved to '{output_filename}'")

    print("\n--- Analysis of Persistently High-Error Points ---")
    persistent_errors = final_df.groupby('original_index')['round'].nunique().sort_values(ascending=False)
    
    print("Top 10 most frequently high-error data points (index and count of rounds):")
    print(persistent_errors.head(10))
    
    total_rounds = len(round_folders)
    # Let's define "frequent" as appearing in more than 50% of rounds, to catch more cases.
    frequent_threshold = total_rounds * 0.5 
    frequent_offenders = persistent_errors[persistent_errors > frequent_threshold]
    
    if not frequent_offenders.empty:
        print(f"\nIndices with high error in over {frequent_threshold:.0f} of the {total_rounds} rounds ({len(frequent_offenders)} points):")
        print(list(frequent_offenders.index))
    else:
        print(f"\nNo data points showed high error in more than {frequent_threshold:.0f} of the rounds.")


if __name__ == "__main__":
    base_results_path = "results_RiserAngle_ExtraTrees_RFE_167_to_1"
    analyze_base_model_predictions(base_results_path)
