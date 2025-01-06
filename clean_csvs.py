import pandas as pd
import numpy as np

def process_csvs(file_paths):
    dfs = [pd.read_csv(path) for path in file_paths]
    
    # Process scores and identify rows to drop
    rows_to_drop = []
    for i in range(1, len(dfs[0])):
        # Calculate average scores across files
        row_averages = [df.iloc[i, -3:].astype(float).mean() for df in dfs]
        
        # Mark row for dropping if 2 or more files have zero average
        if sum(np.isclose(avg, 0.0, atol=1e-10) for avg in row_averages) >= 2:
            rows_to_drop.append(i)
    
    # Process and save each file
    for path, df in zip(file_paths, dfs):
        # Clean and renumber
        df_cleaned = df.drop(rows_to_drop)
        df_cleaned.iloc[2:, 0] = range(1, len(df_cleaned) - 2 + 1)
        
        # Save
        new_path = f"{path.rsplit('.', 1)[0]}_cleaned.csv"
        df_cleaned.to_csv(new_path, index=False)
        print(f"Saved {new_path} ({len(df)} rows → {len(df_cleaned)} rows)")

# File paths
file_paths = [
    "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022.csv",
    "/content/drive/MyDrive/eval_outputs/gpt-4o-mini-2024-07-18/combined_results_gpt-4o-mini-2024-07-18.csv",
    "/content/drive/MyDrive/eval_outputs/gemini-2.0-flash-exp/combined_results_gemini-2.0-flash-exp.csv"
]

process_csvs(file_paths)
