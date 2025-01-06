import pandas as pd
import os
import numpy as np

def process_csvs(file_paths):
    # Read all CSV files
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        
        # Get the last three columns before the last empty column (columns I, J, K)
        cols = df.columns.tolist()
        score_cols = cols[-3:]  # This will get the last three columns (I, J, K)
        
        # Convert score columns to numeric and fill NaN with 0
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"\nFirst 5 rows of scores from {os.path.basename(path)}:")
        print(df[score_cols].head())
        
        dfs.append(df)
    
    # Process each row
    rows_to_drop = []
    for i in range(1, len(dfs[0])):  # Start from 1 to skip header
        row_file_averages = []
        
        # Calculate average for each file's three score columns
        for df in dfs:
            scores = df.iloc[i, -3:].values  # Get last three columns
            avg = float(np.mean(scores))
            row_file_averages.append(avg)
        
        # Debug print for first few rows
        if i <= 5:
            print(f"\nRow {i}:")
            for j, df in enumerate(dfs):
                scores = df.iloc[i, -3:].values.tolist()
                print(f"File {j+1} scores: {scores} -> avg: {row_file_averages[j]:.2f}")
        
        # Count how many files have zero average for this row
        zero_count = sum(1 for avg in row_file_averages if np.isclose(avg, 0.0, atol=1e-10))
        
        if zero_count >= 2:
            rows_to_drop.append(i)
            print(f"\nRow {i} marked for dropping:")
            for j, avg in enumerate(row_file_averages):
                print(f"  File {j+1} average: {avg:.2f}")
    
    # Report number of rows to be excluded
    print(f"\nNumber of rows to be excluded: {len(rows_to_drop)}")
    if len(rows_to_drop) > 0:
        print("Rows to be dropped:", rows_to_drop)
    
    # Create cleaned versions of each file
    for i, path in enumerate(file_paths):
        df = dfs[i]
        df_cleaned = df.drop(rows_to_drop)
        
        base_path = os.path.splitext(path)[0]
        new_path = f"{base_path}_cleaned.csv"
        
        df_cleaned.to_csv(new_path, index=False)
        print(f"\nCreated cleaned file: {new_path}")
        print(f"Removed {len(rows_to_drop)} rows from {os.path.basename(path)}")
        print(f"Original file rows: {len(df)}, Cleaned file rows: {len(df_cleaned)}")

# File paths
file_paths = [
    "/content/drive/MyDrive/eval_outputs/claude-3-5-haiku-20241022/combined_results_claude-3-5-haiku-20241022.csv",
    "/content/drive/MyDrive/eval_outputs/gpt-4o-mini-2024-07-18/combined_results_gpt-4o-mini-2024-07-18.csv",
    "/content/drive/MyDrive/eval_outputs/gemini-2.0-flash-exp/combined_results_gemini-2.0-flash-exp.csv"
]

# Process the files
process_csvs(file_paths)