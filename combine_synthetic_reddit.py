# Arturs Kanepajs 2025-01-18
# This file finalizes (1) combines csvs with synthetic and reddit results, and (2) overwrites IDs and tags from an existing combined file. 
# Output is a csv that contains aggregate benchmark results for a given model; and can be used accordingly for analysis, as input for other scripts. 

import pandas as pd
import os

# File paths - edit these as needed
SYNTHETIC_PATH = '/content/drive/MyDrive/eval_outputs/synthetic_gemini-1.5-flash-002.csv'
REDDIT_PATH = '/content/drive/MyDrive/eval_outputs/reddit_gemini-1.5-flash-002.csv'
RESULTS_PATH = '/content/drive/MyDrive/eval_outputs/results_gpt-4o-mini-2024-07-18.csv' # this is an example, which has the results in correct format
OUTPUT_PATH = '/content/drive/MyDrive/eval_outputs/results_gemini-1.5-flash-002.csv'

def combine_csvs():
    # Read the CSVs
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    reddit_df = pd.read_csv(REDDIT_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    
    print("\nInitial column counts:")
    print(f"Synthetic columns: {len(synthetic_df.columns)}")
    print(f"Reddit columns: {len(reddit_df.columns)}")
    print(f"Results columns: {len(results_df.columns)}")
    
    # Get number of columns needed
    n_cols = len(results_df.columns)
    
    # Store original synthetic columns and extend if needed
    synthetic_columns = list(synthetic_df.columns)
    while len(synthetic_columns) < n_cols:
        synthetic_columns.append(f'dummy_{len(synthetic_columns)}')
    
    # Add any missing columns to synthetic_df
    for col in synthetic_columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = None
            
    # Ensure reddit_df has the same columns in the same order
    for col in synthetic_columns:
        if col not in reddit_df.columns:
            reddit_df[col] = None
    
    # Reorder columns to match
    synthetic_df = synthetic_df[synthetic_columns]
    reddit_df = reddit_df[synthetic_columns]
    
    print("\nAfter adding dummy columns:")
    print(f"Synthetic columns: {len(synthetic_df.columns)}")
    print(f"Reddit columns: {len(reddit_df.columns)}")
    
    # Combine synthetic and reddit with explicit columns
    combined_df = pd.concat([synthetic_df, reddit_df], axis=0, ignore_index=True)
    
    print(f"\nCombined dataframe columns: {len(combined_df.columns)}")
    
    # Get the column names from results file for the columns we'll override
    results_columns = list(results_df.columns)
    
    # Create final column list from synthetic, but replace first and last 4 with results
    final_columns = synthetic_columns.copy()
    final_columns[0] = results_columns[0]  # First column
    final_columns[-4:] = results_columns[-4:]  # Last 4 columns
    
    print(f"Length of final_columns: {len(final_columns)}")
    print(f"Length of combined_df columns: {len(combined_df.columns)}")
    
    # Override first column data
    combined_df.iloc[:, 0] = results_df.iloc[:, 0]
    
    # Override last 4 columns data
    for i in range(-4, 0):
        combined_df.iloc[:, i] = results_df.iloc[:, i]
    
    # Set the final column names
    combined_df.columns = final_columns
    
    # Save the combined dataframe
    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCombined CSV saved to: {OUTPUT_PATH}")
    
    # Print validation info
    print("\nValidation Information:")
    print(f"Number of rows from synthetic: {len(synthetic_df)}")
    print(f"Number of rows from reddit: {len(reddit_df)}")
    print(f"Total rows in combined file: {len(combined_df)}")
    print(f"Number of columns: {len(combined_df.columns)}")
    print("\nColumn names in combined file:")
    for i, col in enumerate(combined_df.columns):
        print(f"{i+1}. {col}")

if __name__ == "__main__":
    combine_csvs()
