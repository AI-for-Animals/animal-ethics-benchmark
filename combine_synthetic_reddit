# Arturs Kanepajs 2025-01-18
# This file finalizes (1) combines csvs with synthetic and reddit results, and (2) overwrites IDs and tags from an existing combined file. 
# Output is a csv that contains aggregate benchmark results for a given model; and can be used accordingly for analysis, as input for other scripts. 

import pandas as pd
import os

# File paths - edit these as needed
SYNTHETIC_PATH = 'synthetic.csv'
REDDIT_PATH = 'reddit.csv'
RESULTS_PATH = 'results_gpt-4o-mini-2024-07-18.csv'
OUTPUT_PATH = 'combined.csv'  # Will be saved in the same directory as synthetic.csv

def combine_csvs():
    # Read the CSVs
    synthetic_df = pd.read_csv(SYNTHETIC_PATH)
    reddit_df = pd.read_csv(REDDIT_PATH)
    results_df = pd.read_csv(RESULTS_PATH)
    
    # Get number of columns from results file
    n_cols = len(results_df.columns)
    
    # Trim synthetic and reddit dataframes to match results column count
    synthetic_df = synthetic_df.iloc[:, :n_cols]
    reddit_df = reddit_df.iloc[:, :n_cols]
    
    # Combine synthetic and reddit (excluding reddit header)
    combined_df = pd.concat([synthetic_df, reddit_df], axis=0, ignore_index=True)
    
    # Get the column names from synthetic (these will be our base column names)
    final_columns = list(synthetic_df.columns)
    
    # Override first column and last 4 columns with data from results
    # First, ensure the combined_df has enough columns
    while len(combined_df.columns) < len(results_df.columns):
        combined_df[f'dummy_{len(combined_df.columns)}'] = None
        
    # Override first column
    combined_df.iloc[:, 0] = results_df.iloc[:, 0]
    final_columns[0] = results_df.columns[0]  # Update column name
    
    # Override last 4 columns
    for i in range(-4, 0):
        combined_df.iloc[:, i] = results_df.iloc[:, i]
        final_columns[i] = results_df.columns[i]  # Update column names
    
    # Set the final column names
    combined_df.columns = final_columns
    
    # Save the combined dataframe
    output_dir = os.path.dirname(SYNTHETIC_PATH)
    if output_dir:
        output_path = os.path.join(output_dir, OUTPUT_PATH)
    else:
        output_path = OUTPUT_PATH
        
    combined_df.to_csv(output_path, index=False)
    print(f"Combined CSV saved to: {output_path}")
    
    # Print some validation info
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
