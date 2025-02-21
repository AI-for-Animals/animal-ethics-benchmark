import pandas as pd
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Split CSV file based on second column mapping')
    parser.add_argument('--combined_full_filepath', required=True,
                      help='Path to the combined full data CSV file')
    parser.add_argument('--public_mapping_filepath', required=True,
                      help='Path to the public mapping CSV file')
    parser.add_argument('--public_split_filepath', required=True,
                      help='Output path for public split CSV file')
    parser.add_argument('--private_split_filepath', required=True,
                      help='Output path for private split CSV file')
    
    args = parser.parse_args()

    try:
        # Read the CSV files
        combined_df = pd.read_csv(args.combined_full_filepath)
        public_mapping_df = pd.read_csv(args.public_mapping_filepath)
        
        # Get the names of the second columns
        combined_second_col = combined_df.columns[1]
        mapping_second_col = public_mapping_df.columns[1]
        
        # Create set of values from second column in public mapping
        public_values = set(public_mapping_df[mapping_second_col])
        
        # Split the combined dataframe
        public_df = combined_df[combined_df[combined_second_col].isin(public_values)]
        private_df = combined_df[~combined_df[combined_second_col].isin(public_values)]
        
        # Reset the index column in both dataframes to start from 1
        public_df.iloc[:, 0] = range(1, len(public_df) + 1)
        private_df.iloc[:, 0] = range(1, len(private_df) + 1)
        
        # Save the split dataframes
        public_df.to_csv(args.public_split_filepath, index=False)
        private_df.to_csv(args.private_split_filepath, index=False)
        
        # Print summary
        print(f"Split complete:")
        print(f"Public split rows: {len(public_df)}")
        print(f"Private split rows: {len(private_df)}")
        print(f"Total original rows: {len(combined_df)}")
        print(f"Mapping used second columns:")
        print(f"  - Combined file: '{combined_second_col}'")
        print(f"  - Public mapping file: '{mapping_second_col}'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()