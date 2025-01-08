import pandas as pd
import json
import glob
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def process_files(input_dir, json_path, output_dir, start_batch, num_batches, samples_per_batch):
    # Load and extract relevant tags from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    start_idx = start_batch * samples_per_batch
    tags_list = [data[i].get('tags', []) for i in range(start_idx, start_idx + (num_batches * samples_per_batch))]
    max_tags = max(len(tags) for tags in tags_list)
    
    # Combine CSV files
    csv_files = sorted(glob.glob(os.path.join(input_dir, 'results_*.csv')))
    if not csv_files:
        raise ValueError("No CSV files found to combine")
    
    combined_df = pd.concat([pd.read_csv(csv_files[0])] + 
                          [pd.read_csv(f, header=0) for f in csv_files[1:]], 
                          ignore_index=True)
    
    # Reassign IDs and add tags
    combined_df['sample_id'] = range(1, len(combined_df) + 1)
    for i in range(max_tags):
        tag_col = f'tag{i+1}'
        combined_df[tag_col] = ''
        for idx, tags in enumerate(tags_list):
            if idx < len(combined_df) and i < len(tags):
                combined_df.at[idx, tag_col] = tags[i]
    
    combined_df.to_csv(os.path.join(output_dir, 'combined.csv'), index=False)
    logging.info("Combined CSV saved successfully")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--json-file', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--start-batch', type=int, required=True)
    parser.add_argument('--num-batches', type=int, required=True)
    parser.add_argument('--samples-per-batch', type=int, required=True)
    args = parser.parse_args()
    
    try:
        process_files(args.input_dir, args.json_file, args.output_dir, 
                     args.start_batch, args.num_batches, args.samples_per_batch)
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        raise

if __name__ == "__main__":
    main()
