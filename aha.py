import argparse
import os
import json
import logging
import time
from pathlib import Path
import random
import shutil
import glob
import pandas as pd
from typing import List, Dict, Any
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa

# Configure logging for inspect framework
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class Config:
    def __init__(self, args):
        self.model = args.model
        self.scorer_models = args.scorer_models.split(',')
        self.dataset_path = args.dataset or "/content/anai/artifacts/aha_data.json"
        self.output_dir = args.output_dir or "/content/drive/MyDrive/eval_outputs"
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.temperature = args.temperature
        self.run_analysis = args.run_analysis
        self.num_batches = args.num_batches
        self.current_batch = 1

def setup_environment(config: Config) -> None:
    """Setup directories and environment"""
    os.makedirs(config.output_dir, exist_ok=True)
    random.seed(config.seed)
    
    # Set model in environment
    os.environ["INSPECT_EVAL_MODEL"] = config.model
    
    # Print configuration for visibility
    print(f"\nAHA benchmark ({config.batch_size * config.num_batches} total samples, {config.num_batches} batches of {config.batch_size}): {config.model}")
    print(f"max_tokens: 1000, temperature: {config.temperature}, seed: {config.seed}")

def load_and_sample_data(config: Config, full_data: List[Dict[str, Any]], used_indices: set) -> tuple[str, List[Dict[str, Any]]]:
    """Load dataset and sample specified number of observations for current batch"""
    # Get indices for current batch avoiding previously used indices
    available_indices = set(range(len(full_data))) - used_indices
    if len(available_indices) < config.batch_size:
        used_indices.clear()
        available_indices = set(range(len(full_data)))
    
    batch_indices = set(random.sample(list(available_indices), config.batch_size))
    used_indices.update(batch_indices)
    
    sampled_data = [full_data[i] for i in batch_indices]
    sampled_file = os.path.join(os.path.dirname(config.dataset_path), 
                                f"sampled_data_batch{config.current_batch}.json")
    with open(sampled_file, 'w') as f:
        json.dump(sampled_data, f)
    
    return sampled_file, sampled_data

def record_to_sample(record: Dict[str, Any]) -> Sample:
    """Convert record to Sample format"""
    return Sample(
        input=f"{record['question']} Your answer must be EXACTLY 30–40 words long.",
        target=SOLVER_INSTRUCTIONS,
        metadata=record.get("generation_tags", {})
    )

@task
def aha_evaluation() -> Task:
    """Main evaluation task"""
    global config, dataset_path
    dataset = json_dataset(dataset_path, sample_fields=record_to_sample)
    
    return Task(
        dataset=dataset,
        solver=generate(
            max_tokens=1000,
            temperature=config.temperature,
            cache=True
        ),
        scorer=final_digit_model_graded_qa(config.scorer_models),
        max_retries=3,
        max_messages=20,
        num_batches=1  # We handle batches at a higher level
    )

def combine_csv_files(config: Config, full_data: List[Dict[str, Any]], start_batch: int):
    """Combine CSV files from multiple batches and add tags"""
    try:
        # Get all CSV files in output directory
        csv_files = sorted(glob.glob(os.path.join(config.output_dir, 'results_*.csv')))
        if not csv_files:
            logging.error("No CSV files found to combine")
            return

        # Combine CSV files
        combined_df = pd.concat(
            [pd.read_csv(csv_files[0])] +
            [pd.read_csv(f, header=0) for f in csv_files[1:]],
            ignore_index=True
        )

        # Extract tags from the original data (if needed)
        start_idx = start_batch * config.batch_size
        tags_list = [
            data.get('tags', [])
            for data in full_data[start_idx:start_idx + (config.num_batches * config.batch_size)]
        ]
        max_tags = max((len(tags) for tags in tags_list), default=0)

        # Reassign IDs and add tags
        combined_df['sample_id'] = range(1, len(combined_df) + 1)
        for i in range(max_tags):
            tag_col = f'tag{i+1}'
            combined_df[tag_col] = ''
            for idx, tags in enumerate(tags_list):
                if idx < len(combined_df) and i < len(tags):
                    combined_df.at[idx, tag_col] = tags[i]

        # Save combined file
        combined_path = os.path.join(config.output_dir, 'results_combined.csv')
        combined_df.to_csv(combined_path, index=False)
        logging.info(f"Combined CSV saved to: {combined_path}")

    except Exception as e:
        logging.error(f"Error combining CSV files: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run the AHA benchmark evaluation")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--scorer_models', 
                       default="anthropic/claude-3-5-haiku-20241022,google/gemini-1.5-flash-002,openai/gpt-4o-mini-2024-07-18",
                       help='Comma-separated list of scoring models')
    parser.add_argument('--dataset', help='Path to dataset JSON')
    parser.add_argument('--output_dir', help='Directory for output files')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for generation')
    parser.add_argument('--run-analysis', action='store_true', help='Run logfileanalysis.py after evaluation')
    parser.add_argument('--start-batch', type=int, default=0, help='Starting batch number for combining results')
    
    args = parser.parse_args()
    global config, dataset_path
    config = Config(args)
    
    try:
        setup_environment(config)
        
        # Load full dataset once
        with open(config.dataset_path, 'r') as f:
            full_data = json.load(f)
        
        start_time = time.time()
        sampled_files = []
        used_indices = set()
        
        # Run evaluation for each batch
        for batch in range(config.num_batches):
            config.current_batch = batch + 1
            sampled_file, _ = load_and_sample_data(config, full_data, used_indices)
            sampled_files.append(sampled_file)
            dataset_path = sampled_file
            
            print(f"\nProcessing batch {config.current_batch}/{config.num_batches}")
            eval("aha_evaluation")
            
            if config.run_analysis:
                # Get the latest log file for this batch
                log_dir = "./logs"
                latest_logs = [f for f in os.listdir(log_dir) if f.endswith('.eval')]
                if latest_logs:
                    latest_log = max(latest_logs, key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))
                    latest_log_path = os.path.join(log_dir, latest_log)
                    
                    # Run analysis for this batch
                    print(f"\nResults for batch {config.current_batch}:")
                    os.system(f"python /content/anai/logfileanalysis.py --log-file {latest_log_path} --output-dir {config.output_dir}")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal time for all batches: {total_time:.2f}s")
        
        # Combine CSV files if analysis was run
        if config.run_analysis:
            print("\nCombining results from all batches...")
            combine_csv_files(config, full_data, args.start_batch)

            # NEW SECTION: Run final analysis on results_combined.csv
            combined_path = os.path.join(config.output_dir, 'results_combined.csv')
            if os.path.exists(combined_path):
                print("\nRunning final analysis on results_combined.csv ...")
                os.system(f"python /content/anai/logfileanalysis.py --csv-file {combined_path} --output-dir {config.output_dir}")
            else:
                print("results_combined.csv not found, skipping final CSV-based analysis")

    finally:
        # Cleanup
        for sampled_file in sampled_files:
            if os.path.exists(sampled_file):
                os.remove(sampled_file)

if __name__ == "__main__":
    main()
