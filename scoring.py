import argparse
import os
import json
import logging
import time
from pathlib import Path
import random
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
        self.dataset_path = args.dataset or "/content/anai/artifacts/synthetic.json"
        self.output_dir = args.output_dir or "/content/drive/MyDrive/eval_outputs"
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.temperature = args.temperature
        self.run_analysis = args.run_analysis

def setup_environment(config: Config) -> None:
    """Setup directories and environment"""
    os.makedirs(config.output_dir, exist_ok=True)
    random.seed(config.seed)
    
    # Set model in environment
    os.environ["INSPECT_EVAL_MODEL"] = config.model
    
    # Print configuration for visibility
    print(f"\nAHA benchmark ({config.batch_size} samples): {config.model}")
    print(f"max_tokens: 1000, temperature: {config.temperature}, seed: {config.seed}")

def load_and_sample_data(config: Config) -> List[Dict[str, Any]]:
    """Load dataset and sample specified number of observations"""
    with open(config.dataset_path, 'r') as f:
        full_data = json.load(f)
    
    # Create temporary sampled dataset file
    sampled_data = random.sample(full_data, min(config.batch_size, len(full_data)))
    sampled_file = os.path.join(os.path.dirname(config.dataset_path), "sampled_data.json")
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
        num_batches=1
    )

def main():
    parser = argparse.ArgumentParser(description="Run the AHA benchmark evaluation")
    parser.add_argument('--model', required=True, help='Model to evaluate')
    parser.add_argument('--scorer_models', 
                       default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-flash-002,openai/gpt-4o-mini-2024-07-18",
                       help='Comma-separated list of scoring models')
    parser.add_argument('--dataset', help='Path to dataset JSON')
    parser.add_argument('--output_dir', help='Directory for output files')
    parser.add_argument('--batch_size', type=int, default=3, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--run-analysis', action='store_true', help='Run logfileanalysis.py after evaluation')
    
    args = parser.parse_args()
    global config, dataset_path
    config = Config(args)
    
    try:
        setup_environment(config)
        sampled_file, data = load_and_sample_data(config)
        dataset_path = sampled_file
        
        # Run evaluation with inspect framework
        start_time = time.time()
        eval("aha_evaluation")
        end_time = time.time()
        
        # Print timing information
        total_time = end_time - start_time
        print(f"\ntotal time: {total_time:.2f}s")
        
        if config.run_analysis:
            os.system(f"python /content/anai/logfileanalysis.py --output-dir {config.output_dir}")
        
    finally:
        # Cleanup
        if os.path.exists(sampled_file):
            os.remove(sampled_file)

if __name__ == "__main__":
    main()