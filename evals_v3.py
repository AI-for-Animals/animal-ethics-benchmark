"""
Model Evaluation Script using INSPECT Framework for Google Colab

This script runs evaluations on various Large Language Models (LLMs) using the INSPECT framework.
"""

import subprocess
import os
from pathlib import Path

#def install_packages():
#    """Install required packages using pip."""
#    packages = [
#        'inspect-ai',
#        'git+https://github.com/UKGovernmentBEIS/inspect_evals',
#        'openai',
#        'anthropic',
#        'google-generativeai',
#        'mistralai'
#    ]
#    
#    for package in packages:
#        try:
#            if package == 'google-generativeai':
#                subprocess.run(['pip', 'install', '--upgrade', package], check=True)
#            else:
#                subprocess.run(['pip', 'install', package], check=True)
#        except subprocess.CalledProcessError as e:
#            print(f"Error installing {package}: {e}")
#            raise

def verify_task_file():
    """Verify that the task file exists and is in the correct location."""
    task_path = Path('/content/anai/anai_inspect_task_v3.py')
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found at {task_path}")
    print(f"Found task file at: {task_path}")
    return task_path

def verify_dataset():
    """Verify that the dataset directory exists."""
    dataset_dir = Path('/content/anai/artifacts')
    if not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created dataset directory at {dataset_dir}")
    return dataset_dir

def run_evaluation(model, task_path):
    """Run evaluation for a specific model."""
    try:
        os.chdir(task_path.parent)  # Change to the directory containing the task file

        cmd = [
            'inspect', 'eval',
            str(task_path.name),  # Use just the filename since we're in the correct directory
            '--model', model,
            '--limit', '300'
        ]
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation for {model}: {e}")
        raise

def main():
    """Main function to run the evaluation process."""
    try:
        task_path = verify_task_file()
        dataset_dir = verify_dataset()

        # Define available models
        models = {
            'claude_haiku': 'anthropic/claude-3-haiku-20240307',
            'claude_sonnet': 'anthropic/claude-3-5-sonnet-20241022',
            'gpt4': 'openai/gpt-4o-2024-08-06',
            'gemini': 'google/gemini-1.5-pro-002',
            'llama': 'together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
            'openai_preview': 'openai/o1-preview-2024-09-12',
            'openai_mini': 'openai/gpt-4o-mini-2024-07-18',
            'mistral': 'mistral/mistral-large-2407'
        }
        
        # Example usage: Claude Haiku
        #print(f"Running evaluation for Claude Haiku model...")
        #run_evaluation(models['claude_haiku'], task_path)

        # Uncomment below lines to run evaluations for other models
        # run_evaluation(models['claude_sonnet'], task_path)
        # run_evaluation(models['gpt4'], task_path)
        # run_evaluation(models['gemini'], task_path)
        # run_evaluation(models['llama'], task_path)
        # run_evaluation(models['openai_preview'], task_path)
        # run_evaluation(models['mistral'], task_path)
        run_evaluation(models['openai_mini'], task_path)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
