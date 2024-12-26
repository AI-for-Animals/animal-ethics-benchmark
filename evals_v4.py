import subprocess
from importlib.metadata import version, PackageNotFoundError
import sys
import os
import argparse

def get_installed_version(package_name):
    """Check if the given package is installed and return its version if available."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def install_packages():
    """Check and install the required packages."""
    packages = [
        'inspect-ai',
        'openai',
        'anthropic',
        'google-generativeai',
        'mistralai',
        'git+https://github.com/UKGovernmentBEIS/inspect_evals#egg=inspect_evals'
    ]

    for package in packages:
        package_name = package.split('#egg=')[-1] if 'git+' in package else package
        if not get_installed_version(package_name):
            try:
                print(f"Installing {package}...")
                subprocess.run(['pip', 'install', package], check=True)
                print(f"Successfully installed {package_name}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package}. Please check your connection or package source.")
        else:
            print(f"{package_name} is already installed")

def main():
    # Install required packages
    install_packages()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the ANAI evaluation task.")
    parser.add_argument('--model', type=str, required=True, help='Model to be evaluated')
    parser.add_argument('--scorer_models', type=str, required=True, help='Comma-separated list of models for scoring')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of samples to evaluate')
    args = parser.parse_args()

    # Run the anai_inspect_task_v3.py script to register and execute the task
    cmd = [
        'python', '/content/anai/anai_inspect_task_v4.py',
        '--model', args.model,
        '--scorer_models', args.scorer_models,
        '--dataset', args.dataset,
        '--limit', str(args.limit)
    ]
    try:
        subprocess.run(cmd, check=True)
        print("Evaluation completed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to execute the evaluation script. Please review the arguments and try again.")

if __name__ == "__main__":
    main()
