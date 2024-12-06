import subprocess
import pkg_resources
import sys
import os
import re
import argparse

def get_installed_version(package):
    try:
        if package.startswith('git+'): 
            return pkg_resources.get_distribution(package.split('/')[-1]).version
        return pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        return None

def install_packages():
    packages = [
        'inspect-ai',
        'git+https://github.com/UKGovernmentBEIS/inspect_evals',
        'openai', 'anthropic', 'google-generativeai', 'mistralai'
    ]
    for package in packages:
        if not get_installed_version(package):
            print(f"Installing {package}...")
            subprocess.run(['pip', 'install', package], check=True)
        else:
            print(f"{package} is already installed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--scorer-models', nargs='+', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    script_path = os.path.relpath('/content/anai/anai_inspect_task_v3.py', os.getcwd())
    
    # Keep model names intact
    os.environ['SCORER_MODELS'] = ','.join(args.scorer_models)
    os.environ['DATASET'] = args.dataset

    cmd = ['inspect', 'eval', script_path, '--model', args.model, '--limit', str(args.limit)]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
