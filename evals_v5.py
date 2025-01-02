import subprocess
import argparse
from importlib.metadata import distribution, PackageNotFoundError

def check_package(package):
    package_name = package.split('#egg=')[1] if '#egg=' in package else package
    try:
        distribution(package_name)
        return True
    except PackageNotFoundError:
        return False

def install_package(package):
    package_name = package.split('#egg=')[1] if '#egg=' in package else package
    if not check_package(package_name):
        print(f"Installing {package_name}...")
        subprocess.run(['pip', 'install', '--quiet', package], check=True)

def install_packages():
    packages = [
        'inspect-ai',
        'openai',
        'anthropic',
        'google-generativeai',
        'mistralai',
        'krippendorff',
        'git+https://github.com/UKGovernmentBEIS/inspect_evals#egg=inspect_evals'
    ]
    for package in packages:
        try:
            install_package(package)
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}")

def main():
    install_packages()

    parser = argparse.ArgumentParser(description="Run the ANAI evaluation task.")
    parser.add_argument('--model', required=True, help='Model to be evaluated')
    parser.add_argument('--scorer_models', required=True, help='Comma-separated list of models for scoring')
    parser.add_argument('--dataset', required=True, help='Path to the dataset')
    parser.add_argument('--limit', type=int, help='Limit the number of samples to evaluate')
    parser.add_argument('--temperature', type=float, help='Temperature setting')
    parser.add_argument('--seed', type=int, help='Random seed')
    args = parser.parse_args()

    cmd = [
        'python', '/content/anai/anai_inspect_task_v5.py',
        '--model', args.model,
        '--scorer_models', args.scorer_models,
        '--dataset', args.dataset,
    ]
    if args.limit is not None:
        cmd.extend(['--limit', str(args.limit)])
    if args.temperature is not None:
        cmd.extend(['--temperature', str(args.temperature)])
    if args.seed is not None:
        cmd.extend(['--seed', str(args.seed)])
    try:
        subprocess.run(cmd, check=True)
        print("Evaluation completed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to execute the evaluation script.")

if __name__ == "__main__":
    main()
