import argparse
import subprocess
import asyncio
from importlib.metadata import version, PackageNotFoundError
import requests

def get_latest_version(package_name):
    try:
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        return response.json()["info"]["version"]
    except:
        return None

def check_package_status(package):
    package_name = package.split('#egg=')[1] if '#egg=' in package else package
    try:
        current_version = version(package_name)
        latest_version = get_latest_version(package_name)
        if latest_version and current_version != latest_version:
            return "upgrade", current_version, latest_version
        return "ok", current_version, current_version
    except PackageNotFoundError:
        return "install", None, None

def install_or_upgrade_package(package, action, current_version=None, latest_version=None):
    package_name = package.split('#egg=')[1] if '#egg=' in package else package
    
    if action == "install":
        print(f"Installing {package_name}...")
        cmd = ['pip', 'install', package]
    elif action == "upgrade":
        print(f"Upgrading {package_name} from {current_version} to {latest_version}...")
        cmd = ['pip', 'install', '--upgrade', package]
    else:
        print(f"✓ {package_name} is up to date (version {current_version})")
        return True

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"{'Installed' if action == 'install' else 'Upgraded'} {package_name} successfully")
            return True
        else:
            print(f"Failed to {action} {package_name}")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error during {action} of {package_name}: {str(e)}")
        return False

def check_and_install_packages():
    packages = [
        'inspect-ai',
        'openai',
        'anthropic',
        'google-generativeai',
        'mistralai',
        'krippendorff',
        'git+https://github.com/UKGovernmentBEIS/inspect_evals#egg=inspect_evals'
    ]
    
    print("Checking package status...")
    all_successful = True
    for package in packages:
        action, current_version, latest_version = check_package_status(package)
        if not install_or_upgrade_package(package, action, current_version, latest_version):
            all_successful = False
    
    if not all_successful:
        print("\n⚠️ Some packages failed to install or upgrade.")
        print("Please review the error messages above.")
    else:
        print("\n✓ All packages are installed and up to date.")
    
    return all_successful

def main():
    if not check_and_install_packages():
        print("Exiting due to package installation/upgrade issues.")
        return

    parser = argparse.ArgumentParser(description="Run the ANAI evaluation task.")
    parser.add_argument('--model', required=True, help='Model to be evaluated')
    parser.add_argument('--scorer_models', required=True, help='Comma-separated list of models for scoring')
    parser.add_argument('--dataset', required=True, help='Path to the dataset')
    parser.add_argument('--limit', type=int, help='Limit the number of samples to evaluate')
    parser.add_argument('--temperature', type=float, help='Temperature setting')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--max-connections', type=int, default=20, help='Maximum number of concurrent model connections')
    parser.add_argument('--max-tasks', type=int, default=5, help='Maximum number of concurrent tasks')
    args = parser.parse_args()

    cmd = [
        'python', '/content/anai/anai_inspect_task_v5.py',
        '--model', args.model,
        '--scorer_models', args.scorer_models,
        '--dataset', args.dataset,
        '--max-connections', str(args.max_connections),
        '--max-tasks', str(args.max_tasks)
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
