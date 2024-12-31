import subprocess
from importlib.metadata import version, PackageNotFoundError
import argparse

def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def is_upgrade_needed(package, required_version=None):
    """Check if an upgrade is needed based on the required version."""
    installed_version = get_installed_version(package)
    if installed_version is None:
        return True  # Package not installed
    if required_version is None:
        return False  # No specific version required
    # Compare installed version with required version
    from packaging.version import parse
    return parse(installed_version) < parse(required_version)

def install_or_upgrade_package(package, required_version=None):
    package_name = package.split('#egg=')[-1] if 'git+' in package else package
    if get_installed_version(package_name) is None:
        print(f"Installing {package}...")
        subprocess.run(['pip', 'install', package], check=True)
        print(f"Successfully installed {package_name}")
    elif is_upgrade_needed(package_name, required_version):
        print(f"{package_name} version does not meet requirements. Upgrading...")
        subprocess.run(['pip', 'install', '--upgrade', package], check=True)
        print(f"Successfully upgraded {package_name}")
    else:
        print(f"{package_name} is already installed and meets the required version.")

def install_packages():
    packages = [
        ('inspect-ai', '0.3.55'),
        ('openai', '1.58.1'),
        ('anthropic', '0.42.0'),
        ('google-generativeai', '0.8.3'),
        ('mistralai', '1.2.5'),
        ('git+https://github.com/UKGovernmentBEIS/inspect_evals#egg=inspect_evals', None)
    ]
    for package, required_version in packages:
        try:
            install_or_upgrade_package(package, required_version)
        except subprocess.CalledProcessError:
            print(f"Failed to install or upgrade {package}. Check your connection or package source.")

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
        print("Failed to execute the evaluation script. Please review the arguments and try again.")

if __name__ == "__main__":
    main()
