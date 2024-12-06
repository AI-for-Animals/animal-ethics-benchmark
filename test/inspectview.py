#!/usr/bin/env python3
"""
INSPECT Framework Monitoring and Tunneling Script
...
"""

import subprocess
import sys
import argparse
import os
import socket

# Install required packages
def install_requirements():
    """Install required packages using pip."""
    packages = ['pyngrok', 'flask', 'psutil']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

install_requirements()

# Import required libraries
from pyngrok import ngrok
import time
import signal
import psutil
from google.colab import userdata

# Helper function to kill a process and its children
def kill_process_and_children(proc_pid):
    try:
        parent = psutil.Process(proc_pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

# Wait for inspect view to be ready
def wait_for_inspect_ready(port, timeout=120):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                result = sock.connect_ex(('127.0.0.1', port))
                if result == 0:
                    print("\n✅ Inspect view is ready!")
                    return True
        except Exception as e:
            print(f"\nSocket error: {str(e)}")
        time.sleep(1)
        print(f"\rWaiting for inspect view to initialize... {int(time.time() - start_time)}s", end="")
    print("\n❌ Timeout waiting for inspect view to initialize")
    return False

# Wait for logs to be generated
def wait_for_logs(log_dir, timeout=30, retries=3):
    log_dir = os.path.abspath(log_dir)
    print(f"🔍 Checking log directory: {log_dir}")
    start_time = time.time()
    
    for attempt in range(retries):
        print(f"Attempt {attempt + 1}/{retries} to find log files...")
        while time.time() - start_time < timeout:
            log_files = [f for f in os.listdir(log_dir) if f.endswith(".eval")]
            if log_files:
                print(f"✅ Logs detected: {log_files}")
                return True
            time.sleep(1)

        print(f"❌ No log files detected in attempt {attempt + 1}, retrying...")

    print("❌ No log files detected within all retry attempts")
    return False

# Clean start for the inspect session
def clean_start_with_inspect(authtoken, port=7575, log_level="info", log_dir="/content/logs"):
    try:
        # Clean up existing inspect and ngrok processes
        print("🧹 Cleaning up old processes...")

        for proc_name in ["inspect", "ngrok"]:
            try:
                procs = subprocess.check_output(["pgrep", proc_name]).decode().split()
                for pid in procs:
                    kill_process_and_children(int(pid))
            except subprocess.CalledProcessError:
                pass

        # Clean up any process using the port
        try:
            port_processes = subprocess.check_output(["lsof", "-t", "-i:{}".format(port)]).decode().split()
            for pid in port_processes:
                kill_process_and_children(int(pid))
        except subprocess.CalledProcessError:
            pass

        time.sleep(2)

        # Ensure the log directory exists and has read permissions
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            print(f"⚠️ Log directory {log_dir} does not exist. Creating it now...")
            os.makedirs(log_dir)

        if not os.access(log_dir, os.R_OK):
            print(f"Adjusting permissions for the log directory: {log_dir}")
            os.chmod(log_dir, 0o755)

        # Wait for logs to be generated before starting the inspect view
        if not wait_for_logs(log_dir):
            print("❌ Log files not found, aborting...")
            return None

        # Set ngrok authtoken
        print("🔑 Setting up ngrok authtoken...")
        ngrok.set_auth_token(authtoken)

        # Start the inspect view with specified port
        print(f"🔍 Starting inspect view on port {port} with logs from {log_dir}...")
        inspect_process = subprocess.Popen(
            [
                "inspect", "view", "start",
                "--port", str(port),
                "--log-level", log_level,
                "--host", "127.0.0.1",
                "--log-dir", log_dir
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check if the process starts properly
        time.sleep(2)
        if inspect_process.poll() is not None:
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"❌ Inspect process failed immediately!")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            raise Exception("Inspect failed to start")

        # Wait for inspect view to be ready
        if not wait_for_inspect_ready(port, timeout=120):
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"\nProcess STDERR: {stderr}")
            print(f"Process STDOUT: {stdout}")
            raise Exception("Inspect failed to initialize within timeout period")

        # Start ngrok tunnel
        print("🚀 Starting ngrok tunnel...")
        tunnel = ngrok.connect(port)

        print(f"\n✅ Setup complete!")
        print(f"📡 Public URL: {tunnel.public_url}")
        print("\nImportant Notes:")
        print("1. The inspect interface is now ready to use")
        print("2. Keep this cell running to maintain the tunnel")
        print("3. If you need to stop, use the interrupt kernel button (■)")
        print(f"4. Inspect view is running on local port {port}")
        print(f"5. Reading logs from: {log_dir}")

        return {
            'url': tunnel.public_url,
            'process': inspect_process,
            'port': port,
            'log_dir': log_dir
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if 'inspect_process' in locals():
            kill_process_and_children(inspect_process.pid)
        return None

# Monitor the inspect session
def monitor_inspect_session(session_info):
    if not session_info:
        return

    start_time = time.time()
    try:
        while True:
            if session_info['process'].poll() is not None:
                print("\n❌ Inspect view process has terminated!")
                break

            elapsed = int(time.time() - start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60

            print(f"\rRunning for: {hours:02d}:{minutes:02d}:{seconds:02d} | URL: {session_info['url']} | Logs: {session_info['log_dir']} (Ctrl+C to stop)", end="")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        if session_info['process']:
            kill_process_and_children(session_info['process'].pid)
        ngrok.kill()

# Main function to run the script
def main():
    try:
        parser = argparse.ArgumentParser(description='Run INSPECT view with ngrok tunnel')
        parser.add_argument('--token', help='Ngrok authentication token')
        parser.add_argument('--log-dir', default='/content/logs',
                            help='Directory containing log files')
        args = parser.parse_args()

        # Try to get token from environment variable if not provided as argument
        authtoken = args.token or os.environ.get('NGROK_AUTHTOKEN')

        if not authtoken:
            print("❌ Error: NGROK_AUTHTOKEN not found")
            print("Please either:")
            print("1. Set NGROK_AUTHTOKEN environment variable, or")
            print("2. Pass token as argument: --token YOUR_TOKEN")
            return

        session_info = clean_start_with_inspect(authtoken, log_dir=args.log_dir)
        if session_info:
            monitor_inspect_session(session_info)
    except Exception as e:
        print(f"❌ Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
