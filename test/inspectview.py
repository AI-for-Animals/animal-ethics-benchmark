#!/usr/bin/env python3
"""
INSPECT Framework Monitoring and Tunneling Script
This script sets up and monitors an INSPECT evaluation interface with a public URL using ngrok tunneling.
"""

import subprocess
import sys
import os
import time
import socket
import psutil
import argparse

# Install required packages before proceeding
def install_requirements():
    """Install necessary packages if not already available."""
    packages = ['pyngrok', 'flask', 'psutil']
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_requirements()

# Import after installation
from pyngrok import ngrok

# Helper function to kill a process and its children
def kill_process_and_children(proc_pid):
    """Kill a process and all its child processes."""
    try:
        parent = psutil.Process(proc_pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

# Clean up processes by name or using a specific port
def clean_up_processes(proc_names, port=None):
    """Clean up any existing processes by name or using a specific port."""
    for proc_name in proc_names:
        try:
            procs = subprocess.check_output(["pgrep", proc_name]).decode().split()
            for pid in procs:
                kill_process_and_children(int(pid))
        except subprocess.CalledProcessError:
            pass

    if port:
        try:
            port_processes = subprocess.check_output(["lsof", "-t", "-i:{}".format(port)]).decode().split()
            for pid in port_processes:
                kill_process_and_children(int(pid))
        except subprocess.CalledProcessError:
            pass

# Wait for inspect view to be ready
def wait_for_inspect_ready(port, timeout=120):
    """Wait for inspect view to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    print("\n✅ Inspect view is ready!")
                    return True
        except socket.error:
            pass
        time.sleep(1)
    return False

# Clean start for the inspect session
def clean_start_with_inspect(authtoken, port=7575, log_level="info", log_dir="/content/logs"):
    """Start a new INSPECT session, cleaning up any previous sessions."""
    try:
        # Clean up old processes
        clean_up_processes(["inspect", "ngrok"], port)

        # Ensure the log directory exists
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        # Set ngrok authtoken
        ngrok.set_auth_token(authtoken)

        # Start the inspect view with specified port
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

        # Wait for inspect view to be ready
        if not wait_for_inspect_ready(port):
            kill_process_and_children(inspect_process.pid)
            return None

        # Start ngrok tunnel
        tunnel = ngrok.connect(port)

        print(f"\n✅ Setup complete!\n📡 Public URL: {tunnel.public_url}\n")
        print(f"Inspect view is running on local port {port}")
        return {
            'url': tunnel.public_url,
            'process': inspect_process,
            'port': port,
            'log_dir': log_dir
        }

    except Exception as e:
        print(f"❌ Error during setup: {str(e)}")
        return None

# Monitor the inspect session
def monitor_inspect_session(session_info):
    """Monitor the running INSPECT session."""
    if not session_info:
        return

    start_time = time.time()
    try:
        while True:
            if session_info['process'].poll() is not None:
                print("\n❌ Inspect view process has terminated!")
                break

            elapsed = int(time.time() - start_time)
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"\rRunning for: {hours:02d}:{minutes:02d}:{seconds:02d} | URL: {session_info['url']} (Ctrl+C to stop)", end="")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        if session_info['process']:
            kill_process_and_children(session_info['process'].pid)
        ngrok.kill()

# Main function to run the script
def main():
    parser = argparse.ArgumentParser(description='Run INSPECT view with ngrok tunnel')
    parser.add_argument('--token', help='Ngrok authentication token')
    parser.add_argument('--log-dir', default='/content/logs', help='Directory containing log files')
    args = parser.parse_args()

    authtoken = args.token or os.environ.get('NGROK_AUTHTOKEN')

    if not authtoken:
        print("❌ Error: NGROK_AUTHTOKEN not found. Set it or pass as argument.")
        return

    session_info = clean_start_with_inspect(authtoken, log_dir=args.log_dir)
    if session_info:
        monitor_inspect_session(session_info)

if __name__ == "__main__":
    main()
