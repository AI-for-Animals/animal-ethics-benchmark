#!/usr/bin/env python3
"""
INSPECT Framework Monitoring and Tunneling Script
This script sets up and monitors an INSPECT evaluation interface with a public URL using ngrok tunneling.
"""

import subprocess
import sys
import argparse
import os
import socket
import time
import psutil
from pyngrok import ngrok

# Install required packages
def install_requirements():
    """Install necessary packages if not already available."""
    packages = ['pyngrok', 'flask', 'psutil']
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)

install_requirements()

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
    print("🧹 Cleaning up old processes...")
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
        except Exception as e:
            print(f"\nSocket error: {str(e)}")
        time.sleep(1)
        print(f"\rWaiting for inspect view to initialize... {int(time.time() - start_time)}s", end="")
    print("\n❌ Timeout waiting for inspect view to initialize")
    return False

# Wait for logs to be generated
def wait_for_logs(log_dir, timeout=5):
    """Wait briefly for the logs to be generated in the given directory."""
    log_dir = os.path.abspath(log_dir)
    print(f"🔍 Checking log directory: {log_dir}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        log_files = [f for f in os.listdir(log_dir) if f.endswith(".eval")]
        if log_files:
            print(f"✅ Logs detected: {log_files}")
            return True
        time.sleep(1)

    print("❌ No log files detected, but proceeding assuming they will be generated.")
    return False

# Clean start for the inspect session
def clean_start_with_inspect(authtoken, port=7575, log_level="info", log_dir="/content/logs"):
    """Start a new INSPECT session, cleaning up any previous sessions."""
    try:
        # Clean up old processes
        clean_up_processes(["inspect", "ngrok"], port)

        # Ensure the log directory exists
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            print(f"⚠️ Log directory {log_dir} does not exist. Creating it now...")
            os.makedirs(log_dir, exist_ok=True)

        if not os.access(log_dir, os.R_OK):
            print(f"Adjusting permissions for the log directory: {log_dir}")
            os.chmod(log_dir, 0o755)

        # Wait for logs to be generated before starting the inspect view
        wait_for_logs(log_dir)

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
        if inspect_process.poll() is not None:
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"❌ Inspect process failed immediately!\nSTDERR: {stderr}\nSTDOUT: {stdout}")
            raise Exception("Inspect failed to start")

        # Wait for inspect view to be ready
        if not wait_for_inspect_ready(port, timeout=120):
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"\nProcess STDERR: {stderr}\nProcess STDOUT: {stdout}")
            raise Exception("Inspect failed to initialize within timeout period")

        # Start ngrok tunnel
        print("🚀 Starting ngrok tunnel...")
        tunnel = ngrok.connect(port)

        print(f"\n✅ Setup complete!\n📡 Public URL: {tunnel.public_url}\n")
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
