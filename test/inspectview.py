#!/usr/bin/env python3
"""
INSPECT Framework Monitoring and Tunneling Script

This script sets up and monitors an INSPECT evaluation interface with a public URL using ngrok tunneling.
It provides a clean way to start, monitor, and manage INSPECT view sessions in Google Colab,
making the evaluation interface accessible from anywhere.

Features:
   - Clean process management (kills existing inspect/ngrok processes)
   - Ngrok tunnel setup for public access
   - Active initialization monitoring
   - Runtime monitoring with elapsed time
   - Graceful shutdown handling
   - Error handling and cleanup
   - Custom log directory support

Requirements:
   - Google Colab environment
   - Ngrok authtoken (stored in Colab secrets)
   - Required packages: pyngrok, flask, psutil

Usage:
   1. Add NGROK_AUTHTOKEN to Colab secrets
   2. Run: !python inspectview.py [--log-dir /path/to/logs]
   3. Access the provided public URL
   4. Use Ctrl+C or interrupt kernel to stop
"""

import subprocess
import sys
import argparse
import os
import socket

def install_requirements():
    """Install required packages using pip."""
    packages = ['pyngrok', 'flask', 'psutil']
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install required packages
install_requirements()

# Import required libraries
from pyngrok import ngrok
import time
import signal
import psutil
from google.colab import userdata

def kill_process_and_children(proc_pid):
    """
    Terminate a process and all its child processes.
    """
    try:
        parent = psutil.Process(proc_pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def wait_for_inspect_ready(port, timeout=120):
    """
    Wait for inspect view to be ready by checking if the port is listening.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                print("\n✅ Inspect view is ready!")
                return True
        except Exception as e:
            print(f"\nSocket error: {str(e)}")
        time.sleep(1)
        print(f"\rWaiting for inspect view to initialize... {int(time.time() - start_time)}s", end="")
    print("\n❌ Timeout waiting for inspect view to initialize")
    return False

def clean_start_with_inspect(authtoken, port=7575, log_level="info", log_dir="/content/anai/logs"):
    """
    Initialize a fresh INSPECT view session with ngrok tunneling.
    
    Args:
        authtoken (str): Ngrok authentication token
        port (int): Port to run inspect view on (default: 7575)
        log_level (str): Log level (default: info)
        log_dir (str): Directory containing log files (default: /content/anai/logs)
    """
    try:
        # Clean up any existing processes
        print("🧹 Cleaning up old processes...")
        
        # Kill any existing inspect processes
        try:
            inspect_procs = subprocess.check_output(["pgrep", "inspect"]).decode().split()
            for pid in inspect_procs:
                kill_process_and_children(int(pid))
        except subprocess.CalledProcessError:
            pass
            
        # Kill any existing ngrok processes
        try:
            ngrok_procs = subprocess.check_output(["pgrep", "ngrok"]).decode().split()
            for pid in ngrok_procs:
                kill_process_and_children(int(pid))
        except subprocess.CalledProcessError:
            pass
            
        time.sleep(2)
        
        # Verify log directory exists
        if not os.path.exists(log_dir):
            print(f"⚠️  Warning: Log directory {log_dir} does not exist!")
            return None
            
        # Set authtoken
        print("🔑 Setting up ngrok authtoken...")
        ngrok.set_auth_token(authtoken)
        
        # Start inspect view with specified port
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
        
        # Check immediate failures
        time.sleep(2)
        if inspect_process.poll() is not None:
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"❌ Inspect process failed immediately!")
            print(f"STDERR: {stderr}")
            print(f"STDOUT: {stdout}")
            raise Exception("Inspect failed to start")
            
        # Wait for service to be ready
        if not wait_for_inspect_ready(port, timeout=120):
            stderr = inspect_process.stderr.read()
            stdout = inspect_process.stdout.read()
            print(f"\nProcess STDERR: {stderr}")
            print(f"Process STDOUT: {stdout}")
            raise Exception("Inspect failed to initialize within timeout period")
        
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

def monitor_inspect_session(session_info):
    """
    Monitor a running INSPECT session and display runtime statistics.
    """
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

def main():
    """Main function to run the script."""
    try:
        parser = argparse.ArgumentParser(description='Run INSPECT view with ngrok tunnel')
        parser.add_argument('--token', help='Ngrok authentication token')
        parser.add_argument('--log-dir', default='/content/anai/logs', 
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
