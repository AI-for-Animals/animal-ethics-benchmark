"""
INSPECT Framework Monitoring and Tunneling Script

This script sets up and monitors an INSPECT evaluation interface with a public URL using ngrok tunneling.
It provides a clean way to start, monitor, and manage INSPECT view sessions in Google Colab,
making the evaluation interface accessible from anywhere.

Features:
   - Clean process management (kills existing inspect/ngrok processes)
   - Ngrok tunnel setup for public access
   - Runtime monitoring with elapsed time
   - Graceful shutdown handling
   - Error handling and cleanup

Requirements:
   - Google Colab environment
   - Ngrok authtoken (stored in Colab secrets)
   - Required packages: pyngrok, flask, psutil

Usage:
   1. Add NGROK_AUTHTOKEN to Colab secrets
   2. Run the script
   3. Access the provided public URL
   4. Use Ctrl+C or interrupt kernel to stop

Author: Arturs Kanepajs
Date: 2024-11-02
"""

import subprocess
import sys
import argparse
import os

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
    
    Args:
        proc_pid (int): Process ID to terminate
        
    Notes:
        Uses psutil to ensure complete process tree cleanup
    """
    try:
        parent = psutil.Process(proc_pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def clean_start_with_inspect(authtoken, port=7575):
    """
    Initialize a fresh INSPECT view session with ngrok tunneling.
    
    Args:
        authtoken (str): Ngrok authentication token
        port (int): Port to run inspect view on (default: 7575 per docs)
        
    Returns:
        dict: Session information containing URL, process info, and port
              Returns None if setup fails
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
        
        # Set authtoken
        print("🔑 Setting up ngrok authtoken...")
        ngrok.set_auth_token(authtoken)
        
        # Start inspect view with specified port
        print(f"🔍 Starting inspect view on port {port}...")
        inspect_process = subprocess.Popen(
            ["inspect", "view", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        print("⏳ Giving inspect view time to initialize...")
        time.sleep(5)
        
        if inspect_process.poll() is not None:
            stderr = inspect_process.stderr.read().decode()
            raise Exception(f"Inspect view failed to start: {stderr}")
        
        print("🚀 Starting ngrok tunnel...")
        tunnel = ngrok.connect(port)
        
        print(f"\n✅ Setup complete!")
        print(f"📡 Public URL: {tunnel.public_url}")
        print("\nImportant Notes:")
        print("1. The inspect interface may take a few minutes to fully initialize")
        print("2. Keep this cell running to maintain the tunnel")
        print("3. If you need to stop, use the interrupt kernel button (■)")
        print(f"4. Inspect view is running on local port {port}")
        
        return {
            'url': tunnel.public_url,
            'process': inspect_process,
            'port': port
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if 'inspect_process' in locals():
            kill_process_and_children(inspect_process.pid)
        return None

def monitor_inspect_session(session_info):
    """
    Monitor a running INSPECT session and display runtime statistics.
    
    Args:
        session_info (dict): Dictionary containing session information
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
            
            print(f"\rRunning for: {hours:02d}:{minutes:02d}:{seconds:02d} | URL: {session_info['url']} (Ctrl+C to stop)", end="")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    finally:
        if session_info['process']:
            kill_process_and_children(session_info['process'].pid)
        ngrok.kill()

def main():
    """Main function to run the script."""
    # Try to get token from environment variable first
    authtoken = os.environ.get('NGROK_AUTHTOKEN')
    
    # If not in environment, try command line argument
    if not authtoken:
        parser = argparse.ArgumentParser(description='Run INSPECT view with ngrok tunnel')
        parser.add_argument('--token', help='Ngrok authentication token')
        args = parser.parse_args()
        authtoken = args.token

    if not authtoken:
        print("❌ Error: NGROK_AUTHTOKEN not found")
        print("Please either:")
        print("1. Set NGROK_AUTHTOKEN environment variable, or")
        print("2. Pass token as argument: --token YOUR_TOKEN")
        return

    session_info = clean_start_with_inspect(authtoken)
    if session_info:
        monitor_inspect_session(session_info)

if __name__ == "__main__":
    main()
