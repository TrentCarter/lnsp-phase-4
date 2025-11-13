#!/usr/bin/env python3
"""
Root-level shortcut for starting PAS services and common operations.

Usage:
    python start.py                    # Start all PAS services
    python start.py --dashboard        # Start HMI dashboard only
    python start.py --model-pool       # Start Model Pool Manager only
    python start.py --status           # Check service status
    python start.py --stop             # Stop all services
    python start.py --help             # Show this help
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def run_command(cmd, check=True, capture_output=False):
    """Run shell command with proper error handling."""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        else:
            subprocess.run(cmd, shell=True, check=check, cwd=PROJECT_ROOT)
            return None, None, 0
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        return None, None, e.returncode

def check_venv():
    """Check if virtual environment is activated."""
    if not (os.path.exists('.venv') or hasattr(sys, 'real_prefix') or 
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print(f"{Colors.YELLOW}Warning: Virtual environment not detected. Consider activating .venv{Colors.RESET}")
        return False
    return True

def start_all_services():
    """Start all PAS services."""
    print(f"{Colors.GREEN}{Colors.BOLD}🚀 Starting all PAS services...{Colors.RESET}")
    
    if not check_venv():
        return
    
    script_path = PROJECT_ROOT / "scripts" / "start_all_pas_services.sh"
    if script_path.exists():
        print(f"{Colors.BLUE}Running: {script_path}{Colors.RESET}")
        run_command(f"bash {script_path}")
    else:
        print(f"{Colors.RED}Error: {script_path} not found{Colors.RESET}")

def start_dashboard_only():
    """Start HMI dashboard only."""
    print(f"{Colors.GREEN}{Colors.BOLD}🖥️  Starting HMI dashboard...{Colors.RESET}")
    
    if not check_venv():
        return
    
    script_path = PROJECT_ROOT / "scripts" / "start_hmi_server.sh"
    if script_path.exists():
        print(f"{Colors.BLUE}Running: {script_path}{Colors.RESET}")
        run_command(f"bash {script_path}")
    else:
        print(f"{Colors.RED}Error: {script_path} not found{Colors.RESET}")

def start_model_pool_only():
    """Start Model Pool Manager only."""
    print(f"{Colors.GREEN}{Colors.BOLD}🤖 Starting Model Pool Manager...{Colors.RESET}")
    
    if not check_venv():
        return
    
    cmd = "./.venv/bin/python -m uvicorn services.model_pool_manager.model_pool_manager:app --host 127.0.0.1 --port 8050"
    print(f"{Colors.BLUE}Running: {cmd}{Colors.RESET}")
    run_command(cmd)

def check_status():
    """Check service status."""
    print(f"{Colors.GREEN}{Colors.BOLD}📊 Checking service status...{Colors.RESET}")
    
    services = [
        ("Registry", "6121"),
        ("Heartbeat Monitor", "6109"),
        ("Resource Manager", "6104"),
        ("Token Governor", "6105"),
        ("Model Pool Manager", "8050"),
        ("HMI Dashboard", "6101"),
    ]
    
    for name, port in services:
        stdout, stderr, code = run_command(f"lsof -ti:{port}", capture_output=True)
        if code == 0 and stdout:
            print(f"{Colors.GREEN}✅ {name}: Running on port {port}{Colors.RESET}")
        else:
            print(f"{Colors.RED}❌ {name}: Not running on port {port}{Colors.RESET}")

def stop_all_services():
    """Stop all services."""
    print(f"{Colors.RED}{Colors.BOLD}🛑 Stopping all services...{Colors.RESET}")
    
    script_path = PROJECT_ROOT / "scripts" / "stop_all_pas_services.sh"
    if script_path.exists():
        print(f"{Colors.BLUE}Running: {script_path}{Colors.RESET}")
        run_command(f"bash {script_path}")
    else:
        print(f"{Colors.YELLOW}Using manual stop...{Colors.RESET}")
        ports = ["6121", "6109", "6104", "6105", "8050", "6101"]
        for port in ports:
            run_command(f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true", check=False)
        print(f"{Colors.GREEN}Services stopped{Colors.RESET}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PAS Services Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                    # Start all services
  python start.py --dashboard        # Start dashboard only
  python start.py --status           # Check what's running
  python start.py --stop             # Stop everything
        """
    )
    
    parser.add_argument(
        "--dashboard", 
        action="store_true", 
        help="Start HMI dashboard only"
    )
    
    parser.add_argument(
        "--model-pool", 
        action="store_true", 
        help="Start Model Pool Manager only"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Check service status"
    )
    
    parser.add_argument(
        "--stop", 
        action="store_true", 
        help="Stop all services"
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"{Colors.BLUE}{Colors.BOLD}PAS Services Manager{Colors.RESET}")
    print(f"{Colors.BLUE}Project: {PROJECT_ROOT}{Colors.RESET}")
    print()
    
    # Handle arguments
    if args.dashboard:
        start_dashboard_only()
    elif args.model_pool:
        start_model_pool_only()
    elif args.status:
        check_status()
    elif args.stop:
        stop_all_services()
    else:
        # Default: start all services
        start_all_services()

if __name__ == "__main__":
    main()
