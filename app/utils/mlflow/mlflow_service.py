# 20250722T162749_v1.0
"""
MLflow Service Manager for Latent Neurolese Project

This module provides automatic MLflow server management including:
- Autostart functionality
- Service status checking
- Process management
- Health monitoring

Author: Cascade AI Assistant
Date: 2025-07-22
Version: 1.0
"""

import os
import time
import signal
import subprocess
import psutil
import requests
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import atexit

# Optional MLflow server integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLflowServiceManager:
    """Manages MLflow tracking server lifecycle"""
    
    def __init__(self, 
                 port: int = 5005,
                 backend_store_uri: str = "sqlite:///mlflow.db",
                 artifact_root: str = "output/test/mlflow/artifacts",
                 auto_start: bool = True):
        """
        Initialize MLflow service manager
        
        Args:
            port: Port to run MLflow server on
            backend_store_uri: Database URI (now defaults to root directory)
            artifact_root: Artifact storage directory
            auto_start: Whether to automatically start server if not running
        """
        self.port = port
        self.backend_store_uri = backend_store_uri
        self.artifact_root = artifact_root
        self.auto_start = auto_start
        self.process = None
        self.tracking_uri = f"http://localhost:{port}"
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def is_server_running(self) -> bool:
        """Check if MLflow server is running and responsive"""
        try:
            response = requests.get(f"{self.tracking_uri}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def find_mlflow_process(self) -> Optional[psutil.Process]:
        """Find existing MLflow server process"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if (cmdline and 
                        'mlflow' in cmdline and 
                        'server' in cmdline and 
                        str(self.port) in ' '.join(cmdline)):
                        return proc
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Error finding MLflow process: {e}")
        return None
    
    def start_server(self, force_restart: bool = False) -> Dict[str, Any]:
        """
        Start MLflow server
        
        Args:
            force_restart: Kill existing server and start new one
            
        Returns:
            Status dictionary with success/failure info
        """
        if not MLFLOW_AVAILABLE:
            return {
                'success': False,
                'message': 'MLflow not installed. Install with: pip install mlflow',
                'tracking_uri': None
            }
        
        # Check if server is already running
        if self.is_server_running() and not force_restart:
            return {
                'success': True,
                'message': f'MLflow server already running at {self.tracking_uri}',
                'tracking_uri': self.tracking_uri,
                'pid': self.find_mlflow_process().pid if self.find_mlflow_process() else None
            }
        
        # Stop existing server if force restart
        if force_restart:
            self.stop_server()
            time.sleep(2)  # Give it time to shut down
        
        try:
            # Ensure artifact directory exists
            Path(self.artifact_root).mkdir(parents=True, exist_ok=True)
            
            # Build command using venv mlflow binary - use shell approach for better daemon behavior
            import sys
            from pathlib import Path as PathLib
            
            # Get the venv directory from current Python executable
            python_path = PathLib(sys.executable)
            venv_bin_dir = python_path.parent
            mlflow_binary = venv_bin_dir / "mlflow"
            
            # Create shell command that runs in background
            shell_cmd = f"{mlflow_binary} server --backend-store-uri {self.backend_store_uri} --default-artifact-root {self.artifact_root} --host 0.0.0.0 --port {self.port} > /dev/null 2>&1 &"
            
            # Start server using shell to ensure proper background execution
            result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'message': f'Failed to start MLflow server: {result.stderr}',
                    'tracking_uri': None
                }
            
            # Wait longer for MLflow to initialize (first startup takes 20+ seconds)
            print(f"⏳ Waiting for MLflow server to initialize (this may take 30+ seconds)...")
            time.sleep(5)
            
            # Wait for server to be responsive (increased timeout for database initialization)
            for i in range(30):  # Try for 30 seconds
                if self.is_server_running():
                    # Find the actual MLflow process to get PID
                    mlflow_proc = self.find_mlflow_process()
                    pid = mlflow_proc.pid if mlflow_proc else "Unknown"
                    
                    return {
                        'success': True,
                        'message': f'MLflow server started successfully at {self.tracking_uri}',
                        'tracking_uri': self.tracking_uri,
                        'pid': pid,
                        'database': self.backend_store_uri,
                        'artifacts': self.artifact_root
                    }
                if i % 5 == 0 and i > 0:  # Progress indicator every 5 seconds
                    print(f"⏳ Still waiting... ({i}/30 seconds)")
                time.sleep(1)
            
            # Server not responsive after 30 seconds
            return {
                'success': False,
                'message': f'MLflow server failed to become responsive at {self.tracking_uri} after 30 seconds',
                'tracking_uri': self.tracking_uri,
                'pid': None
            }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error starting MLflow server: {e}',
                'tracking_uri': None
            }
    
    def stop_server(self) -> Dict[str, Any]:
        """Stop MLflow server"""
        try:
            # Find and kill existing MLflow process
            existing_process = self.find_mlflow_process()
            if existing_process:
                try:
                    # Try graceful shutdown first
                    existing_process.terminate()
                    existing_process.wait(timeout=5)
                    
                    return {
                        'success': True,
                        'message': f'MLflow server stopped (PID: {existing_process.pid})'
                    }
                except psutil.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    existing_process.kill()
                    return {
                        'success': True,
                        'message': f'MLflow server force-killed (PID: {existing_process.pid})'
                    }
            else:
                return {
                    'success': True,
                    'message': 'No MLflow server process found'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Error stopping MLflow server: {e}'
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive MLflow server status"""
        status = {
            'server_running': self.is_server_running(),
            'tracking_uri': self.tracking_uri,
            'database': self.backend_store_uri,
            'artifacts': self.artifact_root,
            'database_exists': False,
            'artifacts_dir_exists': False,
            'process_info': None
        }
        
        # Check database existence
        if self.backend_store_uri.startswith('sqlite:///'):
            db_path = self.backend_store_uri[10:]  # Remove 'sqlite:///'
            status['database_exists'] = os.path.exists(db_path)
            if status['database_exists']:
                status['database_size'] = os.path.getsize(db_path)
        
        # Check artifacts directory
        status['artifacts_dir_exists'] = os.path.exists(self.artifact_root)
        
        # Get process info
        process = self.find_mlflow_process()
        if process:
            try:
                status['process_info'] = {
                    'pid': process.pid,
                    'memory_mb': round(process.memory_info().rss / 1024 / 1024, 1),
                    'cpu_percent': process.cpu_percent(),
                    'create_time': process.create_time()
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                status['process_info'] = {'error': 'Process info not accessible'}
        
        return status
    
    def ensure_running(self) -> Dict[str, Any]:
        """Ensure MLflow server is running (autostart if enabled)"""
        if self.is_server_running():
            return {
                'success': True,
                'message': 'MLflow server already running',
                'action': 'none'
            }
        
        if self.auto_start:
            result = self.start_server()
            result['action'] = 'started'
            return result
        else:
            return {
                'success': False,
                'message': 'MLflow server not running and auto_start disabled',
                'action': 'none'
            }
    
    def cleanup(self):
        """Cleanup on exit (registered with atexit)"""
        if self.process and self.process.poll() is None:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except Exception:
                pass


# Global service manager instance
_service_manager = None

def get_service_manager(**kwargs) -> MLflowServiceManager:
    """Get or create global MLflow service manager"""
    global _service_manager
    if _service_manager is None:
        _service_manager = MLflowServiceManager(**kwargs)
    return _service_manager

def autostart_mlflow_server(**kwargs) -> Dict[str, Any]:
    """Convenience function to autostart MLflow server"""
    manager = get_service_manager(**kwargs)
    return manager.ensure_running()

def get_mlflow_status() -> Dict[str, Any]:
    """Get MLflow server status"""
    manager = get_service_manager()
    return manager.get_status()

def stop_mlflow_server() -> Dict[str, Any]:
    """Stop MLflow server"""
    manager = get_service_manager()
    return manager.stop_server()


if __name__ == "__main__":
    # Command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description="MLflow Service Manager")
    parser.add_argument("action", choices=["start", "stop", "status", "restart"], 
                       help="Action to perform")
    parser.add_argument("--port", type=int, default=5005, help="Port to run on")
    parser.add_argument("--auto-start", action="store_true", default=True, 
                       help="Enable auto-start")
    
    args = parser.parse_args()
    
    manager = MLflowServiceManager(port=args.port, auto_start=args.auto_start)
    
    if args.action == "start":
        result = manager.start_server()
        print(f"{'✅' if result['success'] else '❌'} {result['message']}")
        if result['success'] and 'tracking_uri' in result:
            print(f"🌐 Access MLflow UI at: {result['tracking_uri']}")
    
    elif args.action == "stop":
        result = manager.stop_server()
        print(f"{'✅' if result['success'] else '❌'} {result['message']}")
    
    elif args.action == "restart":
        result = manager.start_server(force_restart=True)
        print(f"{'✅' if result['success'] else '❌'} {result['message']}")
        if result['success'] and 'tracking_uri' in result:
            print(f"🌐 Access MLflow UI at: {result['tracking_uri']}")
    
    elif args.action == "status":
        status = manager.get_status()
        print(f"MLflow Server Status:")
        print(f"  Running: {'✅' if status['server_running'] else '❌'}")
        print(f"  URL: {status['tracking_uri']}")
        print(f"  Database: {status['database']} ({'✅' if status['database_exists'] else '❌'})")
        print(f"  Artifacts: {status['artifacts']} ({'✅' if status['artifacts_dir_exists'] else '❌'})")
        
        if status['process_info']:
            info = status['process_info']
            if 'error' not in info:
                print(f"  Process: PID {info['pid']}, {info['memory_mb']}MB RAM, {info['cpu_percent']}% CPU")
            else:
                print(f"  Process: {info['error']}")
