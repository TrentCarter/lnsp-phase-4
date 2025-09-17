# 20250817T121330_v1.1
"""
MLflow Integration Utilities for Latent Neurolese Project

This module handles saving MLflow metadata to multiple destinations:
- Local JSON files
- Checkpoint .pth files
- MLflow tracking server (optional)

Author: Cascade AI Assistant
Date: 2025-07-22
Version: 1.0
"""

import json
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from app.utils.mlflow.project_to_mlflow import (
    validate_mlflow_metadata, 
    save_mlflow_metadata_to_file,
    MLflowMetadataVersion
)

# Optional MLflow server integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed. Server sync disabled. Install with: pip install mlflow")


def save_mlflow_metadata_everywhere(
    mlflow_data: Dict[str, Any], 
    project_id: str,
    checkpoint_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Save MLflow metadata to all destinations
    
    Args:
        mlflow_data: MLflow metadata dictionary
        project_id: Project identifier
        checkpoint_path: Optional path to checkpoint file to embed metadata
        
    Returns:
        Dictionary with paths/status of saves
    """
    results = {}
    
    # Validate metadata first
    if not validate_mlflow_metadata(mlflow_data):
        raise ValueError("Invalid MLflow metadata structure")
    
    # 1. Save to local JSON file
    try:
        local_file_path = save_mlflow_metadata_to_file(mlflow_data, project_id)
        results['local_file'] = local_file_path
        logging.info(f"✅ MLflow metadata saved to local file: {local_file_path}")
    except Exception as e:
        results['local_file'] = f"ERROR: {e}"
        logging.error(f"❌ Failed to save MLflow metadata to local file: {e}")
    
    # 2. Embed in checkpoint file (if provided)
    if checkpoint_path:
        try:
            embed_mlflow_in_checkpoint(checkpoint_path, mlflow_data)
            results['checkpoint'] = checkpoint_path
            logging.info(f"✅ MLflow metadata embedded in checkpoint: {checkpoint_path}")
        except Exception as e:
            results['checkpoint'] = f"ERROR: {e}"
            logging.error(f"❌ Failed to embed MLflow metadata in checkpoint: {e}")
    
    # 3. Sync to MLflow server (optional)
    try:
        server_result = sync_to_mlflow_server(mlflow_data)
        results['mlflow_server'] = server_result
        if server_result.startswith('SUCCESS'):
            logging.info(f"✅ MLflow metadata synced to server: {server_result}")
        else:
            logging.warning(f"⚠️ MLflow server sync: {server_result}")
    except Exception as e:
        results['mlflow_server'] = f"SKIPPED: {e}"
        logging.info(f"ℹ️ MLflow server sync skipped: {e}")
    
    return results


def embed_mlflow_in_checkpoint(checkpoint_path: str, mlflow_data: Dict[str, Any]) -> None:
    """
    Embed MLflow metadata into existing checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        mlflow_data: MLflow metadata to embed
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load existing checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Add MLflow metadata
    checkpoint['mlops_metadata'] = mlflow_data
    
    # Add MLflow signature for model inputs/outputs
    checkpoint['mlflow_signature'] = generate_mlflow_signature(mlflow_data)
    
    # Save updated checkpoint
    torch.save(checkpoint, checkpoint_path)
    
    logging.info(f"MLflow metadata embedded in checkpoint: {checkpoint_path}")


def generate_mlflow_signature(mlflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate MLflow model signature from metadata
    
    Args:
        mlflow_data: MLflow metadata
        
    Returns:
        MLflow signature dictionary
    """
    try:
        params = mlflow_data['data']['params']
        input_dim = params.get('input_dim', 384)
        student_dim = params.get('student_dim', 256)
        
        signature = {
            'inputs': [
                {
                    'name': 'teacher_embedding',
                    'type': 'tensor',
                    'shape': [input_dim],
                    'dtype': 'float32'
                }
            ],
            'outputs': [
                {
                    'name': 'student_embedding', 
                    'type': 'tensor',
                    'shape': [student_dim],
                    'dtype': 'float32'
                }
            ]
        }
        
        return signature
        
    except Exception as e:
        logging.warning(f"Could not generate MLflow signature: {e}")
        return {'inputs': [], 'outputs': []}


def sync_to_mlflow_server(mlflow_data: Dict[str, Any]) -> str:
    """
    Sync MLflow metadata to MLflow tracking server
    
    Args:
        mlflow_data: MLflow metadata to sync
        
    Returns:
        Status string indicating success/failure
    """
    if not MLFLOW_AVAILABLE:
        return "SKIPPED: MLflow not installed"
    
    try:
        # Check if MLflow server is configured
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5007')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Try to connect to server
        try:
            mlflow.get_experiment_by_name("Default")
        except Exception:
            return f"SKIPPED: MLflow server not available at {tracking_uri}"
        
        # Create or get experiment
        experiment_name = f"latent_neurolese_{mlflow_data['info']['experiment_id']}"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            return f"ERROR: Could not create/get experiment: {e}"
        
        # Start MLflow run (create new run, don't reuse run_id)
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=mlflow_data['info']['run_name']
        ) as run:
            # Log parameters
            mlflow.log_params(mlflow_data['data']['params'])
            
            # Log metrics (if any)
            if mlflow_data['data']['metrics']:
                mlflow.log_metrics(mlflow_data['data']['metrics'])
            
            # Set tags
            mlflow.set_tags(mlflow_data['info']['tags'])
            
            return f"SUCCESS: Run {run.info.run_id} logged to {tracking_uri}"
            
    except Exception as e:
        return f"ERROR: {e}"


def extract_mlflow_from_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract MLflow metadata from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        
    Returns:
        MLflow metadata if found, None otherwise
    """
    try:
        if not os.path.exists(checkpoint_path):
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Check for MLflow metadata
        if 'mlops_metadata' in checkpoint:
            mlflow_data = checkpoint['mlops_metadata']
            
            # Validate the extracted metadata
            if validate_mlflow_metadata(mlflow_data):
                return mlflow_data
            else:
                logging.warning(f"Invalid MLflow metadata found in {checkpoint_path}")
                return None
        
        return None
        
    except Exception as e:
        logging.error(f"Error extracting MLflow metadata from {checkpoint_path}: {e}")
        return None


def start_mlflow_server(port: int = 5005, backend_store_uri: str = None, artifact_root: str = None) -> str:
    """
    Start MLflow tracking server locally
    
    Args:
        port: Port to run server on
        backend_store_uri: Optional backend store URI
        artifact_root: Optional artifact root directory
        
    Returns:
        Status message
    """
    if not MLFLOW_AVAILABLE:
        return "ERROR: MLflow not installed"
    
    try:
        import subprocess
        
        # Default backend store in project root
        if not backend_store_uri:
            backend_store_uri = "sqlite:///mlflow.db"
        
        # Default artifact root
        if not artifact_root:
            artifact_root = "output/test/mlflow/artifacts"
        
        # Ensure artifact directory exists
        Path(artifact_root).mkdir(parents=True, exist_ok=True)
        
        # Start server command
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", artifact_root,
            "--host", "0.0.0.0",
            "--port", str(port)
        ]
        
        print(f"Starting MLflow server: {' '.join(cmd)}")
        print(f"Database: {backend_store_uri}")
        print(f"Artifacts: {artifact_root}")
        print(f"Server will be available at: http://localhost:{port}")
        
        # Start server in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return f"SUCCESS: MLflow server starting on port {port} (PID: {process.pid})"
        
    except Exception as e:
        return f"ERROR: Could not start MLflow server: {e}"


def get_mlflow_server_status() -> str:
    """
    Check if MLflow server is running
    
    Returns:
        Status string
    """
    try:
        import requests
        
        tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5007')
        response = requests.get(f"{tracking_uri}/health", timeout=5)
        
        if response.status_code == 200:
            return f"RUNNING: MLflow server active at {tracking_uri}"
        else:
            return f"ERROR: MLflow server returned status {response.status_code}"
            
    except Exception as e:
        return f"NOT_RUNNING: {e}"


if __name__ == "__main__":
    # Test MLflow integration
    print("MLflow Integration Test")
    print(f"MLflow Available: {MLFLOW_AVAILABLE}")
    print(f"Server Status: {get_mlflow_server_status()}")
