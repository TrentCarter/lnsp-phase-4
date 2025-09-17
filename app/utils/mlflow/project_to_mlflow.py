# 20250722T161419_v1.0
"""
MLflow Metadata Converter for Latent Neurolese Project

This module converts Project_*.json files to MLflow-compatible metadata format
and handles embedding MLflow data into checkpoint files.

Author: Cascade AI Assistant
Date: 2025-07-22
Version: 1.0
"""

import json
import os
import time
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pytz

from app.utils.config_loader import get_file_path


class MLflowMetadataVersion:
    """MLflow metadata schema version management"""
    CURRENT_VERSION = "1.0"
    SUPPORTED_VERSIONS = ["1.0"]


def generate_run_id() -> str:
    """Generate a unique MLflow run ID using UUID4"""
    return str(uuid.uuid4())


def _get_safe_runtime_sn() -> str:
    """Get runtime SN safely without raising exceptions"""
    try:
        from app.runtime_sn import get_runtime_sn
        return get_runtime_sn()
    except Exception:
        # Fallback to SN000000 if runtime SN not available
        return 'SN000000'


def extract_checkpoint_architecture_info(checkpoint_path: str) -> Dict[str, Any]:
    """Extract architectural information from checkpoint tensors
    
    Args:
        checkpoint_path: Path to checkpoint .pth file
        
    Returns:
        Dictionary with architectural dimensions extracted from tensors
    """
    arch_info = {}
    
    try:
        import torch
        if not os.path.exists(checkpoint_path):
            return arch_info
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', {})
        
        # Extract dimensions from key layers
        layer_dims = {}
        
        # Input layer dimension (first compression layer)
        if 'layers.compress_1.weight' in state_dict:
            tensor = state_dict['layers.compress_1.weight']
            layer_dims['actual_input_dim'] = tensor.shape[1]  # Input dimension
            layer_dims['compress_1_output_dim'] = tensor.shape[0]  # Output dimension
            
        # Second compression layer
        if 'layers.compress_2.weight' in state_dict:
            tensor = state_dict['layers.compress_2.weight']
            layer_dims['compress_2_input_dim'] = tensor.shape[1]
            layer_dims['compress_2_output_dim'] = tensor.shape[0]
            
        # Nuclear compression (bottleneck)
        if 'layers.nuclear_compress.weight' in state_dict:
            tensor = state_dict['layers.nuclear_compress.weight']
            layer_dims['nuclear_compress_input_dim'] = tensor.shape[1]
            layer_dims['bottleneck_dim'] = tensor.shape[0]
            
        # Nuclear expansion
        if 'layers.nuclear_expand.weight' in state_dict:
            tensor = state_dict['layers.nuclear_expand.weight']
            layer_dims['nuclear_expand_input_dim'] = tensor.shape[1]
            layer_dims['nuclear_expand_output_dim'] = tensor.shape[0]
            
        # Teacher alignment layer
        if 'layers.teacher_align.weight' in state_dict:
            tensor = state_dict['layers.teacher_align.weight']
            layer_dims['teacher_align_input_dim'] = tensor.shape[1]
            layer_dims['actual_teacher_dim'] = tensor.shape[0]
            
        # Expansion layers
        if 'layers.expand_2.weight' in state_dict:
            tensor = state_dict['layers.expand_2.weight']
            layer_dims['expand_2_input_dim'] = tensor.shape[1]
            layer_dims['expand_2_output_dim'] = tensor.shape[0]
            
        # Multi-head attention dimensions
        if 'layers.multi_head_attention.in_proj_weight' in state_dict:
            tensor = state_dict['layers.multi_head_attention.in_proj_weight']
            layer_dims['attention_input_dim'] = tensor.shape[1]
            layer_dims['attention_proj_dim'] = tensor.shape[0]
        elif 'layers.multi_head_attention.attention.in_proj_weight' in state_dict:
            tensor = state_dict['layers.multi_head_attention.attention.in_proj_weight']
            layer_dims['attention_input_dim'] = tensor.shape[1]
            layer_dims['attention_proj_dim'] = tensor.shape[0]
            
        # Create architecture flow from actual dimensions
        if 'actual_input_dim' in layer_dims and 'bottleneck_dim' in layer_dims and 'actual_teacher_dim' in layer_dims:
            arch_info['actual_architecture_flow'] = f"{layer_dims['actual_input_dim']}→{layer_dims.get('compress_1_output_dim', '?')}→{layer_dims.get('compress_2_output_dim', '?')}→{layer_dims['bottleneck_dim']}→{layer_dims.get('nuclear_expand_output_dim', '?')}→{layer_dims.get('expand_2_output_dim', '?')}→{layer_dims['actual_teacher_dim']}"
            
        # Add all extracted dimensions to arch_info
        arch_info.update(layer_dims)
        
        # Calculate total parameters
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        arch_info['total_parameters'] = total_params
        arch_info['total_parameters_mb'] = round(total_params * 4 / (1024 * 1024), 2)  # Assuming float32
        
        # Model complexity metrics
        if 'actual_input_dim' in layer_dims and 'bottleneck_dim' in layer_dims:
            compression_ratio = layer_dims['actual_input_dim'] / layer_dims['bottleneck_dim']
            arch_info['compression_ratio'] = round(compression_ratio, 2)
            
    except Exception as e:
        print(f"Warning: Could not extract checkpoint architecture info: {e}")
        
    return arch_info


def convert_project_to_mlflow(project_json_path: str, run_id: Optional[str] = None, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert Project_*.json to MLflow metadata format
    
    Args:
        project_json_path: Path to Project_*.json file
        run_id: Optional MLflow run ID
        checkpoint_path: Optional path to model checkpoint for architecture extraction
        
    Returns:
        MLflow metadata dictionary
    """
    try:
        # Load project configuration
        with open(project_json_path, 'r') as f:
            config = json.load(f)
            
        # Extract project info
        project_info = config.get('project', {})
        metadata = project_info.get('metadata', {})
        
        # Get base parameters from project config
        base_params = flatten_project_config_to_params(config)
        
        # Extract actual architecture info from checkpoint if available
        checkpoint_arch_info = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_arch_info = extract_checkpoint_architecture_info(checkpoint_path)
            print(f"[MLflow] Extracted {len(checkpoint_arch_info)} architectural parameters from checkpoint")
        
        # Merge parameters (checkpoint info takes precedence for actual dimensions)
        all_params = {**base_params, **checkpoint_arch_info}
        
        # Create MLflow metadata
        mlflow_metadata = {
            'run_id': run_id or f"run_{_get_safe_runtime_sn()}",
            'experiment_name': metadata.get('experiment_name', 'latent_neurolese'),
            'run_name': f"{metadata.get('architecture_type', 'unknown')}_{metadata.get('version', '1.0')}",
            'tags': {
                'architecture_type': metadata.get('architecture_type', 'unknown'),
                'version': metadata.get('version', '1.0'),
                'serial_number': _get_safe_runtime_sn(),
                'project_file': os.path.basename(project_json_path),
                'has_checkpoint_analysis': bool(checkpoint_arch_info)
            },
            'params': all_params,
            'metrics': {},  # Will be populated during training/testing
            'artifacts': []  # Will be populated with model checkpoints
        }
        
        # Add checkpoint-specific tags if available
        if checkpoint_arch_info:
            if 'actual_architecture_flow' in checkpoint_arch_info:
                mlflow_metadata['tags']['actual_architecture'] = checkpoint_arch_info['actual_architecture_flow']
            if 'total_parameters' in checkpoint_arch_info:
                mlflow_metadata['tags']['model_size'] = f"{checkpoint_arch_info['total_parameters_mb']}MB"
        
        return mlflow_metadata
        
    except Exception as e:
        print(f"Error converting project to MLflow format: {e}")
        return {}


def flatten_project_config_to_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert nested project config to flat MLflow params
    
    Args:
        config: Project configuration dictionary
        
    Returns:
        Flattened parameter dictionary suitable for MLflow
    """
    params = {}
    
    try:
        # Architecture parameters
        arch = config.get('training', {}).get('architecture', {})
        params.update({
            'model_type': arch.get('model_type', 'unknown'),
            'input_dim': arch.get('input_dim', 384),
            'student_dim': arch.get('student_dim', 256),
            'teacher_dim': arch.get('teacher_dim', 384)
        })
        
        # Enhanced architectural dimensions
        params.update({
            'architecture_type': config.get('project', {}).get('metadata', {}).get('architecture_type', 'unknown'),
            'pipeline_type': config.get('project', {}).get('metadata', {}).get('pipeline_type', 'unknown')
        })
        
        # Teacher model information
        project_info = config.get('project', {})
        params.update({
            'teacher_model': project_info.get('model', 'unknown'),
            'local_model_path': project_info.get('local_model_path', 'none')
        })
        
        # Extract compression stages and internal dimensions
        compression_stages = arch.get('pyramid_ln_config', {}).get('compression_stages', [])
        if not compression_stages:
            # Fallback to other config locations
            compression_stages = arch.get('hybrid_ln_config', {}).get('compression_stages', [])
        
        if compression_stages:
            # Count different layer types
            linear_layers = [stage for stage in compression_stages if stage.get('type') == 'linear']
            norm_layers = [stage for stage in compression_stages if stage.get('type') == 'layer_norm']
            attention_layers = [stage for stage in compression_stages if stage.get('type') == 'attention']
            
            params.update({
                'total_compression_stages': len(compression_stages),
                'linear_layers_count': len(linear_layers),
                'norm_layers_count': len(norm_layers),
                'attention_layers_count': len(attention_layers)
            })
            
            # Extract key dimensional transitions
            for i, stage in enumerate(compression_stages):
                if stage.get('type') == 'linear' and 'in' in stage and 'out' in stage:
                    params[f'stage_{i:02d}_{stage.get("layer", "unknown")}_in'] = stage['in']
                    params[f'stage_{i:02d}_{stage.get("layer", "unknown")}_out'] = stage['out']
                elif stage.get('type') == 'layer_norm' and 'dim' in stage:
                    params[f'stage_{i:02d}_{stage.get("layer", "unknown")}_dim'] = stage['dim']
                elif stage.get('type') == 'attention' and 'dim' in stage:
                    params[f'stage_{i:02d}_{stage.get("layer", "unknown")}_dim'] = stage['dim']
                    if 'heads' in stage:
                        params[f'stage_{i:02d}_{stage.get("layer", "unknown")}_heads'] = stage['heads']
            
            # Create architecture flow summary
            linear_dims = []
            for stage in linear_layers:
                if 'in' in stage and 'out' in stage:
                    linear_dims.append(f"{stage['in']}→{stage['out']}")
            
            if linear_dims:
                params['architecture_flow'] = ' → '.join(linear_dims)
        
        # Semantic GPS configuration (if present)
        semantic_gps = arch.get('semantic_gps_config', {})
        gps_enabled = semantic_gps.get('enabled', False)
        
        # Core GPS enablement status
        params['semantic_gps_enabled'] = gps_enabled
        
        if semantic_gps and gps_enabled:
            # Core GPS positioning parameters
            params.update({
                'semantic_gps_d_model': semantic_gps.get('d_model', 384),
                'semantic_gps_max_concepts': semantic_gps.get('max_concepts', 50),
                'semantic_gps_n_domains': semantic_gps.get('n_domains', 8),
                'semantic_gps_temperature': semantic_gps.get('temperature', 0.1),
                'semantic_gps_use_dynamic_routing': semantic_gps.get('use_dynamic_routing', True),
                'semantic_gps_topographic_attention': semantic_gps.get('topographic_attention', True),
                'semantic_gps_coordinate_tracking': semantic_gps.get('coordinate_tracking', True)
            })
            
            # 3D Position parameters from compression stages
            compression_stages = arch.get('pyramid_ln_config', {}).get('compression_stages', [])
            if not compression_stages:
                compression_stages = arch.get('hybrid_ln_config', {}).get('compression_stages', [])
            
            # Find GPS positioning layer in compression stages
            gps_layer_found = False
            for i, stage in enumerate(compression_stages):
                if stage.get('type') == 'semantic_gps' or stage.get('layer') == 'semantic_gps_positioning':
                    gps_layer_found = True
                    params.update({
                        'gps_layer_position': i,
                        'gps_layer_dim': stage.get('dim', 384),
                        'gps_layer_dynamic_routing': stage.get('dynamic_routing', True),
                        'gps_layer_type': stage.get('type', 'semantic_gps')
                    })
                    break
            
            params['gps_layer_configured'] = gps_layer_found
            
            # GPS coordinate space analysis
            gps_dim = semantic_gps.get('d_model', 384)
            max_concepts = semantic_gps.get('max_concepts', 50)
            n_domains = semantic_gps.get('n_domains', 16)
            
            params.update({
                'gps_coordinate_space_size': gps_dim * max_concepts,
                'gps_concepts_per_domain': max_concepts // n_domains if n_domains > 0 else max_concepts,
                'gps_domain_density': n_domains / max_concepts if max_concepts > 0 else 0.0,
                'gps_3d_projection_enabled': gps_dim >= 3  # Can project to 3D if dim >= 3
            })
        else:
            # GPS disabled - set default values for tracking
            params.update({
                'semantic_gps_d_model': 0,
                'semantic_gps_max_concepts': 0,
                'semantic_gps_n_domains': 0,
                'semantic_gps_temperature': 0.0,
                'semantic_gps_use_dynamic_routing': False,
                'semantic_gps_topographic_attention': False,
                'semantic_gps_coordinate_tracking': False,
                'gps_layer_configured': False,
                'gps_coordinate_space_size': 0,
                'gps_3d_projection_enabled': False
            })
        
        # Semantic preservation parameters
        semantic_pres = arch.get('semantic_preservation', {})
        params.update({
            'nuclear_diversity_weight': semantic_pres.get('nuclear_diversity_weight', 1.0),
            'alignment_weight': semantic_pres.get('alignment_weight', 0.1),
            'attention_diversity_weight': semantic_pres.get('attention_diversity_weight', 0.15),
            'bottleneck_dim': semantic_pres.get('bottleneck_dim', 128)
        })
        
        # Attention configuration
        attention_config = arch.get('attention_config', {})
        params.update({
            'attention_heads': attention_config.get('num_heads', 8),
            'attention_head_dim': attention_config.get('dim_head', 16),
            'scale_attention': attention_config.get('scale_attention', True)
        })
        
        # Training hyperparameters
        training = config.get('training', {}).get('hyperparameters', {})
        params.update({
            'learning_rate': training.get('learning_rate', 0.001),
            'batch_size': training.get('batch_size', 32),
            'optimizer': training.get('optimizer', 'AdamW'),
            'loss_function': training.get('loss_function', 'triplet_margin')
        })
        
        # Data parameters
        data = config.get('training', {}).get('data', {})
        params.update({
            'epochs': data.get('epochs', 10),
            'samples_per_batch': data.get('sampling', {}).get('samples_per_batch', 1000)
        })
        
        # Dataset weights
        datasets = data.get('datasets', {})
        for dataset_name, weight in datasets.items():
            if isinstance(weight, (int, float)):
                params[f'dataset_weight_{dataset_name}'] = weight
        
        # GPS Loss Configuration (if GPS is enabled)
        if gps_enabled and semantic_gps:
            gps_loss_config = semantic_gps.get('loss_weights', {})
            params.update({
                'gps_loss_clustering_weight': gps_loss_config.get('clustering', 1.0),
                'gps_loss_smoothness_weight': gps_loss_config.get('smoothness', 0.5),
                'gps_loss_separation_weight': gps_loss_config.get('separation', 0.3),
                'gps_loss_efficiency_weight': gps_loss_config.get('efficiency', 0.2),
                'gps_loss_topographic_weight': gps_loss_config.get('topographic_attention', 0.1),
                'gps_loss_total_weight': sum(gps_loss_config.values()) if gps_loss_config else 2.1
            })
            
            # GPS training configuration
            gps_training = semantic_gps.get('training', {})
            params.update({
                'gps_warmup_epochs': gps_training.get('warmup_epochs', 5),
                'gps_loss_schedule': gps_training.get('loss_schedule', 'linear'),
                'gps_coordinate_init': gps_training.get('coordinate_initialization', 'random')
            })
        else:
            # GPS disabled - set loss weights to 0
            params.update({
                'gps_loss_clustering_weight': 0.0,
                'gps_loss_smoothness_weight': 0.0,
                'gps_loss_separation_weight': 0.0,
                'gps_loss_efficiency_weight': 0.0,
                'gps_loss_topographic_weight': 0.0,
                'gps_loss_total_weight': 0.0,
                'gps_warmup_epochs': 0,
                'gps_loss_schedule': 'disabled',
                'gps_coordinate_init': 'none'
            })
        
    except Exception as e:
        print(f"Warning: Error flattening config parameters: {e}")
        # Continue with partial params rather than failing
    
    return params


def extract_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract dataset information for MLflow inputs
    
    Args:
        config: Project configuration dictionary
        
    Returns:
        Dataset information dictionary
    """
    dataset_info = {}
    
    try:
        data_config = config.get('training', {}).get('data', {})
        datasets = data_config.get('datasets', {})
        
        dataset_info = {
            'datasets_used': list(datasets.keys()),
            'dataset_weights': datasets,
            'total_samples': data_config.get('sampling', {}).get('samples_per_batch', 1000),
            'caching_enabled': data_config.get('caching', {}).get('duplet_cache', False)
        }
        
        # Add dataset paths if available
        dataset_paths = data_config.get('dataset_paths', {})
        if dataset_paths:
            dataset_info['dataset_paths'] = dataset_paths
            
    except Exception as e:
        print(f"Warning: Error extracting dataset info: {e}")
        dataset_info = {'datasets_used': [], 'error': str(e)}
    
    return dataset_info


def update_mlflow_metrics(mlflow_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update MLflow metadata with training/testing metrics
    
    Args:
        mlflow_data: Existing MLflow metadata
        metrics: New metrics to add
        
    Returns:
        Updated MLflow metadata
    """
    mlflow_data['data']['metrics'].update(metrics)
    
    # Update timestamp
    current_time = datetime.now(pytz.timezone('US/Eastern'))
    mlflow_data['info']['end_time'] = int(current_time.timestamp() * 1000)
    
    return mlflow_data


def finalize_mlflow_metadata(mlflow_data: Dict[str, Any], status: str = 'FINISHED') -> Dict[str, Any]:
    """
    Finalize MLflow metadata when training/testing is complete
    
    Args:
        mlflow_data: MLflow metadata to finalize
        status: Final status ('FINISHED', 'FAILED', etc.)
        
    Returns:
        Finalized MLflow metadata
    """
    mlflow_data['info']['status'] = status
    
    if not mlflow_data['info']['end_time']:
        current_time = datetime.now(pytz.timezone('US/Eastern'))
        mlflow_data['info']['end_time'] = int(current_time.timestamp() * 1000)
    
    return mlflow_data


def validate_mlflow_metadata(mlflow_data: Dict[str, Any]) -> bool:
    """
    Simple validation of MLflow metadata structure
    
    Args:
        mlflow_data: MLflow metadata to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check required top-level keys
        required_keys = ['mlflow_metadata_version', 'info', 'data', 'inputs']
        if not all(key in mlflow_data for key in required_keys):
            return False
        
        # Check info section
        info_required = ['run_id', 'experiment_id', 'status', 'start_time']
        if not all(key in mlflow_data['info'] for key in info_required):
            return False
        
        # Check data section
        data_required = ['params', 'metrics', 'tags']
        if not all(key in mlflow_data['data'] for key in data_required):
            return False
        
        # Check version compatibility
        version = mlflow_data.get('mlflow_metadata_version')
        if version not in MLflowMetadataVersion.SUPPORTED_VERSIONS:
            return False
        
        return True
        
    except Exception:
        return False


def save_mlflow_metadata_to_file(mlflow_data: Dict[str, Any], project_id: str) -> str:
    """
    Save MLflow metadata to local file using standard naming convention
    
    Args:
        mlflow_data: MLflow metadata to save
        project_id: Project identifier for filename
        
    Returns:
        Path to saved file
    """
    try:
        # Try to get file path using centralized config
        mlflow_file_path = get_file_path('mlflow_metadata', project_id=project_id)
        
    except Exception as e:
        # Fallback to manual path construction if centralized config fails
        print(f"Warning: Centralized config failed ({e}), using fallback path")
        
        # Create fallback path with current timestamp
        from datetime import datetime
        import pytz
        est = pytz.timezone('US/Eastern')
        now = datetime.now(est)
        date = now.strftime('%Y%m%d')
        time = now.strftime('%H%M%S')
        
        # Get SN from MLflow metadata if available
        sn = 'SN000000'  # Default fallback
        try:
            sn = mlflow_data['info']['tags']['ln.serial_number']
        except (KeyError, TypeError):
            pass
        
        # Construct fallback filename
        filename = f"{date}T{time}_{sn}_{project_id}_mlflow.json"
        mlflow_file_path = Path('output/test/mlflow') / filename
    
    try:
        # Ensure directory exists
        mlflow_dir = Path(mlflow_file_path).parent
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(mlflow_file_path, 'w') as f:
            json.dump(mlflow_data, f, indent=2)
        
        print(f"MLflow metadata saved to: {mlflow_file_path}")
        return str(mlflow_file_path)
        
    except Exception as e:
        print(f"Error saving MLflow metadata: {e}")
        raise


if __name__ == "__main__":
    # Test the converter
    test_project = "/Users/trentcarter/Artificial_Intelligence/AI_Projects/latent-neurolese-v1/CascadeProjects/windsurf-project/inputs/projects/Project_V1p4_72225_3-2-192-2-3.json"
    if os.path.exists(test_project):
        mlflow_data = convert_project_to_mlflow(test_project)
        print("MLflow conversion successful!")
        print(f"Run ID: {mlflow_data['info']['run_id']}")
        print(f"Parameters: {len(mlflow_data['data']['params'])}")
        print(f"Validation: {validate_mlflow_metadata(mlflow_data)}")
    else:
        print("Test project file not found")
