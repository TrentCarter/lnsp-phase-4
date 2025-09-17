#!/usr/bin/env python3
# 20250820T195939_1
"""
Standalone VMMoE vec2vec wrapper for subprocess execution
"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path

def main():
    # Get input/output paths from command line
    if len(sys.argv) != 3:
        print("Usage: vmmoe_wrapper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    
    try:
        # Load input data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        vectors = torch.from_numpy(data['vectors'])
        checkpoint_path = data.get('checkpoint_path', 
                                   "output/vmmoe_full_parameter_v1p25/best_model.pth")
        normalize_output = data.get('normalize_output', True)
        debug = data.get('debug', False)
        
        # Debug data type
        if debug:
            print(f"[DEBUG] Input data type: {type(data['vectors'])}")
            print(f"[DEBUG] Input data shape: {data['vectors'].shape if hasattr(data['vectors'], 'shape') else 'no shape'}")
        
        # Setup device management - VMMoE can use MPS
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Setup device - use MPS if available, fallback to CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        if debug:
            print(f"[DEBUG] VMMoE using device: {device}")
            print(f"[DEBUG] Checkpoint: {checkpoint_path}")
        
        # Import VMMoE model loader
        from app.vmmoe.models.factory import load_vmmoe_for_inference
        
        # Load model with validation
        print(f"[DEBUG] Loading VMMoE model...")
        
        loading_error = None
        
        try:
            # First attempt: Load model normally
            model = load_vmmoe_for_inference(
                checkpoint_path=checkpoint_path,
                device=str(device)
            )
            
            # Check if the model parameters look like they're from the expected checkpoint
            total_params = sum(p.numel() for p in model.parameters())
            
            # Expected param count for V1.11 model with LoRA rank 8: ~8-10M parameters
            # If we see 23M+ parameters, it's likely the untrained fallback model
            if total_params > 20_000_000:
                loading_error = "LoRA rank mismatch detected - attempting config-aware reload"
                if debug:
                    print(f"[DEBUG] Parameter count mismatch: {total_params:,} (expected <10M)")
                
                # Second attempt: Detect ACTUAL LoRA rank from checkpoint weights, not config
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    model_state_dict = checkpoint.get('model_state_dict', {})
                    
                    # Detect actual LoRA rank from weight shapes
                    actual_lora_rank = None
                    for key, tensor in model_state_dict.items():
                        if 'lora_A.weight' in key:
                            actual_lora_rank = tensor.shape[0]  # LoRA rank is first dimension of lora_A
                            break
                    
                    if actual_lora_rank:
                        if debug:
                            print(f"[DEBUG] Detected ACTUAL LoRA rank from weights: {actual_lora_rank}")
                        
                        # Get base config from project_config but override LoRA rank with actual
                        if 'project_config' in checkpoint:
                            full_project_config = checkpoint['project_config']
                            training_config = full_project_config.get('training', {})
                            architecture = training_config.get('architecture', {})
                            peft_config = architecture.get('peft_config', {}).copy()
                            
                            # Override with actual trained LoRA rank
                            if 'lora_config' in peft_config:
                                peft_config['lora_config'] = peft_config['lora_config'].copy()
                                peft_config['lora_config']['r'] = actual_lora_rank
                                
                                # Create model with corrected LoRA rank
                                from app.vmmoe.models.factory import create_vmmoe_from_config
                                
                                model_config = {
                                    'mamba_config': architecture.get('mamba_config', {}),
                                    'positional_config': architecture.get('positional_config', {}),
                                    'peft_config': peft_config,
                                    'normalization_config': training_config.get('loss_config', {}).get('normalization_enforcement', {})
                                }
                                
                                if debug:
                                    print(f"[DEBUG] Creating model with corrected LoRA rank: {actual_lora_rank}")
                                
                                model = create_vmmoe_from_config(model_config, str(device))
                                model.load_state_dict(model_state_dict, strict=False)
                                
                                config_rank = full_project_config['training']['architecture']['peft_config']['lora_config'].get('r', 'unknown')
                                loading_error = f"Fixed: Config rank {config_rank} → actual rank {actual_lora_rank} from weights"
                                if debug:
                                    print(f"[DEBUG] Successfully loaded with corrected LoRA rank {actual_lora_rank}")
                            else:
                                loading_error = "No lora_config found in peft_config"
                        else:
                            loading_error = "No project_config found for base configuration"
                    else:
                        loading_error = "Could not detect LoRA rank from checkpoint weights"
                        
                except Exception as reload_error:
                    loading_error = f"Weight-based config reload failed: {reload_error}"
            
            print(f"[DEBUG] VMMoE model loaded {'with warnings' if loading_error else 'successfully'}")
            
        except Exception as e:
            loading_error = f"Model loading failed: {str(e)}"
            model = None
        
        if model is not None:
            # Ensure model is on the correct device
            model = model.to(device)
            model.eval()
            
            # Ensure vectors are on the correct device and proper sequence format
            # VMMoE expects [batch_size, sequence_length, embedding_dim] and uses x[:, 0, :] 
            if vectors.dim() == 2:
                vectors = vectors.unsqueeze(1)  # [B, D] -> [B, 1, D] for sequence model
            
            # Use model's actual device to avoid any ambiguity (e.g., different MPS instances)
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = device
            vectors = vectors.to(model_device)
            
            if debug:
                print(f"[DEBUG] Input vectors shape: {vectors.shape}, device: {vectors.device}")
            
            # Process vectors
            with torch.no_grad():
                if debug:
                    print(f"[DEBUG] About to call model with vectors: shape={vectors.shape}, dtype={vectors.dtype}, device={vectors.device}")
                
                try:
                    output = model(vectors)
                    if debug:
                        print(f"[DEBUG] Model forward pass successful")
                except Exception as forward_error:
                    if debug:
                        print(f"[DEBUG] Model forward pass failed: {forward_error}")
                    raise forward_error
                
                # Handle tuple output
                if isinstance(output, tuple):
                    output = output[0]
                
                # Squeeze time dimension
                if output.dim() == 3:
                    output = output.squeeze(1)  # [B, 768]
                
                # Normalize if requested
                if normalize_output:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
            
            if debug:
                print(f"[DEBUG] About to convert to numpy: shape={output.shape}, dtype={output.dtype}, device={output.device}")
            
            # Convert to numpy for serialization
            try:
                output_np = output.cpu().numpy()
                if debug:
                    print(f"[DEBUG] Numpy conversion successful: shape={output_np.shape}, dtype={output_np.dtype}")
            except Exception as np_error:
                if debug:
                    print(f"[DEBUG] Numpy conversion failed: {np_error}")
                raise np_error
            
            # Include loading error in metadata if present
            if loading_error:
                output = {
                    'status': 'success_with_warning', 
                    'result': output_np,
                    'loading_error': loading_error
                }
            else:
                output = {'status': 'success', 'result': output_np}
        else:
            # Model failed to load completely
            output = {'status': 'error', 'error': loading_error or "Failed to load VMMoE model"}
        
    except Exception as e:
        output = {'status': 'error', 'error': str(e)}
        if debug:
            print(f"[DEBUG] Exception caught: {e}")
    
    try:
        if debug:
            print(f"[DEBUG] About to pickle dump result: {type(output)}")
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        if debug:
            print(f"[DEBUG] Pickle dump successful")
    except Exception as pickle_error:
        if debug:
            print(f"[DEBUG] Pickle dump failed: {pickle_error}")
        # Fallback: dump error message only
        error_output = {'status': 'error', 'error': f'Serialization failed: {pickle_error}'}
        with open(output_path, 'wb') as f:
            pickle.dump(error_output, f)

if __name__ == '__main__':
    main()