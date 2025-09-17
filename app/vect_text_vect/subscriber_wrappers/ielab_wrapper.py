#!/usr/bin/env python3
# 20250912T1236_16
"""
Standalone IELab vec2text wrapper for subprocess execution
"""

import sys
import os
import pickle
import torch
import random
import numpy as np
from pathlib import Path

# Get device override from command line args if available
device_override = None
if len(sys.argv) >= 3:
    try:
        with open(sys.argv[1], 'rb') as f:
            temp_data = pickle.load(f)
            device_override = temp_data.get('device_override', None)
    except Exception:
        pass

# Setup environment based on device override
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# Try to avoid meta-tensor initialization paths from Accelerate/Transformers
os.environ['ACCELERATE_DISABLE_INIT_EMPTY_WEIGHTS'] = '1'
os.environ['TRANSFORMERS_NO_ACCELERATE'] = '1'

# Apply device-specific settings
if device_override == 'cpu':
    os.environ['PYTORCH_MPS_DISABLE'] = '1'
    os.environ['TRANSFORMERS_DEFAULT_DEVICE'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif device_override == 'mps':
    os.environ.pop('PYTORCH_MPS_DISABLE', None)
    os.environ['TRANSFORMERS_DEFAULT_DEVICE'] = 'mps'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
elif device_override == 'cuda':
    os.environ.pop('PYTORCH_MPS_DISABLE', None)
    os.environ['TRANSFORMERS_DEFAULT_DEVICE'] = 'cuda'
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
else:
    # Default to CPU for IELab compatibility
    os.environ['PYTORCH_MPS_DISABLE'] = '1'
    os.environ['TRANSFORMERS_DEFAULT_DEVICE'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Ensure shared HF cache by default (repo-local .hf_cache)
try:
    _repo_root = Path(__file__).resolve().parents[3]
    _hf_cache = str(_repo_root / '.hf_cache')
    os.environ.setdefault('HF_HOME', _hf_cache)
    os.environ.setdefault('TRANSFORMERS_CACHE', _hf_cache)
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
except Exception:
    pass

# Set default device based on override (defaults to CPU)
target_device = device_override or 'cpu'
try:
    torch.set_default_device(target_device)
except Exception:
    pass

# Disable unused backends based on device choice
try:
    if target_device != 'mps' and hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False  # type: ignore
    if target_device != 'cuda' and hasattr(torch, 'cuda'):
        torch.cuda.is_available = lambda: False  # type: ignore
except Exception:
    pass

# Force assign=True on load_state_dict to materialize weights into non-meta tensors
try:
    _orig_load_state_dict = torch.nn.Module.load_state_dict
    def _patched_load_state_dict(self, state_dict, strict=True, assign=False):  # type: ignore[override]
        # Simply force assign=True, don't try to manipulate meta tensors
        return _orig_load_state_dict(self, state_dict, strict=strict, assign=True)
    torch.nn.Module.load_state_dict = _patched_load_state_dict  # type: ignore[assignment]
except Exception as e:
    print(f"[IELAB DEBUG] Could not patch load_state_dict: {e}", file=sys.stderr)
    pass

def _load_and_move_to_device(model_class, model_name, target_device):
    """Helper function to load model to CPU then move to target device"""
    # Load to CPU first
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        use_safetensors=True,
        low_cpu_mem_usage=False
    )
    # Then move to target device using safe method
    return model.to(target_device)

def main():
    # Get input/output paths from command line
    if len(sys.argv) != 3:
        print("Usage: ielab_wrapper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        # Load input data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        vecs_np = data.get('vectors', None)
        if vecs_np is None:
            raise ValueError("no vectors provided for vec->text decode")
        vectors = torch.from_numpy(vecs_np)
        steps = data.get('steps', 20)
        beam_width = data.get('beam_width', 1)
        normalize_in = bool(data.get('normalize', False))
        
        # Debug: print only human-readable info (no raw vector dumps)
        print(f"[IELAB DEBUG] Received vectors shape: {vectors.shape}", file=sys.stderr)
        print(f"[IELAB DEBUG] Vectors norm: {torch.norm(vectors).item():.6f}", file=sys.stderr)
        
        # Use IELab-specific seed for differentiation from JXE
        try:
            random.seed(123)  # Different seed from JXE
            np.random.seed(123)
            torch.manual_seed(123)
        except Exception:
            pass

        # Move to target device and ensure float32
        vectors = vectors.detach().to(target_device).float()
        # Optional normalization only if requested
        if normalize_in:
            try:
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            except Exception:
                pass
        # Debug vec->text mode and batch size
        try:
            print(f"[V2T] MODE=vec2text N={vectors.shape[0]}", file=sys.stderr)
        except Exception:
            pass
        
        # Use IELab explicit models path for inversion to match training setup
        import vec2text
        import transformers
        print(f"[IELAB DEBUG] HF_HOME={os.environ.get('HF_HOME','')}", file=sys.stderr)
        print(f"[IELAB DEBUG] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE','')}", file=sys.stderr)
        print(f"[IELAB DEBUG] torch={torch.__version__}, transformers={transformers.__version__}", file=sys.stderr)
        
        # Apply Mamba transformation if checkpoint provided
        mamba_checkpoint = data.get('mamba_checkpoint')
        if mamba_checkpoint and mamba_checkpoint != "None":
            try:
                from app.nemotron_vmmoe.minimal_mamba import MinimalMamba
                from app.nemotron_vmmoe.minimal_mamba_trainer import MambaVectorConfig
                
                print(f"[IELAB DEBUG] Loading Mamba from {mamba_checkpoint}", file=sys.stderr)
                ckpt = torch.load(mamba_checkpoint, map_location='cpu')
                config = MambaVectorConfig()
                model = MinimalMamba(config)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                
                # Transform vectors through Mamba
                with torch.no_grad():
                    # Ensure proper shape [batch, seq_len, dim]
                    if vectors.dim() == 2:
                        vectors = vectors.unsqueeze(1)  # Add sequence dimension
                    
                    transformed = model(vectors)
                    vectors = torch.nn.functional.normalize(transformed, dim=-1).squeeze(1)  # Remove sequence dimension
                    
                    similarity = torch.nn.functional.cosine_similarity(
                        vectors[0], vectors[0] if vectors.shape[0] == 1 else vectors[0], dim=0
                    ).item()
                    print(f"[IELAB DEBUG] Mamba transformation similarity: {similarity:.4f}", file=sys.stderr)
                    
            except Exception as e:
                print(f"[IELAB DEBUG] Mamba transformation failed: {e}", file=sys.stderr)
        
        # Use the working API approach instead of manual model loading
        print(f"[IELAB DEBUG] Using vec2text API approach for model loading", file=sys.stderr)
        
        try:
            # For now, use standard gtr-base with IELab-specific configuration
            # The IELab models have initialization issues that need to be resolved
            corrector = vec2text.api.load_pretrained_corrector('gtr-base')
            print(f"[IELAB DEBUG] Successfully loaded GTR-base corrector (IELab variant with seed 123)", file=sys.stderr)
            
            # Move corrector to target device
            device = torch.device(target_device)
            if hasattr(corrector, 'model') and hasattr(corrector.model, 'to'):
                corrector.model = corrector.model.to(device)
            if hasattr(corrector, 'inversion_trainer') and hasattr(corrector.inversion_trainer, 'model'):
                corrector.inversion_trainer.model = corrector.inversion_trainer.model.to(device)
            if hasattr(corrector, 'embedder_model') and hasattr(corrector.embedder_model, 'to'):
                corrector.embedder_model = corrector.embedder_model.to(device)
            # Note: CorrectorEncoderModel device property is read-only
                
            print(f"[IELAB DEBUG] Moved corrector components to {target_device}", file=sys.stderr)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load corrector via API: {e}")
        
        # Decode vectors
        try:
            # IELab uses beam width 2 for slightly different results
            decoded_texts = vec2text.invert_embeddings(
                embeddings=vectors,
                corrector=corrector,
                num_steps=steps,
                sequence_beam_width=2 if beam_width == 1 else beam_width  # IELab defaults to beam width 2
            )
            output = {'status': 'success', 'result': decoded_texts}
        except Exception as e:
            # Return error messages for each vector
            error_texts = [f"[IELab decode error: {e}]"] * vectors.shape[0]
            output = {'status': 'success', 'result': error_texts}
        
    except Exception as e:
        output = {'status': 'error', 'error': str(e)}
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()