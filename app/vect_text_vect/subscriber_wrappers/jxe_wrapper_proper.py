#!/usr/bin/env python3
# 20250913T1019_2
"""
Proper JXE vec2text wrapper using vec2text's own embedding pipeline
"""

import sys
import os
import pickle
import numpy as np
import random
from pathlib import Path


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n < 1e-8] = 1e-8
    return x / n


def _coerce_vectors(payload: dict):
    V = payload.get("vectors", None)
    if V is None:
        return None, "missing 'vectors'"
    if not isinstance(V, np.ndarray):
        try:
            V = np.array(V, dtype=np.float32)
        except Exception as e:
            return None, f"cannot convert to np.float32: {e}"
    if V.ndim != 2 or V.shape[1] != 768:
        return None, f"expected shape [N,768], got {getattr(V, 'shape', None)}"
    if V.dtype != np.float32:
        V = V.astype(np.float32)
    return V, None

def main():
    # Get input/output paths from command line
    if len(sys.argv) != 3:
        print("Usage: jxe_wrapper_proper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        # Load device override first
        device_override = None
        try:
            with open(input_path, 'rb') as f:
                temp_data = pickle.load(f)
                device_override = temp_data.get('device_override', None)
        except Exception:
            pass
        
        # Setup environment for vec2text compatibility - MUST BE BEFORE ANY IMPORTS
        import os
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Apply device override if specified
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
            # Default to CPU
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
        
        # Import torch after setting environment
        import torch
        # Force default device to CPU and disable MPS/CUDA completely
        try:
            torch.set_default_device('cpu')
        except Exception:
            pass
        try:
            if hasattr(torch.backends, 'mps'):
                torch.backends.mps.is_available = lambda: False  # type: ignore
            if hasattr(torch, 'cuda'):
                torch.cuda.is_available = lambda: False  # type: ignore
        except Exception:
            pass
        # Use JXE-specific seed for differentiation from IELab
        try:
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
        except Exception:
            pass
        
        # Load input data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        # Common params
        metadata = data.get('metadata', {})
        steps = data.get('steps', 1)
        debug = data.get('debug', False)
        device_override = data.get('device_override', None)
        
        # Import vec2text
        import vec2text
        import vec2text.api as vapi
        import transformers as _tf
        if debug:
            vver = getattr(vec2text, '__version__', '?')
            print(f"[JXE DEBUG] HF_HOME={os.environ.get('HF_HOME','')}", file=sys.stderr)
            print(f"[JXE DEBUG] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE','')}", file=sys.stderr)
            print(f"[JXE DEBUG] vec2text={vver}, transformers={_tf.__version__}", file=sys.stderr)
        
        # Prefer decoding from provided vectors only (force vec->text)
        vectors_np = data.get('vectors', None)
        if vectors_np is not None:
            V, err = _coerce_vectors({'vectors': vectors_np})
            if err:
                raise ValueError(f"bad vector payload: {err}")
            else:
                # Optional: JXE expects unit vectors
                V = _l2(V)
                N = int(V.shape[0])
                bs = int(data.get('batch_size', 1)) or 1
                out: list[str] = []
                # Use standard JXE approach with GTR-base corrector
                corrector = vapi.load_pretrained_corrector("gtr-base")
                if debug:
                    print("[JXE DEBUG] Loaded standard GTR-base corrector (JXE variant)", file=sys.stderr)
                
                # Force corrector to CPU to avoid device mismatch
                def force_to_cpu(model):
                    """Recursively force model to CPU"""
                    model = model.cpu()
                    for name, module in model.named_modules():
                        try:
                            module = module.cpu()
                        except:
                            pass
                    return model
                
                if hasattr(corrector, 'model'):
                    corrector.model = force_to_cpu(corrector.model)
                if hasattr(corrector, 'inversion_trainer') and hasattr(corrector.inversion_trainer, 'model'):
                    corrector.inversion_trainer.model = force_to_cpu(corrector.inversion_trainer.model)
                if hasattr(corrector, 'embedder_model'):
                    corrector.embedder_model = force_to_cpu(corrector.embedder_model)
                if hasattr(corrector, 'encoder'):
                    corrector.encoder = force_to_cpu(corrector.encoder)
                # Note: CorrectorEncoderModel device property is read-only
                
                if debug:
                    print(f"[V2T] MODE=vec2text N={N}", file=sys.stderr)
                for i in range(0, N, bs):
                    sub = V[i:i+bs]
                    try:
                        tens = torch.from_numpy(sub).detach().cpu().float()
                        texts = vec2text.invert_embeddings(
                            embeddings=tens,
                            corrector=corrector,
                            num_steps=steps,
                            sequence_beam_width=1,  # JXE uses beam width 1
                        )
                        if not isinstance(texts, list) or len(texts) != len(sub):
                            miss = len(sub) - (len(texts) if isinstance(texts, list) else 0)
                            texts = (texts if isinstance(texts, list) else []) + ["<decode_error>"] * miss
                        out.extend(texts)
                    except Exception as e:
                        if debug:
                            print(f"[JXE] batch {i}:{i+len(sub)} error: {e}", file=sys.stderr)
                        out.extend(["<decode_error>"] * len(sub))
                if len(out) != N:
                    if debug:
                        print(f"[JXE] adjusting output len {len(out)} -> {N}", file=sys.stderr)
                    out = (out + ["<decode_error_pad>"] * N)[:N]
                results = out
        else:
            # No vectors provided: do not fall back to text->text
            raise ValueError("no vectors provided for vec->text decode")
        
        if debug:
            print(f"[DEBUG] Vec2text returned {len(results) if results else 0} results")
        
        # Guard against invalid results
        if not isinstance(results, list):
            results = []
        elif len(results) == 0:
            results = []
        
        # Save output
        output = {'status': 'success', 'result': results}
        
    except Exception as e:
        try:
            if debug:
                import traceback
                traceback.print_exc()
        except (NameError, UnboundLocalError):
            import traceback
            traceback.print_exc()
        output = {'status': 'error', 'error': str(e)}
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()