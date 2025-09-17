#!/usr/bin/env python3
# 20250910T1459_1
"""
Standalone Mamba vec2vec wrapper for subprocess execution
"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from dataclasses import asdict

def main():
    if len(sys.argv) != 3:
        print("Usage: mamba_wrapper.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        vectors = torch.from_numpy(data['vectors'])
        checkpoint_path = data.get('checkpoint_path')
        debug = data.get('debug', False)

        if not checkpoint_path:
            raise ValueError("Mamba checkpoint path not provided.")

        from app.nemotron_vmmoe.minimal_mamba import MinimalMamba
        from app.nemotron_vmmoe.minimal_mamba_trainer import MambaVectorConfig

        def device_select() -> torch.device:
            return torch.device("cpu")  # Force CPU to avoid device issues

        device = device_select()
        if debug:
            print(f"[DEBUG] Mamba using device: {device}")
            print(f"[DEBUG] Checkpoint: {checkpoint_path}")

        # Load Model
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        config = MambaVectorConfig()
        model = MinimalMamba(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

        # Transform vectors through Mamba
        with torch.no_grad():
            # Ensure proper input shape
            if vectors.dim() == 1:
                vectors = vectors.unsqueeze(0)  # [768] -> [1, 768]
            
            # Add sequence dimension for Mamba: [B, D] -> [B, 1, D]
            vectors = vectors.unsqueeze(1)
            
            transformed = model(vectors)
            transformed = torch.nn.functional.normalize(transformed, dim=-1)
            
            # Remove sequence dimension: [B, 1, D] -> [B, D]
            transformed = transformed.squeeze(1)
            
            # Convert back to numpy
            output_vectors = transformed.cpu().numpy()
            
            if debug:
                similarity = torch.nn.functional.cosine_similarity(
                    vectors.squeeze(1), transformed, dim=1
                ).item()
                print(f"[DEBUG] Mamba transformation similarity: {similarity:.4f}")
        
        # Return transformed vectors (numpy). Decoding is handled by vec2text subscribers.
        output = {'status': 'success', 'result': output_vectors}

    except Exception as e:
        output = {'status': 'error', 'error': str(e)}
        if debug:
            import traceback
            print(f"[DEBUG] Exception in Mamba wrapper: {traceback.format_exc()}")

    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()
