#!/usr/bin/env python3
"""
Fast JXE vec2text wrapper using proper vec2text API
"""

import sys
import pickle
import numpy as np
from pathlib import Path

def main():
    # Get input/output paths from command line
    if len(sys.argv) != 3:
        print("Usage: jxe_wrapper_fixed.py <input_file> <output_file>", file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        # Force CPU for vec2text compatibility (before any imports)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Import torch after setting environment
        import torch
        
        # Disable MPS entirely
        if hasattr(torch.backends, 'mps'):
            torch.backends.mps.is_available = lambda: False
        
        # Load input data
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        vectors = torch.from_numpy(data['vectors'])
        metadata = data.get('metadata', {})
        steps = data.get('steps', 1)
        debug = data.get('debug', False)
        
        # Import vec2text API
        import vec2text.api
        
        if debug:
            print(f"[DEBUG] Processing {vectors.shape[0]} vectors with {steps} steps")
        
        # Load the corrector for GTR embeddings (force CPU)
        corrector = vec2text.api.load_pretrained_corrector("gtr-base")
        
        # Convert vectors to the right format (ensure CPU for vec2text)
        embeddings = vectors.detach().cpu()
        
        # Use vec2text API directly - try with better parameters
        results = vec2text.api.invert_embeddings(
            embeddings=embeddings,
            corrector=corrector,
            num_steps=max(3, steps),  # Use at least 3 steps for quality
            sequence_beam_width=1
        )
        
        if debug:
            print(f"[DEBUG] Vec2text returned {len(results)} results")
        
        # Save output
        output = {'status': 'success', 'result': results}
        
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        output = {'status': 'error', 'error': str(e)}
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()