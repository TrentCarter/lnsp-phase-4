#!/usr/bin/env python3
"""
Standalone JXE vec2text wrapper for subprocess execution
"""

import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path

def main():
    # Get input/output paths from command line
    if len(sys.argv) != 3:
        print("Usage: jxe_wrapper.py <input_file> <output_file>", file=sys.stderr)
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
        metadata = data.get('metadata', {})
        steps = data.get('steps', 1)
        debug = data.get('debug', False)
        
        # Import and initialize subscriber
        from app.agents.vec2text_agent import create_vec2text_processor
        
        processor = create_vec2text_processor(
            teacher_model_name="data/teacher_models/gtr-t5-base"
        )
        
        # Process each vector
        results = []
        for i in range(vectors.shape[0]):
            vector = vectors[i]
            original_text = metadata.get('original_texts', [''])[i] if metadata else ''
            
            # Prepare embedding
            embedding_flat = vector.squeeze().detach()
            if embedding_flat.dim() > 1:
                embedding_flat = embedding_flat.view(-1)
            
            # Mock the get_vector function
            original_get_vector = processor.get_vector_from_source
            
            def mock_get_vector(text, vector_source):
                if vector_source == 'teacher':
                    return embedding_flat
                return original_get_vector(text, vector_source)
            
            processor.get_vector_from_source = mock_get_vector
            
            try:
                result = processor.iterative_vec2text_process(
                    input_text=original_text or " ",
                    vector_source='teacher',
                    num_iterations=max(1, steps)
                )
                
                decoded = (result or {}).get('final_text', '')
                if decoded:
                    results.append(decoded.strip())
                else:
                    results.append("[JXE: No text returned]")
                    
            except Exception as e:
                results.append(f"[JXE decode error: {e}]")
            finally:
                processor.get_vector_from_source = original_get_vector
        
        # Save output
        output = {'status': 'success', 'result': results}
        
    except Exception as e:
        output = {'status': 'error', 'error': str(e)}
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)

if __name__ == '__main__':
    main()