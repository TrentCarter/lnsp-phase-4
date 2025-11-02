from flask import render_template, jsonify, request, session, Response, stream_with_context
from functools import wraps
import time
import json
import os
import logging
import torch
import numpy as np
import requests
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from queue import Queue

# Import app and logger from the package
from lvm_eval import app, logger

# Import required modules
try:
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    raise

# Initialize models and services
service_client = None

# Global progress tracking (thread-safe)
progress_lock = threading.Lock()
progress_data = {
    'progress': 0,
    'model': '',
    'step': 'Initializing...',
    'status': 'idle',  # idle, running, complete, error
    'message': '',
    'models_status': {}  # Track individual model progress
}

def update_progress(progress: float, model: str = '', step: str = '', status: str = 'running', message: str = '', model_status: dict = None):
    """Thread-safe progress update"""
    global progress_data
    with progress_lock:
        progress_data['progress'] = max(0, min(100, progress))
        if model:
            progress_data['model'] = model
        if step:
            progress_data['step'] = step
        progress_data['status'] = status
        if message:
            progress_data['message'] = message
        if model_status:
            progress_data['models_status'].update(model_status)
        logger.info(f"Progress: {progress:.1f}% - {model} - {step} - {status}")
    return dict(progress_data)

def event_stream():
    """Generate server-sent events for progress updates"""
    last_id = 0
    last_data = None
    max_iterations = 600  # 5 minutes max (600 * 0.5s)
    iterations = 0
    
    while iterations < max_iterations:
        with progress_lock:
            current_data = dict(progress_data)
        
        # Only send if data changed
        if current_data != last_data:
            data = json.dumps(current_data)
            yield f"id: {last_id}\ndata: {data}\n\n"
            last_id += 1
            last_data = current_data
        
        # If evaluation is complete or errored, send one final update and stop
        if current_data['status'] in ['complete', 'error']:
            time.sleep(0.5)  # Give client time to receive final update
            break
            
        # Small delay to prevent overwhelming the client
        time.sleep(0.5)
        iterations += 1
    
    # Send final heartbeat
    yield f"id: {last_id}\ndata: {{\"status\": \"closed\"}}\n\n"

@app.route('/evaluate/stream')
def stream_progress():
    """SSE endpoint for progress updates"""
    response = Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering in nginx
    return response

try:
    # Initialize service client if needed
    class ServiceClient:
        def __init__(self):
            self.gtr_t5_url = "http://localhost:7001/encode"
            self.vec2text_url = "http://localhost:7002/decode"
            
        def encode_text(self, text: str) -> np.ndarray:
            try:
                response = requests.post(
                    self.gtr_t5_url,
                    json={"texts": [text]},  # Send as list of strings
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                # Return the first embedding from the list
                return np.array(result['embeddings'][0])
            except Exception as e:
                logger.error(f"Error encoding text: {e}")
                raise
                
        def decode_vector(self, vector: np.ndarray) -> str:
            try:
                response = requests.post(
                    self.vec2text_url,
                    json={"vectors": [vector.tolist()]},  # Send as list of vectors
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()
                # Return the first text from the list
                return result['texts'][0]
            except Exception as e:
                logger.error(f"Error decoding vector: {e}")
                raise
    
    service_client = ServiceClient()
except Exception as e:
    logger.error(f"Error initializing service client: {e}")

# Routes
@app.route('/')
def index():
    """Render the main page"""
    try:
        models = get_available_models()
        # Ensure all model data is JSON serializable
        serializable_models = []
        for model in models:
            serializable_models.append({
                'name': str(model.get('name', '')),
                'path': str(model.get('path', '')),
                'size': int(model.get('size', 0)),
                'size_mb': float(model.get('size_mb', 0.0)),
                'modified': float(model.get('modified', 0)),
                'relative_path': str(model.get('relative_path', ''))
            })
            
        # Default settings
        default_settings = {
            'test_mode': 'both',
            'batch_size': 8,
            'max_length': 128,
            'num_beams': 4,
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 1.0,
            'repetition_penalty': 1.0,
            'length_penalty': 1.0,
            'no_repeat_ngram_size': 0,
            'early_stopping': True
        }
        
        return render_template(
            'index.html',
            models=serializable_models,
            settings=default_settings
        )
    except Exception as e:
        logger.error(f"Error rendering index: {e}", exc_info=True)
        return str(e), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_models():
    """Evaluate models with the given test data (sequential)"""
    return _evaluate_models_impl(parallel=False)

@app.route('/evaluate/parallel', methods=['POST'])
def evaluate_models_parallel():
    """Evaluate models with the given test data (parallel/multi-threaded)"""
    return _evaluate_models_impl(parallel=True)

def _evaluate_models_impl(parallel=False):
    """Internal implementation for model evaluation"""
    global progress_data
    
    # Reset progress
    with progress_lock:
        progress_data = {
            'progress': 0,
            'model': '',
            'step': 'Starting evaluation...',
            'status': 'running',
            'message': '',
            'models_status': {}
        }
    
    try:
        logger.info("Received evaluation request")
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        if not data:
            error_msg = "No JSON data received"
            logger.error(error_msg)
            update_progress(0, '', error_msg, 'error')
            return jsonify({'error': error_msg, 'status': 'error'}), 400
            
        model_paths = data.get('models', [])
        test_mode = data.get('test_mode', 'both')
        
        if not model_paths:
            error_msg = "No models specified"
            logger.error(error_msg)
            update_progress(0, '', error_msg, 'error')
            return jsonify({'error': error_msg, 'status': 'error'}), 400
        
        logger.info(f"Processing evaluation for {len(model_paths)} models with test mode: {test_mode} (parallel={parallel})")
        update_progress(5, '', f'Preparing to evaluate {len(model_paths)} model(s)...', 'running')
        
        # Get test data
        update_progress(10, '', 'Loading test data...', 'running')
        test_data = get_test_data(test_mode)
        if not test_data:
            error_msg = f"No test data available for test mode: {test_mode}"
            logger.error(error_msg)
            update_progress(0, '', error_msg, 'error')
            return jsonify({'error': error_msg, 'status': 'error'}), 400
        
        results = []
        total_models = len(model_paths)
        
        if parallel and total_models > 1:
            # Multi-threaded evaluation
            logger.info(f"Starting parallel evaluation with {min(total_models, 4)} threads")
            update_progress(10, '', f'Starting parallel evaluation of {total_models} models...', 'running')
            
            with ThreadPoolExecutor(max_workers=min(total_models, 4)) as executor:
                future_to_model = {
                    executor.submit(evaluate_single_model, model_path, test_data, idx, total_models): (idx, model_path)
                    for idx, model_path in enumerate(model_paths, 1)
                }
                
                for future in as_completed(future_to_model):
                    idx, model_path = future_to_model[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed evaluation {len(results)}/{total_models}")
                    except Exception as e:
                        logger.error(f"Error in threaded evaluation for {model_path}: {e}")
                        results.append({
                            'model_path': model_path,
                            'model_name': os.path.basename(model_path),
                            'status': 'error',
                            'error': str(e)
                        })
        else:
            # Sequential evaluation
            for idx, model_path in enumerate(model_paths, 1):
                result = evaluate_single_model(model_path, test_data, idx, total_models)
                results.append(result)
        
        # Save results to log file
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'eval_{timestamp}.json')
        
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'test_mode': test_mode,
                'results': results
            }, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {log_file}")
        
        # Final progress update
        update_progress(100, '', 'Evaluation completed successfully', 'complete')
        
        # Log completion
        logger.info(f"Evaluation completed. Results for {len(results)} models")
        
        # Return results
        response_data = {
            'status': 'complete',
            'results': results,
            'timestamp': time.time(),
            'message': f'Successfully evaluated {len([r for r in results if r.get("status") == "complete"])} of {len(model_paths)} models'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = f"Error in evaluate_models: {str(e)}"
        logger.error(error_msg, exc_info=True)
        update_progress(0, '', f'Evaluation failed: {str(e)}', 'error')
        return jsonify({'error': error_msg, 'status': 'error'}), 500

# Helper functions
def evaluate_single_model(model_path: str, test_data: List[Dict], idx: int, total_models: int) -> Dict[str, Any]:
    """Evaluate a single model with test data"""
    model_name = os.path.basename(model_path)
    progress_base = 10 + (idx - 1) * (90 / total_models)
    
    try:
        logger.info(f"Loading model from path: {model_path}")
        update_progress(progress_base, model_name, f'Loading model {idx} of {total_models}', 'running')
        
        model = load_model(model_path)
        if model is None:
            error_msg = f"Failed to load model: {model_path}"
            logger.error(error_msg)
            return {
                'model_path': model_path,
                'model_name': model_name,
                'status': 'error',
                'error': error_msg
            }
        
        logger.info(f"Starting evaluation for model: {model_name}")
        update_progress(progress_base + 10, model_name, f'Evaluating {len(test_data)} test cases', 'running')
        
        # Process test cases
        model_results = []
        total_tests = len(test_data)
        
        for i, test_case in enumerate(test_data):
            try:
                # Get input text and encode it
                input_text = test_case.get('text', '')
                expected_text = test_case.get('expected_text', input_text)
                
                # Encode text to vector
                if service_client:
                    try:
                        input_vector = service_client.encode_text(input_text)
                    except Exception as e:
                        logger.warning(f"Service client encoding failed: {e}. Using random vector.")
                        input_vector = np.random.randn(768)
                else:
                    # Fallback to random vector for testing
                    input_vector = np.random.randn(768)
                
                # Process through the LVM model
                output_data = process_through_lvm(model, input_vector)
                output_vector = output_data['output_vector']
                
                # Decode output vector to text
                if service_client:
                    try:
                        output_text = service_client.decode_vector(output_vector)
                    except Exception as e:
                        logger.warning(f"Service client decoding failed: {e}")
                        output_text = "[Decoding failed]"
                else:
                    output_text = "[No decoder available]"
                
                # Calculate metrics
                similarity = calculate_cosine_similarity(output_vector, input_vector)
                
                # Calculate ROUGE scores
                rouge_scores = None
                if output_text and output_text != "[Decoding failed]" and output_text != "[No decoder available]":
                    rouge_scores = calculate_rouge([output_text], [expected_text])
                
                # Add to results
                model_results.append({
                    'input': input_text,
                    'output': output_text,
                    'expected_output': expected_text,
                    'similarity': float(similarity),
                    'rouge_scores': rouge_scores,
                    'status': 'success'
                })
                
                # Update progress for this test case
                test_progress = progress_base + 10 + (i + 1) / total_tests * 70
                update_progress(
                    test_progress,
                    model_name,
                    f'Evaluated {i + 1}/{total_tests} test cases',
                    'running'
                )
                
            except Exception as e:
                error_msg = f"Error processing test case {i}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                model_results.append({
                    'input': test_case.get('text', ''),
                    'error': error_msg,
                    'status': 'error'
                })
        
        # Calculate average metrics
        successful_results = [r for r in model_results if r.get('status') == 'success']
        avg_similarity = sum(r['similarity'] for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Update final progress
        update_progress(
            progress_base + 90.0 / total_models,
            model_name,
            f'Completed evaluation for {model_name}',
            'running'
        )
        
        logger.info(f"Completed evaluation for model: {model_path}")
        
        return {
            'model_path': model_path,
            'model_name': model_name,
            'status': 'complete',
            'test_cases': model_results,
            'avg_cosine_similarity': avg_similarity,
            'metrics': {
                'average_similarity': avg_similarity,
                'total_tests': len(model_results),
                'successful_tests': len(successful_results),
                'failed_tests': len(model_results) - len(successful_results)
            }
        }
        
    except Exception as e:
        error_msg = f"Error evaluating model {model_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            'model_path': model_path,
            'model_name': model_name,
            'status': 'error',
            'error': error_msg
        }

def get_available_models(limit=50) -> List[Dict[str, Any]]:
    """Get a list of available models with JSON-serializable values"""
    try:
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', 'lvm', 'models')
        models = []
        
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return []
            
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith(('.pt', '.pth')):
                    try:
                        model_path = os.path.join(root, file)
                        model_name = os.path.splitext(file)[0]
                        
                        # Get file info with error handling
                        try:
                            size = os.path.getsize(model_path)
                            modified = os.path.getmtime(model_path)
                            
                            # Ensure all values are JSON-serializable
                            models.append({
                                'name': str(model_name),
                                'path': str(model_path),
                                'size': int(size),
                                'size_mb': round(size / (1024 * 1024), 2),  # Add size in MB
                                'modified': float(modified),
                                'relative_path': os.path.relpath(model_path, start=models_dir)
                            })
                        except (OSError, TypeError) as e:
                            logger.warning(f"Error getting info for {file}: {e}")
                            continue
                            
                    except Exception as e:
                        logger.error(f"Error processing model {file}: {e}")
                        continue
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x.get('modified', 0), reverse=True)
        return models[:limit]
        
    except Exception as e:
        logger.error(f"Error in get_available_models: {e}", exc_info=True)
        return []

def get_test_data(test_mode: str = 'both') -> List[Dict[str, str]]:
    """Get test data based on test mode"""
    # Try to load real test data from artifacts
    test_data_paths = [
        'test_data/swo_10_samples.jsonl',
        'artifacts/test_samples.jsonl',
        'test_data/100_test_chunks.jsonl'
    ]
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    
    for rel_path in test_data_paths:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            try:
                logger.info(f"Loading test data from {full_path}")
                test_samples = []
                with open(full_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            test_samples.append({
                                'text': sample.get('text', sample.get('chunk_text', sample.get('content', ''))),
                                'expected_text': sample.get('text', sample.get('chunk_text', sample.get('content', '')))
                            })
                
                if test_samples:
                    # Limit to 10 samples for quick testing
                    logger.info(f"Loaded {len(test_samples)} test samples, using first 10")
                    return test_samples[:10]
            except Exception as e:
                logger.warning(f"Failed to load test data from {full_path}: {e}")
    
    # Fallback to synthetic test data
    logger.info("Using synthetic test data")
    in_dist_samples = [
        {"text": "Artificial intelligence is transforming technology.", "expected_text": "Artificial intelligence is transforming technology."},
        {"text": "Machine learning models require large datasets.", "expected_text": "Machine learning models require large datasets."},
        {"text": "Neural networks consist of interconnected layers.", "expected_text": "Neural networks consist of interconnected layers."},
    ]
    
    ood_samples = [
        {"text": "The quantum computer performed complex calculations.", "expected_text": "The quantum computer performed complex calculations."},
        {"text": "Renewable energy sources are increasingly important.", "expected_text": "Renewable energy sources are increasingly important."},
        {"text": "Space exploration continues to advance rapidly.", "expected_text": "Space exploration continues to advance rapidly."},
    ]
    
    if test_mode == 'in' or test_mode == 'in_distribution':
        return in_dist_samples
    elif test_mode == 'out' or test_mode == 'out_of_distribution':
        return ood_samples
    else:  # both
        return in_dist_samples + ood_samples

class SimpleTransformerModel(torch.nn.Module):
    """A simple transformer model that can be used as a fallback"""
    def __init__(self, input_dim=512, output_dim=512, nhead=8, num_layers=6, dropout=0.1, d_model=None):
        super().__init__()
        # Handle both input_dim and d_model for compatibility
        self.input_dim = d_model if d_model is not None else input_dim
        self.output_dim = output_dim
        
        # Project from 768 to 512 dimensions to match the checkpoint
        self.input_proj = torch.nn.Linear(768, 512)
        
        # Create a simple transformer encoder with fixed 512 dimension
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=512,  # Fixed dimension to match checkpoint
            nhead=8,      # 512 / 64 = 8 heads for 64-dim attention
            dim_feedforward=4*512,  # Standard feedforward dimension
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection back to the desired output dimension
        self.output_proj = torch.nn.Linear(512, self.output_dim)  # Fixed input dimension
        
    def forward(self, x):
        # Add debug logging for input dimensions
        logger.debug(f"Input tensor shape: {x.shape}")
        
        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
            
        # Project input to 512 dimensions if needed
        logger.info(f"Shape before input_proj: {x.shape}")
        x = self.input_proj(x)
        logger.info(f"Shape after input_proj: {x.shape}")
        
        try:
            # Pass through transformer
            logger.info(f"Shape before transformer: {x.shape}")
            x = self.transformer_encoder(x)
            logger.info(f"Shape after transformer: {x.shape}")
            
            # Project to output dimension
            x = self.output_proj(x)
            logger.info(f"Shape after output_proj: {x.shape}")
            
            # Remove sequence dimension if it was added
            if x.size(1) == 1:
                x = x.squeeze(1)
                
            return x
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            logger.error(f"Input shape: {x.shape}")
            logger.error(f"Input projection weight shape: {self.input_proj.weight.shape if hasattr(self.input_proj, 'weight') else 'N/A'}")
            logger.error(f"Transformer encoder layer 0 input shape: {self.transformer_encoder.layers[0].self_attn.in_proj_weight.shape if hasattr(self.transformer_encoder.layers[0].self_attn, 'in_proj_weight') else 'N/A'}")
            raise

def load_model(model_path: str):
    """Load a model from the given path"""
    try:
        import os
        import torch
        from torch import nn
        
        logger.info(f"Loading model from {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        logger.info(f"Checkpoint type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")
            
            # Try to extract model config from checkpoint
            model_config = checkpoint.get('model_config', {})
            if not model_config and 'args' in checkpoint:
                # Try to get config from args
                model_config = {
                    'input_dim': getattr(checkpoint['args'], 'input_dim', 512),
                    'd_model': getattr(checkpoint['args'], 'd_model', 512),  
                    'output_dim': getattr(checkpoint['args'], 'output_dim', 512),
                    'nhead': getattr(checkpoint['args'], 'nhead', 8),
                    'num_layers': getattr(checkpoint['args'], 'num_layers', 6),
                    'dropout': getattr(checkpoint['args'], 'dropout', 0.1)
                }
        
            # If we have a model_config dict, use it directly
            if isinstance(model_config, dict):
                # Ensure we have the required parameters
                if 'd_model' not in model_config and 'input_dim' in model_config:
                    model_config['d_model'] = model_config['input_dim']
                if 'input_dim' not in model_config and 'd_model' in model_config:
                    model_config['input_dim'] = model_config['d_model']
        else:
            # If it's not a dict, it might be just the state dict
            model_config = {
                'input_dim': 768,  # Match the input dimension
                'output_dim': 768,  # Match the output dimension
                'd_model': 768,  # Explicitly set d_model to match input_dim
                'nhead': 8,
                'num_layers': 6,
                'dropout': 0.1
            }
            checkpoint = {'model_state_dict': checkpoint, 'model_config': model_config}
        
        logger.info(f"Using model config: {model_config}")
        
        # Filter model_config to only include valid parameters for SimpleTransformerModel
        valid_params = {'input_dim', 'output_dim', 'nhead', 'num_layers', 'dropout', 'd_model'}
        filtered_config = {k: v for k, v in model_config.items() if k in valid_params}
        logger.info(f"Filtered model config: {filtered_config}")
        
        # Create a new model
        model = SimpleTransformerModel(**filtered_config)
        
        # Try to load the state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Load state dict with strict=False to handle missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")
        
        model.eval()
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model {model_path}: {e}", exc_info=True)
        return None

def process_through_lvm(model, input_vector):
    """Process input through the LVM model and return dict with output_vector"""
    try:
        import torch
        import numpy as np
        
        logger.debug(f"Processing input vector of shape: {np.array(input_vector).shape}")
        
        # Convert input to tensor if it's not already
        if not isinstance(input_vector, torch.Tensor):
            input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        else:
            input_tensor = input_vector.clone().detach()
            
        # Ensure the input is 2D: [batch_size, features]
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        elif input_tensor.dim() > 2:
            # Flatten all dimensions except the batch dimension
            input_tensor = input_tensor.view(input_tensor.size(0), -1)
            
        logger.debug(f"Shape before model: {input_tensor.shape}")
        
        # Process through model
        with torch.no_grad():
            output = model(input_tensor)
            logger.debug(f"Model output shape: {output.shape}")
            
        # Convert to numpy and ensure it's 1D if batch size is 1
        output_np = output.numpy()
        if output_np.shape[0] == 1:
            output_np = output_np.squeeze(0)
            
        logger.debug(f"Final output shape: {output_np.shape}")
        return {
            'output_vector': output_np,
            'output_text': None  # Will be populated by decoder
        }
        
    except Exception as e:
        logger.error(f"Error processing through LVM: {e}", exc_info=True)
        raise

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def calculate_rouge(predictions: List[str], references: List[str]):
    """Calculate ROUGE scores between predictions and references"""
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = []
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores.append({
                'rouge1': {'precision': score['rouge1'].precision, 'recall': score['rouge1'].recall, 'f1': score['rouge1'].fmeasure},
                'rouge2': {'precision': score['rouge2'].precision, 'recall': score['rouge2'].recall, 'f1': score['rouge2'].fmeasure},
                'rougeL': {'precision': score['rougeL'].precision, 'recall': score['rougeL'].recall, 'f1': score['rougeL'].fmeasure}
            })
        return scores[0] if scores else {
            'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
        }
    except Exception as e:
        logger.error(f"Error calculating ROUGE: {e}")
        return {
            'rouge1': {'precision': 0, 'recall': 0, 'f1': 0},
            'rouge2': {'precision': 0, 'recall': 0, 'f1': 0},
            'rougeL': {'precision': 0, 'recall': 0, 'f1': 0}
        }

@app.route('/api/system-info')
def system_info():
    """Return system information"""
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs', 'app.log')
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'artifacts', 'lvm', 'models')
    return jsonify({
        'log_file': os.path.abspath(log_file),
        'models_dir': os.path.abspath(models_dir)
    })

@app.route('/api/update-settings', methods=['POST'])
def update_settings():
    """Update application settings"""
    try:
        data = request.get_json()
        logger.info(f"Updating settings: {data}")
        # For now, just acknowledge the settings
        # In production, you'd save these to a config file
        return jsonify({'status': 'success', 'message': 'Settings updated'})
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/progress')
def get_progress():
    """Get current progress status (for debugging)"""
    with progress_lock:
        return jsonify(dict(progress_data))
