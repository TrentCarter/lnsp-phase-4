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
import psutil
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
    try:
        from bert_score import score as bertscore
        BERTSCORE_AVAILABLE = True
    except Exception:
        BERTSCORE_AVAILABLE = False
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    raise

# Initialize models and services
class Vec2TextServiceClient:
    """Client for Vec2Text encoding/decoding services"""
    
    def __init__(self, encoder_url="http://localhost:7001", decoder_url="http://localhost:7002"):
        self.encoder_url = encoder_url
        self.decoder_url = decoder_url
        self.session = requests.Session()
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to 768D vector using port 7001"""
        try:
            payload = {"texts": [text]}
            response = self.session.post(f"{self.encoder_url}/encode", json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if "embeddings" in data and data["embeddings"]:
                return np.array(data["embeddings"][0])  # Return first (and only) embedding
            else:
                logger.warning(f"Unexpected encoder response format: {data}")
                return np.random.randn(768)
                
        except Exception as e:
            logger.warning(f"Encoder service failed: {e}. Using random vector.")
            return np.random.randn(768)
    
    def decode_vector(self, vector: np.ndarray, steps: int = 3) -> str:
        """Decode 768D vector to text using port 7002"""
        try:
            # Ensure vector is 2D for the API
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            payload = {
                "vectors": vector.tolist(),
                "subscriber": "ielab",  # Use IELab as recommended
                "steps": int(steps)  # Configurable decoding steps
            }
            
            response = self.session.post(f"{self.decoder_url}/decode", json=payload, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if "results" in data and data["results"]:
                return data["results"][0]  # Return first result
            else:
                logger.warning(f"Unexpected decoder response format: {data}")
                return "[Decoding failed]"
                
        except Exception as e:
            logger.warning(f"Decoder service failed: {e}")
            return "[Decoding failed]"

# Initialize service client with proper ports
try:
    service_client = Vec2TextServiceClient(
        encoder_url="http://localhost:7001",
        decoder_url="http://localhost:7002"
    )
    logger.info("Vec2Text service client initialized with ports 7001/7002")
except Exception as e:
    logger.warning(f"Failed to initialize service client: {e}")
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

# Remove duplicate ServiceClient - using Vec2TextServiceClient above

# Routes
@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    from flask import send_from_directory
    return send_from_directory(os.path.join(app.root_path, 'static'), 
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    """Render the main page"""
    try:
        # Load more models for initial display (200)
        models = get_available_models(limit=200)
        # Ensure all model data is JSON serializable
        serializable_models = []
        for model in models:
            mod = model.get('modified', 0)
            try:
                modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(mod)))
            except Exception:
                modified_str = ''
            serializable_models.append({
                'name': str(model.get('name', '')),
                'path': str(model.get('path', '')),
                'size': int(model.get('size', 0)),
                'size_mb': float(model.get('size_mb', 0.0)),
                'modified': float(mod),
                'modified_str': modified_str,
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

@app.route('/evaluate/sweep', methods=['POST'])
def evaluate_sweep():
    """Sweep over vec2text_steps and num_concepts to build a heatmap grid for one or more models.
    For MVP, compute on the first provided model and return a grid of averages.
    """
    try:
        data = request.get_json() or {}
        model_paths = data.get('models', [])
        if not model_paths:
            return jsonify({'error': 'No models specified', 'status': 'error'}), 400
        test_mode = data.get('test_mode', 'both')
        num_test_cases = int(data.get('num_test_cases', 10))
        steps_list = data.get('steps_list', [3, 5, 7])
        concepts_list = data.get('concepts_list', [3, 5, 7])

        # Use the first model for heatmap computation
        model_path = model_paths[0]
        logger.info(f"Starting sweep for model: {model_path} steps={steps_list}, concepts={concepts_list}")

        grid = []
        for concepts in concepts_list:
            # Build test data per concepts setting once
            td = get_test_data(test_mode, num_concepts=int(concepts))
            if not td:
                logger.warning(f"No test data for concepts={concepts}")
                continue
            if len(td) > num_test_cases:
                td = td[:num_test_cases]
            for steps in steps_list:
                # Evaluate single model with given steps and concepts
                res = evaluate_single_model(model_path, td, idx=1, total_models=1, vec2text_steps=int(steps))
                avg_cos = float(res.get('avg_cosine_similarity') or 0.0)
                avg_lat = float(res.get('avg_latency') or 0.0)
                avg_bert = res.get('avg_bert') or {}
                avg_bert_f1 = float(avg_bert.get('f1') or 0.0) if isinstance(avg_bert, dict) else 0.0
                grid.append({
                    'steps': int(steps),
                    'concepts': int(concepts),
                    'avg_cosine': avg_cos,
                    'avg_bert_f1': avg_bert_f1,
                    'avg_latency': avg_lat,
                })

        payload = {
            'status': 'complete',
            'model': model_path,
            'grid': grid,
            'steps_list': steps_list,
            'concepts_list': concepts_list,
        }
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Error in evaluate_sweep: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

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
        num_test_cases = data.get('num_test_cases', 10)  # New parameter
        vec2text_steps = int(data.get('vec2text_steps', 3))
        num_concepts = data.get('num_concepts', 5)  # Number of input chunks for next-chunk prediction
        start_article_index = int(data.get('start_article_index', 0))
        if start_article_index < 0:
            start_article_index = 0
        random_start = bool(data.get('random_start', False))
        
        if not model_paths:
            error_msg = "No models specified"
            logger.error(error_msg)
            update_progress(0, '', error_msg, 'error')
            return jsonify({'error': error_msg, 'status': 'error'}), 400
        
        logger.info(f"Processing evaluation for {len(model_paths)} models with test mode: {test_mode} (parallel={parallel})")
        logger.info(f"Using {num_test_cases} test cases/model, {num_concepts} chunks → 1 next, vec2text_steps={vec2text_steps}, start_article_index={start_article_index}, random_start={random_start}")
        update_progress(5, '', f'Preparing to evaluate {len(model_paths)} model(s) with {num_test_cases} test cases each...', 'running')
        
        # Get test data
        update_progress(10, '', 'Loading test data...', 'running')
        test_data = get_test_data(test_mode, num_concepts=num_concepts, start_article_index=start_article_index, random_start=random_start)
        if not test_data:
            error_msg = f"No test data available for test mode: {test_mode}"
            logger.error(error_msg)
            update_progress(0, '', error_msg, 'error')
            return jsonify({'error': error_msg, 'status': 'error'}), 400
        
        # Limit test data to requested number
        if len(test_data) > num_test_cases:
            test_data = test_data[:num_test_cases]
        elif len(test_data) < num_test_cases:
            logger.warning(f"Only {len(test_data)} test samples available, requested {num_test_cases}")
        
        logger.info(f"Using {len(test_data)} test cases per model")
        
        results = []
        total_models = len(model_paths)
        
        if parallel and total_models > 1:
            # Multi-threaded evaluation
            logger.info(f"Starting parallel evaluation with {min(total_models, 4)} threads")
            update_progress(10, '', f'Starting parallel evaluation of {total_models} models...', 'running')
            
            with ThreadPoolExecutor(max_workers=min(total_models, 4)) as executor:
                future_to_model = {
                    executor.submit(evaluate_single_model, model_path, test_data, idx, total_models, vec2text_steps=vec2text_steps): (idx, model_path)
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
                result = evaluate_single_model(model_path, test_data, idx, total_models, vec2text_steps=vec2text_steps)
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
        
        # Add aggregation for multiple models
        if len(results) > 1:
            aggregated = aggregate_model_results(results)
            response_data = {
                'status': 'complete',
                'results': results,
                'aggregated': aggregated,
                'timestamp': time.time(),
                'message': f'Successfully evaluated {len([r for r in results if r.get("status") == "complete"])} of {len(model_paths)} models with aggregated results'
            }
        else:
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

def aggregate_model_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results from multiple model evaluations"""
    if not results:
        return {}
    
    # Filter successful results
    successful_results = [r for r in results if r.get('status') == 'complete']
    
    if not successful_results:
        return {'error': 'No successful model evaluations to aggregate'}
    
    # Aggregate metrics
    total_tests = sum(r.get('metrics', {}).get('total_tests', 0) for r in successful_results)
    successful_tests = sum(r.get('metrics', {}).get('successful_tests', 0) for r in successful_results)
    failed_tests = sum(r.get('metrics', {}).get('failed_tests', 0) for r in successful_results)
    
    # Average similarity scores
    avg_similarities = [r.get('avg_cosine_similarity', 0) for r in successful_results if r.get('avg_cosine_similarity') is not None]
    overall_avg_similarity = sum(avg_similarities) / len(avg_similarities) if avg_similarities else 0
    
    # Best and worst performing models
    model_performance = []
    for result in successful_results:
        model_performance.append({
            'model_name': result.get('model_name', 'Unknown'),
            'avg_similarity': result.get('avg_cosine_similarity', 0),
            'successful_tests': result.get('metrics', {}).get('successful_tests', 0),
            'total_tests': result.get('metrics', {}).get('total_tests', 0)
        })
    
    # Sort by performance
    model_performance.sort(key=lambda x: x['avg_similarity'], reverse=True)
    
    return {
        'num_models_evaluated': len(successful_results),
        'total_tests_across_models': total_tests,
        'successful_tests_across_models': successful_tests,
        'failed_tests_across_models': failed_tests,
        'overall_avg_similarity': round(overall_avg_similarity, 4),
        'best_performing_model': model_performance[0] if model_performance else None,
        'worst_performing_model': model_performance[-1] if model_performance else None,
        'model_rankings': model_performance,
        'success_rate': round(successful_tests / total_tests * 100, 2) if total_tests > 0 else 0
    }

# Helper functions
def evaluate_single_model(model_path: str, test_data: List[Dict], idx: int, total_models: int, vec2text_steps: int = 3) -> Dict[str, Any]:
    """Evaluate a single model with test data"""
    model_name = os.path.basename(model_path) if model_path != 'DIRECT' else 'DIRECT (No LVM)'
    progress_base = 10 + (idx - 1) * (90 / total_models)
    is_direct = (model_path == 'DIRECT')
    
    try:
        if not is_direct:
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
        else:
            logger.info("Using DIRECT pipeline (no LVM) - text->GTR-T5->768D->vec2text->text")
            update_progress(progress_base, model_name, f'Testing DIRECT pipeline {idx} of {total_models}', 'running')
            model = None  # No model needed for DIRECT
        
        logger.info(f"Starting evaluation for model: {model_name}")
        update_progress(progress_base + 10, model_name, f'Evaluating {len(test_data)} test cases', 'running')
        
        # Initialize metrics tracking
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = initial_memory
        total_latency = 0.0
        test_latencies = []
        bert_p_list, bert_r_list, bert_f1_list = [], [], []
        comp_ratios, out_lengths, exp_lengths = [], [], []
        
        # Process test cases
        model_results = []
        total_tests = len(test_data)
        
        for i, test_case in enumerate(test_data):
            test_start_time = time.time()
            try:
                # Check if we have separate input chunks (new format) or single text (legacy)
                input_chunks = test_case.get('input_chunks', None)
                
                # Determine expected text based on model type
                if is_direct:
                    # For DIRECT model: expected output is the last input chunk (pass-through)
                    expected_text = test_case.get('last_input_chunk', '')
                else:
                    # For LVM models: expected output is the next chunk (N+1)
                    expected_text = test_case.get('expected_text', '')
                
                # For Wikipedia chunks with separate inputs (sequence of chunks)
                if input_chunks and test_case.get('source') == 'wikipedia':
                    if not input_chunks or not expected_text:
                        logger.warning("Missing input chunks or expected text, skipping")
                        continue
                    
                    if is_direct:
                        logger.debug(f"Processing Wikipedia DIRECT pipeline with {len(input_chunks)} input chunks")
                        logger.debug(f"Expected (last input chunk): {expected_text[:100]}...")
                    else:
                        logger.debug(f"Processing Wikipedia next-chunk prediction with {len(input_chunks)} input chunks")
                        logger.debug(f"Expected (next chunk): {expected_text[:100]}...")
                    
                    # Process this case with vector sequence (handled below)
                    input_text = None  # Flag to indicate we need vector sequence processing
                    
                # Legacy format with single text input
                else:
                    input_text = test_case.get('text', '')
                    
                # For ontology chains, check if we need to parse the expected output  
                if input_text and ' -> ' in input_text:
                    concepts = input_text.split(' -> ')
                    if len(concepts) >= 2:
                        if is_direct:
                            # For DIRECT model: use all but last concept as input, last as expected
                            input_text = ' -> '.join(concepts[:-1])
                            expected_text = concepts[-1]
                        else:
                            # For LVM models: use all but last concept as input, last as expected
                            input_text = ' -> '.join(concepts[:-1])
                            expected_text = concepts[-1]
                        logger.debug(f"Parsed chain: input='{input_text}', expected='{expected_text}'")
                    else:
                        logger.warning(f"Invalid ontology chain: {input_text}")
                        continue
                elif input_text:
                    # For other text, ensure we have both input and expected
                    if not expected_text:
                        logger.debug(f"Skipping incomplete test case")
                        continue
                
                # Encode input to vector(s)
                if input_chunks:  # New format: sequence of N chunks
                    # Encode each chunk separately to create a sequence of vectors
                    input_vectors = []
                    if service_client:
                        try:
                            for chunk in input_chunks:
                                vector = service_client.encode_text(chunk)
                                input_vectors.append(vector)
                            # Stack into shape (N, 768) for sequence processing
                            input_vectors = np.stack(input_vectors)
                            logger.debug(f"Encoded {len(input_chunks)} chunks into vector sequence shape: {input_vectors.shape}")
                        except Exception as e:
                            logger.warning(f"Service client encoding failed: {e}. Using random vectors.")
                            input_vectors = np.random.randn(len(input_chunks), 768)
                    else:
                        # Fallback to random vectors for testing
                        input_vectors = np.random.randn(len(input_chunks), 768)
                    
                elif input_text:  # Legacy format: single text
                    if service_client:
                        try:
                            input_vector = service_client.encode_text(input_text)
                            # Convert to sequence format with single vector
                            input_vectors = input_vector.reshape(1, 768)
                        except Exception as e:
                            logger.warning(f"Service client encoding failed: {e}. Using random vector.")
                            input_vectors = np.random.randn(1, 768)
                    else:
                        # Fallback to random vector for testing
                        input_vectors = np.random.randn(1, 768)
                else:
                    logger.warning("No input to encode, skipping")
                    continue
                
                # Process through the LVM model or use DIRECT pipeline
                if is_direct:
                    # DIRECT: Skip LVM, just use the last input vector as output (for testing)
                    # In real use, this would pass through the full sequence
                    output_vector = input_vectors[-1]  # Use last vector from sequence
                    logger.debug(f"DIRECT mode: bypassing LVM, using last of {len(input_vectors)} input vectors as output")
                else:
                    # Normal LVM processing with vector sequence
                    output_data = process_through_lvm(model, input_vectors)
                    output_vector = output_data['output_vector']
                
                # Decode output vector to text (for display purposes) BEFORE computing similarity
                if service_client:
                    try:
                        # Ensure vector is properly formatted
                        if hasattr(output_vector, 'cpu'):
                            output_vector = output_vector.cpu().numpy()
                        output_text = service_client.decode_vector(output_vector, steps=vec2text_steps)
                        logger.debug(f"Decoded output: {output_text[:100]}...")
                    except Exception as e:
                        logger.warning(f"Service client decoding failed: {e}")
                        output_text = "[Decoding failed]"
                else:
                    output_text = "[No decoder available]"

                # For next-token prediction, we need to encode the expected text to compare
                if service_client:
                    try:
                        if is_direct:
                            # For DIRECT model: encode the expected text and encode the decoded output text
                            # This gives the true pipeline quality: text->GTR-T5->768D->vec2text->text->GTR-T5->768D
                            expected_vector = service_client.encode_text(expected_text)
                            reconstructed_vector = service_client.encode_text(output_text)
                            similarity = calculate_cosine_similarity(reconstructed_vector, expected_vector)
                            # Debug: log the actual similarity calculation
                            logger.debug(f"DIRECT: reconstructed vs expected similarity: {similarity:.4f}")
                            logger.debug(f"DIRECT: expected text: {expected_text[:100]}...")
                            logger.debug(f"DIRECT: decoded output: {output_text[:100]}...")
                        else:
                            # For LVM models: encode the expected text and compare to output vector
                            expected_vector = service_client.encode_text(expected_text)
                            similarity = calculate_cosine_similarity(output_vector, expected_vector)
                    except Exception as e:
                        logger.warning(f"Service client expected encoding failed: {e}. Using random comparison.")
                        similarity = 0.0
                else:
                    # Without encoder, we can't properly evaluate semantic similarity
                    similarity = 0.0
                
                # Calculate ROUGE scores for text comparison
                rouge_scores = None
                if output_text and output_text not in ["[Decoding failed]", "[No decoder available]"]:
                    rouge_scores = calculate_rouge([output_text], [expected_text])
                
                # BERTScore (P/R/F1) if available
                bert_scores = None
                if BERTSCORE_AVAILABLE and output_text and expected_text:
                    try:
                        P, R, F1 = bertscore([output_text], [expected_text], lang='en', rescale_with_baseline=True)
                        p, r, f1 = float(P[0]), float(R[0]), float(F1[0])
                        bert_scores = {'p': p, 'r': r, 'f1': f1}
                        bert_p_list.append(p); bert_r_list.append(r); bert_f1_list.append(f1)
                    except Exception as e:
                        logger.warning(f"BERTScore failed: {e}")
                
                # Compression/length metrics
                ol = len(output_text) if output_text else 0
                el = len(expected_text) if expected_text else 0
                cr = (ol / el) if el > 0 else 0.0
                out_lengths.append(ol); exp_lengths.append(el); comp_ratios.append(cr)
                
                # Prepare input display text with numbered chunks (full text)
                if input_chunks:
                    # Show all chunks with [N] annotations (full text)
                    chunk_displays = []
                    for j, chunk in enumerate(input_chunks, 1):
                        chunk_displays.append(f"[{j}]: {chunk}")
                    input_display = "\n".join(chunk_displays)
                else:
                    input_display = input_text if input_text else 'Processed'
                
                # Sample memory after test case operations
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > peak_memory:
                        peak_memory = current_memory
                except Exception:
                    pass

                # Calculate latency for this test case
                test_latency = time.time() - test_start_time
                total_latency += test_latency
                test_latencies.append(test_latency)
                
                # Add to results
                model_results.append({
                    'input': input_display,
                    'input_chunks': input_chunks if input_chunks else None,  # Store full chunks separately
                    'output': output_text,
                    'expected': expected_text,  # Changed from 'expected_output' to match frontend
                    'cosine_similarity': float(similarity),  # Changed from 'similarity' to match frontend
                    'rouge_scores': rouge_scores,
                    'latency': test_latency,
                    'bert': bert_scores,
                    'compression_ratio': cr,
                    'output_length': ol,
                    'expected_length': el,
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
                # Prepare input display text with numbered chunks
                if input_chunks:
                    # Show all chunks with [N] annotations
                    chunk_displays = []
                    for i, chunk in enumerate(input_chunks, 1):
                        # Truncate each chunk to reasonable length
                        chunk_preview = chunk[:80] + "..." if len(chunk) > 80 else chunk
                        chunk_displays.append(f"[{i}]: {chunk_preview}")
                    input_display = "\n".join(chunk_displays)
                else:
                    input_display = input_text[:100] if input_text else 'Processed'
                
                # Sample memory on error path as well
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    if current_memory > peak_memory:
                        peak_memory = current_memory
                except Exception:
                    pass

                test_latency = time.time() - test_start_time
                test_latencies.append(test_latency)
                
                model_results.append({
                    'input': input_display,
                    'input_chunks': input_chunks if input_chunks else None,  # Store full chunks separately
                    'expected': expected_text if expected_text else '',
                    'output': output_text if output_text else '',
                    'cosine_similarity': similarity,
                    'latency': test_latency,
                    'metadata': test_case.get('metadata', {}),
                    'status': 'error'
                })
        
        # Calculate average metrics
        successful_results = [r for r in model_results if r.get('status') == 'success']
        avg_similarity = sum(r['cosine_similarity'] for r in successful_results) / len(successful_results) if successful_results else 0
        avg_latency = total_latency / len(test_latencies) if test_latencies else 0.0
        
        # Aggregate BERTScore and compression/length metrics
        avg_bert = {
            'p': sum(bert_p_list) / len(bert_p_list) if bert_p_list else None,
            'r': sum(bert_r_list) / len(bert_r_list) if bert_r_list else None,
            'f1': sum(bert_f1_list) / len(bert_f1_list) if bert_f1_list else None,
        }
        avg_compression_ratio = sum(comp_ratios) / len(comp_ratios) if comp_ratios else None
        avg_output_length = sum(out_lengths) / len(out_lengths) if out_lengths else None
        avg_expected_length = sum(exp_lengths) / len(exp_lengths) if exp_lengths else None
        
        # Update final progress
        update_progress(
            progress_base + 90.0 / total_models,
            model_name,
            f'Completed evaluation for {model_name}',
            'running'
        )
        
        logger.info(f"Completed evaluation for model: {model_path}")
        
        # Get model metadata for real models
        model_metadata = {}
        if model_path != 'DIRECT' and os.path.exists(model_path):
            try:
                stat = os.stat(model_path)
                model_metadata = {
                    'full_path': model_path,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': stat.st_mtime,
                    'modified_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
                }
                # Try to parse training_history.json from same directory
                try:
                    th_dir = os.path.dirname(model_path)
                    th_path = os.path.join(th_dir, 'training_history.json')
                    if os.path.exists(th_path):
                        with open(th_path, 'r') as fh:
                            th = json.load(fh)
                        # Heuristic extraction of best val
                        best = {}
                        for k in ['best_val_loss', 'best_val_metric', 'best', 'best_metrics']:
                            if k in th:
                                best['value'] = th[k]
                                break
                        for k in ['best_epoch', 'epoch', 'best_step', 'step']:
                            if k in th:
                                best['at'] = th[k]
                                break
                        model_metadata['training_history'] = {
                            'best': best,
                            'has_history': True
                        }
                except Exception as e:
                    logger.warning(f"training_history.json parse failed for {model_path}: {e}")
            except Exception as e:
                logger.warning(f"Could not get model metadata for {model_path}: {e}")
        
        return {
            'model_path': model_path,
            'model_name': model_name,
            'model_metadata': model_metadata,
            'status': 'complete',
            'test_cases': model_results,
            'avg_cosine_similarity': avg_similarity,
            'avg_latency': avg_latency,
            'avg_bert': avg_bert,
            'avg_compression_ratio': avg_compression_ratio,
            'avg_output_length': avg_output_length,
            'avg_expected_length': avg_expected_length,
            'metrics': {
                'average_similarity': avg_similarity,
                'average_latency': avg_latency,
                'average_bert': avg_bert,
                'average_compression_ratio': avg_compression_ratio,
                'average_output_length': avg_output_length,
                'average_expected_length': avg_expected_length,
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

def get_available_models(limit=None, search_all=True) -> List[Dict[str, Any]]:
    """Get a list of available models with JSON-serializable values
    
    Args:
        limit: Maximum number of models to return (None = no limit)
        search_all: If True, search entire project. If False, only artifacts/lvm/models
    """
    models = []
    
    # Add DIRECT model at the top (no LVM, just GTR-T5 -> vec2text pipeline test)
    direct_model = {
        'name': 'DIRECT (No LVM - Pipeline Test)',
        'path': 'DIRECT',  # Special identifier
        'size': 0,
        'size_mb': 0.0,
        'modified': time.time(),  # Current time
        'relative_path': 'DIRECT',
        'is_direct': True  # Flag to identify this special model
    }
    models.append(direct_model)
    logger.info(f"Added DIRECT model to list: {direct_model['name']}")
    
    try:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        
        # Define search directories
        if search_all:
            search_dirs = [
                os.path.join(base_dir, 'artifacts', 'lvm', 'models'),
                os.path.join(base_dir, 'models'),
                os.path.join(base_dir, 'lvm_eval', 'models'),
            ]
        else:
            search_dirs = [os.path.join(base_dir, 'artifacts', 'lvm', 'models')]
        
        # Don't reset models list - we already added DIRECT model above
        
        for models_dir in search_dirs:
            if not os.path.exists(models_dir):
                continue
            logger.info(f"Scanning for models in: {models_dir}")
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
                                    'relative_path': os.path.relpath(model_path, start=base_dir)
                                })
                            except (OSError, TypeError) as e:
                                logger.warning(f"Error getting info for {file}: {e}")
                                continue
                                
                        except Exception as e:
                            logger.error(f"Error processing model {file}: {e}")
                            continue
        
        # Debug: Check if DIRECT model is in the list
        logger.info(f"Total models before filtering: {len(models)}")
        for i, m in enumerate(models[:3]):  # Check first 3 models
            logger.debug(f"Model {i}: name={m.get('name')}, is_direct={m.get('is_direct', False)}")
        
        # Separate DIRECT model from real models
        direct_models = [m for m in models if m.get('is_direct', False)]
        real_models = [m for m in models if not m.get('is_direct', False)]
        
        logger.info(f"Found {len(direct_models)} DIRECT models and {len(real_models)} real models")
        
        # Sort real models by modification time (newest first)
        real_models.sort(key=lambda x: x.get('modified', 0), reverse=True)
        
        # Put DIRECT model at the top, followed by sorted real models
        models = direct_models + real_models
        
        logger.info(f"Found {len(models)} total models (DIRECT first: {models[0]['name'] if models else 'No models'})")
        
        if limit is not None:
            return models[:limit]
        return models
        
    except Exception as e:
        logger.error(f"Error in get_available_models: {e}", exc_info=True)
        return []

def get_test_data(test_mode: str = 'both', num_concepts: int = 5, start_article_index: int = 0, random_start: bool = False) -> List[Dict[str, Any]]:
    """Get test data from Wikipedia chunks for next-chunk prediction
    
    Args:
        test_mode: 'in', 'out', or 'both'
        num_concepts: Number of input chunks (N) - the model will predict chunk N+1
    
    Returns:
        List of test cases, each with N input chunks and 1 expected output chunk
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    
    # Use the real Wikipedia chunks database
    wikipedia_file = 'data/datasets/wikipedia/wikipedia_500k.jsonl'  # Use 500k sample for faster loading
    wikipedia_path = os.path.join(base_dir, wikipedia_file)
    
    if os.path.exists(wikipedia_path):
        try:
            logger.info(f"Loading Wikipedia chunks from {wikipedia_path}")
            logger.info(f"Will use {num_concepts} chunks as input, predicting chunk {num_concepts+1}")
            
            # First, load articles and split them into sentences/chunks
            articles_data = []
            with open(wikipedia_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < start_article_index:
                        continue
                    if i >= start_article_index + 50:  # Read a window of articles to build samples
                        break
                    try:
                        article = json.loads(line)
                        text = article.get('text', '').strip()
                        title = article.get('title', '')
                        
                        if text and len(text) > 500:  # Only use substantial articles
                            # Split article into sentences (chunks)
                            import re
                            # Split on sentence boundaries (., !, ?) followed by space and capital letter
                            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
                            
                            # Filter out very short sentences
                            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
                            
                            if len(sentences) > num_concepts + 1:  # Need at least N+1 sentences
                                articles_data.append({
                                    'title': title,
                                    'sentences': sentences
                                })
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded {len(articles_data)} articles with sufficient sentences (start={start_article_index}, random_start={random_start})")
            
            # Optionally shuffle articles for random start behavior
            if random_start:
                import random as _random
                _random.shuffle(articles_data)

            # Create test samples from articles (N chunks in, 1 chunk out)
            all_samples = []
            max_samples_per_article = 3  # Limit samples per article to prevent excessive evaluation
            total_max_samples = 1000  # Overall limit raised to respect frontend num_test_cases
            
            for article in articles_data:
                sentences = article['sentences']
                title = article['title']
                
                # Create sliding windows of N+1 sentences, but limit per article
                samples_created = 0
                for i in range(min(len(sentences) - num_concepts, max_samples_per_article)):
                    if i + num_concepts < len(sentences) and len(all_samples) < total_max_samples:
                        # Input: N sequential sentences as separate chunks (NOT joined)
                        input_chunks = sentences[i:i+num_concepts]
                        
                        # Expected output: The next sentence (chunk N+1)
                        expected_chunk = sentences[i + num_concepts]
                        last_input_chunk = input_chunks[-1]
                        
                        all_samples.append({
                            'input_chunks': input_chunks,  # List of N separate chunks
                            'expected_text': expected_chunk,  # Chunk N+1 as expected output for LVMs
                            'last_input_chunk': last_input_chunk,  # Last input chunk for DIRECT model
                            'source': 'wikipedia',
                            'metadata': {
                                'title': title,
                                'chunk_index': i,
                                'num_input_chunks': num_concepts
                            }
                        })
                        samples_created += 1
                        
                        # Limit samples per article to avoid bias
                        if len([s for s in all_samples if s['metadata']['title'] == title]) >= 5:
                            break
            
            logger.info(f"Created {len(all_samples)} test samples (next-chunk prediction)")
            
            # Return appropriate samples based on test mode
            if test_mode in ['in', 'in_distribution']:
                selected = all_samples  # Do not cap here; frontend controls num_test_cases
            elif test_mode in ['out', 'out_of_distribution']:
                selected = all_samples[30:60] if len(all_samples) > 30 else all_samples
            else:  # both
                selected = all_samples[:60]
            
            logger.info(f'Selected {len(selected)} test cases for {test_mode} testing')
            logger.info(f'Each test case: {num_concepts} chunks → 1 next chunk prediction')
            return selected
            
        except Exception as e:
            logger.warning(f"Failed to load Wikipedia chunks: {e}")
    
    # Fallback to original logic if ontology chains not available
    # Try to load real test data from artifacts and test_data directories
    test_data_paths = [
        'test_data/swo_10_samples.jsonl',
        'test_data/100_test_chunks.jsonl',
        'artifacts/test_samples.jsonl',
        'artifacts/swo_test_samples.jsonl'
    ]
    
    all_samples = []
    
    # Load all available real data
    for rel_path in test_data_paths:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            try:
                logger.info(f"Loading test data from {full_path}")
                with open(full_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            # Extract text content - handle different data structures
                            text_content = None
                            
                            if isinstance(sample, dict):
                                # Try different possible text fields
                                text_content = (
                                    sample.get('text') or 
                                    sample.get('chunk_text') or 
                                    sample.get('content')
                                )
                                
                                # For ontology chains, use concepts as text
                                if not text_content and 'concepts' in sample:
                                    text_content = ' -> '.join(sample['concepts'])
                                
                                # Fallback to string representation
                                if not text_content:
                                    text_content = str(sample)
                            
                            if text_content and len(text_content.strip()) > 10:  # Only include substantial text
                                all_samples.append({
                                    'text': text_content.strip(),
                                    'source': rel_path,
                                    'metadata': sample if isinstance(sample, dict) else {}
                                })
                                
                logger.info(f"Loaded {len(all_samples)} samples so far from {rel_path}")
            except Exception as e:
                logger.warning(f"Failed to load test data from {full_path}: {e}")
    
    # If we have real data, use it
    if all_samples:
        logger.info(f"Using {len(all_samples)} real test samples")
        
        # Categorize samples based on likely training usage
        # In-training data: structured/scientific content (SWO)
        # Out-of-training data: general text chunks
        in_training_samples = []
        out_training_samples = []
        
        for sample in all_samples:
            source = sample.get('source', '')
            text = sample.get('text', '').lower()
            
            # SWO data is likely training data (structured, scientific)
            if 'swo' in source.lower():
                in_training_samples.append(sample)
            # General chunked data is likely out-of-training data
            else:
                out_training_samples.append(sample)
        
        logger.info(f'Categorized {len(in_training_samples)} in-training samples, {len(out_training_samples)} out-of-training samples')
        
        if test_mode == 'in' or test_mode == 'in_distribution':
            selected_samples = in_training_samples
            logger.info(f'Selected {len(selected_samples)} in-training samples')
        elif test_mode == 'out' or test_mode == 'out_of_distribution':
            selected_samples = out_training_samples
            logger.info(f'Selected {len(selected_samples)} out-of-training samples')
        else:  # both
            selected_samples = in_training_samples + out_training_samples
            logger.info(f'Selected {len(selected_samples)} samples (both in and out of training)')
        
        # Format for evaluation
        test_samples = []
        for sample in selected_samples:
            test_samples.append({
                'text': sample['text'],
                'expected_text': sample['text'],  # For now, expected = input
                'source': sample.get('source', 'unknown'),
                'metadata': sample.get('metadata', {})
            })
        
        # Limit to reasonable size for testing
        return test_samples[:50] if len(test_samples) > 50 else test_samples
    
    # Fallback to synthetic test data if no real data found
    logger.info("No real test data found, using synthetic test data")
    
    # In-training data: structured, scientific, ontological content (likely used for training)
    in_training_samples = [
        {"text": "Artificial intelligence is transforming technology.", "expected_text": "Artificial intelligence is transforming technology."},
        {"text": "Machine learning models require large datasets.", "expected_text": "Machine learning models require large datasets."},
        {"text": "Neural networks consist of interconnected layers.", "expected_text": "Neural networks consist of interconnected layers."},
        {"text": "Deep learning uses multiple processing layers.", "expected_text": "Deep learning uses multiple processing layers."},
        {"text": "Computer vision enables machines to interpret visual information.", "expected_text": "Computer vision enables machines to interpret visual information."},
        {"text": "Natural language processing helps computers understand human language.", "expected_text": "Natural language processing helps computers understand human language."},
        {"text": "Data science combines statistics and programming.", "expected_text": "Data science combines statistics and programming."},
        {"text": "Ontology provides structured knowledge representation.", "expected_text": "Ontology provides structured knowledge representation."},
        {"text": "Knowledge graphs connect concepts and relationships.", "expected_text": "Knowledge graphs connect concepts and relationships."},
        {"text": "Semantic web enables machine-readable data.", "expected_text": "Semantic web enables machine-readable data."},
    ]
    
    # Out-of-training data: general content not typically used for ontological training
    out_training_samples = [
        {"text": "The weather today is sunny and warm.", "expected_text": "The weather today is sunny and warm."},
        {"text": "I enjoy reading books on various topics.", "expected_text": "I enjoy reading books on various topics."},
        {"text": "Cooking dinner requires fresh ingredients.", "expected_text": "Cooking dinner requires fresh ingredients."},
        {"text": "Traveling to new places broadens perspectives.", "expected_text": "Traveling to new places broadens perspectives."},
        {"text": "Music brings joy to many people's lives.", "expected_text": "Music brings joy to many people's lives."},
        {"text": "Exercise is important for maintaining health.", "expected_text": "Exercise is important for maintaining health."},
        {"text": "Learning new skills develops the mind.", "expected_text": "Learning new skills develops the mind."},
        {"text": "Gardening can be a relaxing hobby.", "expected_text": "Gardening can be a relaxing hobby."},
        {"text": "Photography captures memorable moments.", "expected_text": "Photography captures memorable moments."},
        {"text": "Writing stories expresses creativity.", "expected_text": "Writing stories expresses creativity."},
    ]
    
    if test_mode == 'in' or test_mode == 'in_distribution':
        return in_training_samples
    elif test_mode == 'out' or test_mode == 'out_of_distribution':
        return out_training_samples
    else:  # both
        return in_training_samples + out_training_samples

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
        
        logger.info(f"Loading model from request path: {model_path}")

        # Resolve model path: if relative, make it relative to project root
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        candidate_path = model_path
        # Paths displayed in UI like '/artifacts/...' are project-root relative, not OS root
        ui_root_relative_prefixes = ('/artifacts/', '/models/', '/lvm_eval/')
        if (not os.path.isabs(candidate_path)) or candidate_path.startswith(ui_root_relative_prefixes):
            candidate_path = os.path.join(base_dir, candidate_path.lstrip('/'))
        # Normalize and resolve symlinks
        candidate_path = os.path.realpath(candidate_path)
        logger.info(f"Resolved model path: {candidate_path}")
        
        if not os.path.exists(candidate_path):
            logger.error(f"Model file not found: {candidate_path}")
            return None

        # Load the checkpoint
        checkpoint = torch.load(candidate_path, map_location=torch.device('cpu'))
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
        
        model_type = checkpoint.get('model_type', 'transformer')
        logger.info(f"Model type: {model_type}")
        
        # Create model based on type
        if model_type == 'amn':
            # AMN models use a simpler architecture - just use a basic MLP
            input_dim = model_config.get('input_dim', 768)
            output_dim = model_config.get('output_dim', 768)
            hidden_dim = model_config.get('hidden_dim', 512)
            d_model = model_config.get('d_model', 256)
            
            # Create a simple MLP that mimics AMN
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
            )
            logger.info(f"Created AMN-compatible model: {input_dim} -> {d_model} -> {hidden_dim} -> {output_dim}")
        else:
            # Check if this is an encoder-decoder model
            state_dict = checkpoint.get('model_state_dict', {})
            has_decoder = any('decoder' in key for key in state_dict.keys())
            
            if has_decoder:
                # This is an encoder-decoder model, use a compatible architecture
                logger.info("Detected encoder-decoder transformer model")
                
                # Create a wrapper that can load the encoder-decoder model
                class EncoderDecoderTransformer(nn.Module):
                    def __init__(self, **kwargs):
                        super().__init__()
                        self.input_proj = nn.Linear(768, 512)
                        self.pos_encoder = PositionalEncoding(512, 0.1)
                        
                        # Create decoder (the model uses decoder for processing)
                        decoder_layer = nn.TransformerDecoderLayer(
                            d_model=512,
                            nhead=8,
                            dim_feedforward=2048,
                            dropout=0.1,
                            batch_first=True
                        )
                        self.transformer_decoder = nn.TransformerDecoder(
                            decoder_layer,
                            num_layers=4
                        )
                        
                        # Output head
                        self.head = nn.Sequential(
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.LayerNorm(512),
                            nn.Linear(512, 768)
                        )
                    
                    def forward(self, x):
                        # x shape: (batch, seq, 768)
                        x = self.input_proj(x)  # (batch, seq, 512)
                        x = self.pos_encoder(x)  # Add positional encoding
                        
                        # Use decoder with self-attention (memory=x for cross-attention)
                        x = self.transformer_decoder(x, x)  # (batch, seq, 512)
                        
                        # Take last sequence element for next-token prediction
                        x = x[:, -1, :]  # (batch, 512)
                        
                        # Apply output head
                        x = self.head(x)  # (batch, 768)
                        
                        return x
                
                # Define PositionalEncoding if not already defined
                class PositionalEncoding(nn.Module):
                    def __init__(self, d_model, dropout=0.1, max_len=100):
                        super().__init__()
                        self.dropout = nn.Dropout(p=dropout)
                        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
                    
                    def forward(self, x):
                        x = x + self.pe[:, :x.size(1), :]
                        return self.dropout(x)
                
                model = EncoderDecoderTransformer()
                logger.info("Created encoder-decoder transformer model")
            else:
                # Original encoder-only transformer
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

def process_through_lvm(model, input_vectors):
    """Process input sequence through the LVM model and return dict with output_vector
    
    Args:
        model: The LVM model
        input_vectors: Either a single vector (1, 768) or sequence of vectors (N, 768)
                      where N is the number of input chunks/concepts
    
    Returns:
        dict with 'output_vector': the predicted next vector (768,)
    """
    try:
        import torch
        import numpy as np
        
        input_array = np.array(input_vectors)
        logger.debug(f"Processing input sequence of shape: {input_array.shape}")
        
        # Convert input to tensor
        if not isinstance(input_vectors, torch.Tensor):
            input_tensor = torch.tensor(input_array, dtype=torch.float32)
        else:
            input_tensor = input_vectors.clone().detach()
        
        # Handle different input shapes
        if input_tensor.dim() == 1:
            # Single vector without batch dim -> add both batch and sequence dims
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 768)
        elif input_tensor.dim() == 2:
            # Shape is either (N, 768) for sequence or (1, 768) for single
            # Add batch dimension
            input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, N, 768)
        # else: already has batch dimension (batch, sequence, features)
            
        logger.debug(f"Input tensor shape for model: {input_tensor.shape}")
        
        # Process through model - expects (batch, sequence, features)
        with torch.no_grad():
            # The model should process the sequence and output next vector
            output = model(input_tensor)
            logger.debug(f"Model output shape: {output.shape}")
            
        # Convert to numpy and handle different output shapes
        output_np = output.numpy()
        
        # Handle different output shapes
        if output_np.ndim == 3:
            # Shape is (batch, seq, features) - take last token from first batch
            output_np = output_np[0, -1, :]
        elif output_np.ndim == 2:
            # Shape is (batch, features) - take first batch
            output_np = output_np[0] if output_np.shape[0] > 1 else output_np.squeeze()
        # else: already 1D
            
        logger.debug(f"Final output vector shape: {output_np.shape}")
        return {
            'output_vector': output_np,
            'output_text': None  # Will be populated by decoder
        }
        
    except Exception as e:
        logger.error(f"Error processing through LVM: {e}", exc_info=True)
        raise

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    val = float(np.dot(vec1, vec2) / denom)
    # Clamp to [-1, 1] and guard NaN
    if not np.isfinite(val):
        return 0.0
    return max(-1.0, min(1.0, val))

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

@app.route('/api/models')
def get_models():
    """Get list of available models"""
    try:
        # Get limit from query parameter, default to None (all models)
        limit = request.args.get('limit', type=int, default=None)
        search_all = request.args.get('search_all', type=bool, default=True)
        
        models = get_available_models(limit=limit, search_all=search_all)
        return jsonify({
            'status': 'success',
            'models': models,
            'total': len(models),
            'limit': limit
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}", exc_info=True)
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/test')
def test_page():
    """Serve test page for debugging"""
    from flask import send_from_directory
    return send_from_directory(os.path.dirname(__file__), 'test_dashboard.html')

@app.route('/test-ui')
def test_ui_page():
    """Serve comprehensive UI test page"""
    from flask import send_from_directory
    return send_from_directory(os.path.dirname(__file__), 'test_ui.html')

@app.route('/diagnostic')
def diagnostic_page():
    """Serve diagnostic page for troubleshooting"""
    from flask import send_from_directory
    return send_from_directory(os.path.dirname(__file__), 'diagnostic.html')
