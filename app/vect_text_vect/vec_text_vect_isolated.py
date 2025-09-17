#!/usr/bin/env python3
# 20250912T1400_10
"""
Main orchestrator for Text-Vector-Text processing pipeline with isolated environments
Each subscriber runs in its own virtual environment to avoid dependency conflicts
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from typing import Union, List, Dict, Any, Optional
from contextlib import contextmanager
from pathlib import Path
import pickle
import torch
import numpy as np
from tabulate import tabulate

# Set environment variables for compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@contextmanager
def timer(name: str, debug: bool = False):
    """Context manager for timing operations"""
    if debug:
        print(f"[DEBUG] Starting {name}...")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if debug:
            print(f"[DEBUG] {name} completed in {elapsed:.2f}s")


class IsolatedVecTextVectOrchestrator:
    """Main orchestrator with isolated subprocess execution for each subscriber"""
    
    def __init__(self, steps: int = 1, debug: bool = False, vmmoe_checkpoint: str = None, mamba_checkpoint: str = None, vec2text_backend: str = 'unified'):
        self.steps = steps
        self.debug = debug
        self.vmmoe_checkpoint = vmmoe_checkpoint or "output/vmmoe_full_parameter_v1p25/best_model.pth"
        self.mamba_checkpoint = mamba_checkpoint
        self.vec2text_backend = vec2text_backend  # 'unified' or 'isolated'
        self.text_encoder = None
        self._device = None
        self.project_root = Path(__file__).resolve().parents[2]
        
        # Environment paths
        self.venv_paths = {
            'jxe': self.project_root / 'venv_jxe',
            'ielab': self.project_root / 'venv_ielab', 
            'vmmoe': self.project_root / 'venv_vmmoe',
            'mamba': self.project_root / 'venv_vmmoe' # Assuming same venv for now
        }
        
        if self.debug:
            print("[DEBUG] Initializing Isolated Text-Vector-Text Orchestrator")
            print(f"[DEBUG] Vec2text steps: {self.steps}")
            print(f"[DEBUG] Project root: {self.project_root}")
            
        self._setup_device()
        self._load_text_encoder()
        
    def _setup_device(self):
        """Setup compute device"""
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            if self.debug:
                print("[DEBUG] Device selection: MPS available, using mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            if self.debug:
                print("[DEBUG] Device selection: CUDA available, using cuda")
        else:
            self._device = torch.device("cpu")
            if self.debug:
                print("[DEBUG] Device selection: Using cpu")
    
    def _load_text_encoder(self):
        """Load GTR-T5 text encoder using proper method for vec2text compatibility"""
        try:
            with timer("GTR-T5 loading", self.debug):
                from transformers import T5EncoderModel, AutoTokenizer
                # Load only the encoder model to avoid decoder weight warnings
                self.encoder = T5EncoderModel.from_pretrained("sentence-transformers/gtr-t5-base")
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
                # Move to device
                self.encoder = self.encoder.to(self._device)
                self.encoder.eval()
                # Keep SentenceTransformer for fallback/comparison
                from sentence_transformers import SentenceTransformer
                self.text_encoder = SentenceTransformer(
                    "sentence-transformers/gtr-t5-base",
                    device=str(self._device)
                )
        except Exception as e:
            print(f"[ERROR] Failed to load GTR-T5 encoder: {e}")
            sys.exit(1)
    
    def _run_subscriber_subprocess(self, 
                                   subscriber_name: str, 
                                   vectors: torch.Tensor,
                                   metadata: Dict[str, Any] = None,
                                   device_override: str = None) -> Any:
        """Run a subscriber in an isolated subprocess with its own virtual environment"""
        
        # Check if virtual environment exists
        venv_path = self.venv_paths.get(subscriber_name)
        # Use proper JXE wrapper, regular for others
        if subscriber_name == 'jxe':
            wrapper_path = self.project_root / 'app' / 'vect_text_vect' / 'subscriber_wrappers' / 'jxe_wrapper_proper.py'
        elif subscriber_name == 'mamba':
            wrapper_path = self.project_root / 'app' / 'vect_text_vect' / 'subscriber_wrappers' / 'mamba_wrapper.py'
        else:
            wrapper_path = self.project_root / 'app' / 'vect_text_vect' / 'subscriber_wrappers' / f'{subscriber_name}_wrapper.py'
        
        # For now, if venv doesn't exist, try using the main venv311
        if not venv_path or not venv_path.exists():
            venv_path = self.project_root / 'venv311'
            if not venv_path.exists():
                if self.debug:
                    print(f"[DEBUG] No virtual environment found for {subscriber_name}")
                return {'status': 'error', 'error': f'Virtual environment not found'}
        
        if not wrapper_path.exists():
            if self.debug:
                print(f"[DEBUG] Wrapper script not found: {wrapper_path}")
            return {'status': 'error', 'error': f'Wrapper script not found: {wrapper_path}'}
        
        # Create temporary files for data exchange
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as input_file:
            input_path = input_file.name
            
            # Prepare data based on subscriber type
            input_data = {
                'vectors': vectors.cpu().numpy(),
                'metadata': metadata,
                'steps': self.steps,
                'debug': self.debug,
                'device_override': device_override
            }
            
            # Add subscriber-specific parameters
            if subscriber_name == 'vmmoe':
                input_data['checkpoint_path'] = self.vmmoe_checkpoint
            elif subscriber_name == 'mamba':
                input_data['checkpoint_path'] = self.mamba_checkpoint
                input_data['normalize_output'] = True
            elif subscriber_name == 'ielab':
                input_data['beam_width'] = 1
                input_data['device_override'] = device_override
            
            pickle.dump(input_data, input_file)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as output_file:
            output_path = output_file.name
        
        try:
            # Run wrapper script in isolated environment, with optional project venv override
            python_path = venv_path / 'bin' / 'python3'
            try:
                if os.environ.get('VEC2TEXT_FORCE_PROJECT_VENV') == '1':
                    python_path = self.project_root / 'venv' / 'bin' / 'python3'
                    if self.debug:
                        print(f"[DEBUG] Forcing project venv for {subscriber_name}: {python_path}")
            except Exception:
                pass
            
            if self.debug:
                print(f"[DEBUG] Running {subscriber_name} wrapper with {python_path}")
                print(f"[DEBUG] Wrapper path: {wrapper_path}")
            
            env = os.environ.copy()
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            env['TOKENIZERS_PARALLELISM'] = 'false'
            # Enforce shared HF cache across all isolated backends
            try:
                hf_cache = str(self.project_root / '.hf_cache')
                env['HF_HOME'] = hf_cache
                env['TRANSFORMERS_CACHE'] = hf_cache
                env['HF_HUB_DISABLE_TELEMETRY'] = '1'
                if self.debug:
                    print(f"[DEBUG] Using HF cache: {hf_cache}")
            except Exception:
                pass
            # Ensure wrappers can import project modules (e.g., optional Mamba path under app/*)
            try:
                env['PYTHONPATH'] = f"{self.project_root}:{env.get('PYTHONPATH','')}"
            except Exception:
                pass
            if subscriber_name == 'ielab':
                # Apply device-specific settings for IELab (defaults to CPU)
                ielab_device = device_override or 'cpu'
                if ielab_device == 'cpu':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    env['TRANSFORMERS_DEFAULT_DEVICE'] = 'cpu'
                    env['PYTORCH_MPS_DISABLE'] = '1'
                elif ielab_device == 'mps':
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    env['TRANSFORMERS_DEFAULT_DEVICE'] = 'mps'
                    env.pop('PYTORCH_MPS_DISABLE', None)
                elif ielab_device == 'cuda':
                    env.pop('CUDA_VISIBLE_DEVICES', None)
                    env['TRANSFORMERS_DEFAULT_DEVICE'] = 'cuda'
                    env.pop('PYTORCH_MPS_DISABLE', None)
                env['TRANSFORMERS_NO_ACCELERATE'] = '1'
                env['ACCELERATE_DISABLE_INIT_EMPTY_WEIGHTS'] = '1'
            
            result = subprocess.run(
                [str(python_path), str(wrapper_path), input_path, output_path],
                capture_output=True,
                text=True,
                env=env,
                timeout=60  # 60 second timeout
            )
            
            # Always surface debug logs
            if self.debug and (result.stderr or result.stdout):
                print(f"[DEBUG] Subprocess stderr: {result.stderr}")
                print(f"[DEBUG] Subprocess stdout: {result.stdout}")

            if result.returncode != 0:
                error_msg = result.stderr or 'Subprocess failed'
                if self.debug:
                    pass
                return {'status': 'error', 'error': error_msg}
            
            # Load output
            with open(output_path, 'rb') as f:
                output = pickle.load(f)
            
            # Convert numpy arrays back to tensors if needed
            if output['status'] == 'success' and isinstance(output['result'], np.ndarray):
                output['result'] = torch.from_numpy(output['result'])
            
            return output
            
        except subprocess.TimeoutExpired:
            return {'status': 'error', 'error': 'Subprocess timeout'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
        finally:
            # Cleanup temp files
            for path in [input_path, output_path]:
                try:
                    os.unlink(path)
                except:
                    pass
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to vectors using GTR-T5 with vec2text-compatible method"""
        # Use direct transformer encoding for vec2text compatibility
        import torch.nn.functional as F
        
        all_embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                hidden_states = self.encoder(**inputs).last_hidden_state
                
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                s = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
                s = s / torch.sum(attention_mask, dim=1, keepdims=True)
                
                # Normalize
                embedding = F.normalize(s, p=2, dim=1)
                all_embeddings.append(embedding)
        
        return torch.cat(all_embeddings, dim=0)
    
    def process(self, 
                input_data: Union[str, List[str]], 
                subscribers: List[str] = None,
                subscriber_devices: Dict[str, str] = None) -> Dict[str, Any]:
        """Process text through pipeline with isolated subscribers"""
        
        # Convert single string to list
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = input_data
            
        if not texts:
            raise ValueError("No input data provided")
        
        # Default to all subscribers if none specified
        if subscribers is None:
            subscribers = ['vmmoe']  # Only vec2vec models, vec2text will be added as cascades
        
        results = {
            'metadata': {
                'num_texts': len(texts),
                'device': str(self._device),
                'gtr_device': str(self._device),
                'subscribers': subscribers,
                'steps': self.steps
            },
            'texts': texts,
            'results': []
        }
        
        # Encode all texts
        encoding_start = time.time()
        try:
            vectors = self.encode_texts(texts)
            encoding_time = time.time() - encoding_start
            results['metadata']['encoding_time'] = encoding_time
            
            if self.debug:
                print(f"[DEBUG]   Output shape: {vectors.shape}")
                norms = [torch.norm(vectors[i]).item() for i in range(vectors.shape[0])]
                print(f"[DEBUG]   Vector norms: {norms}")
                    
        except Exception as e:
            print(f"[ERROR] Text encoding failed: {e}")
            return results
        
        # Process each text through subscribers
        for i, text in enumerate(texts):
            text_results = {
                'index': i,
                'input_text': text,
                'subscribers': {}
            }
            
            vector = vectors[i:i+1].cpu()  # Keep batch dimension, force CPU for subprocess compatibility
            
            # Process vec2vec subscribers (vmmoe, mamba)
            vec2vec_subscribers = [name for name in subscribers if name in ['vmmoe', 'mamba']]
            for name in vec2vec_subscribers:
                if name not in self.venv_paths:
                    text_results['subscribers'][name] = {
                        'status': 'error',
                        'error': f'Unknown subscriber: {name}'
                    }
                    continue
                
                try:
                    start_time = time.time()
                    
                    # Run subscriber in isolated environment
                    device_for_subscriber = subscriber_devices.get(name) if subscriber_devices else None
                    result = self._run_subscriber_subprocess(
                        name, 
                        vector,
                        metadata={'original_texts': [text]},
                        device_override=device_for_subscriber
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if result['status'] == 'error':
                        text_results['subscribers'][name] = {
                            'status': 'error',
                            'error': result['error'],
                            'time': processing_time
                        }
                    else:
                        # Vector-to-vector transformation
                        if self.debug:
                            print(f"[DEBUG] Vec2Vec ({name}) result type: {type(result['result'])}")
                            if hasattr(result['result'], 'shape'):
                                print(f"[DEBUG] Vec2Vec ({name}) result shape: {result['result'].shape}")
                            if hasattr(result['result'], 'dtype'):
                                print(f"[DEBUG] Vec2Vec ({name}) result dtype: {result['result'].dtype}")
                        
                        output_vector_np = result['result']
                        
                        # Ensure it's numpy array before converting to tensor
                        if isinstance(output_vector_np, torch.Tensor):
                            if self.debug:
                                print(f"[DEBUG] Converting tensor to numpy first")
                            output_vector_np = output_vector_np.cpu().numpy()
                        elif isinstance(output_vector_np, list):
                            # Convert list to numpy array [B, D]
                            output_vector_np = np.asarray(output_vector_np, dtype=np.float32)
                        
                        output_vector = torch.from_numpy(output_vector_np).to(vector.device)
                        
                        # Calculate cosine similarity
                        cosine = torch.cosine_similarity(
                            vector.squeeze(), 
                            output_vector.squeeze(), 
                            dim=0
                        ).item()
                        
                        # Check for loading warnings and errors - FAIL HARD on issues
                        if result['status'] == 'success_with_warning' and 'loading_error' in result:
                            # VMMoE loading error - this is a CRITICAL failure
                            error_msg = f"VMMoE LOADING FAILURE: {result['loading_error']}"
                            text_results['subscribers'][name] = {
                                'status': 'error',
                                'error': error_msg,
                                'time': processing_time
                            }
                            continue  # Skip to next subscriber, do not process this as success
                        
                        output_text = "[Vector output]"
                        
                        # Additional validation for VMMoE output before marking as success
                        # Generic vec2vec validation
                        is_mamba = name == 'mamba'
                        model_name = 'Mamba' if is_mamba else 'VMMoE'

                        # Mamba is generative, expect lower cosine. VMMoE is transformative, expect higher.
                        min_cosine = 0.01 if is_mamba else 0.02
                        max_cosine = 0.90 if is_mamba else 0.98

                        if cosine > max_cosine:
                            text_results['subscribers'][name] = {
                                'status': 'error',
                                'error': f'{model_name} PASSTHROUGH SUSPECTED: Output cosine {cosine:.3f} too high',
                                'time': processing_time
                            }
                        elif cosine < min_cosine:
                            text_results['subscribers'][name] = {
                                'status': 'error',
                                'error': f'{model_name} SEMANTIC DESTRUCTION: Output cosine {cosine:.3f} too low',
                                'time': processing_time
                            }
                        else:
                            text_results['subscribers'][name] = {
                                'status': 'success',
                                'time': processing_time,
                                'cosine_to_original': cosine,
                                'output_type': 'vector',
                                'output_vector': output_vector,
                                'output_text': output_text
                            }
                        
                except Exception as e:
                    text_results['subscribers'][name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Add direct GTR → vec2text operations (no vec2vec transformation)
            vec2text_names = ['jxe', 'ielab']
            for vec2text_name in vec2text_names:
                if vec2text_name in self.venv_paths:
                    try:
                        start_time = time.time()
                        cascade_name = f'gtr → {vec2text_name}'
                        
                        # Choose backend: unified routes all vec2text to JXE
                        actual_runner = 'jxe' if self.vec2text_backend == 'unified' else vec2text_name
                        device_for_subscriber = subscriber_devices.get(vec2text_name) if subscriber_devices else None
                        direct_res = self._run_subscriber_subprocess(
                            actual_runner,
                            vector.cpu(),
                            metadata={'original_texts': [text]},
                            device_override=device_for_subscriber
                        )
                        
                        processing_time = time.time() - start_time
                        
                        if direct_res['status'] == 'error':
                            text_results['subscribers'][cascade_name] = {
                                'status': 'error',
                                'error': direct_res['error'],
                                'time': processing_time
                            }
                        else:
                            output_text = direct_res['result'][0] if isinstance(direct_res['result'], list) else direct_res['result']
                            re_encoded = self.encode_texts([output_text]).cpu()
                            cosine = torch.cosine_similarity(
                                vector.squeeze(),
                                re_encoded.squeeze(),
                                dim=0
                            ).item()
                            text_results['subscribers'][cascade_name] = {
                                'status': 'success',
                                'output': output_text,
                                'time': processing_time,
                                'cosine': cosine,
                                'device': device_for_subscriber or 'cpu'
                            }
                    except Exception as e:
                        text_results['subscribers'][f'gtr → {vec2text_name}'] = {
                            'status': 'error',
                            'error': str(e)
                        }
            

            # Cascaded Processing for vec2vec subscribers
            vec2vec_subscriber_names = [s for s in subscribers if s in ['vmmoe', 'mamba']]
            for vec2vec_name in vec2vec_subscriber_names:
                if vec2vec_name in text_results['subscribers'] and text_results['subscribers'][vec2vec_name]['status'] == 'success':
                    vec2vec_result = text_results['subscribers'][vec2vec_name]
                    if 'output_type' in vec2vec_result and vec2vec_result['output_type'] == 'vector' and 'output_vector' in vec2vec_result:
                        transformed_vector = vec2vec_result['output_vector']
                        
                        vec2text_subscribers = [name for name in subscribers if name not in ['vmmoe', 'mamba']]
                        for vec2text_name in vec2text_subscribers:
                            if vec2text_name in self.venv_paths:
                                try:
                                    start_time = time.time()
                                    cascade_name = f'{vec2vec_name} → {vec2text_name}'

                                    if isinstance(transformed_vector, np.ndarray):
                                        transformed_tensor = torch.from_numpy(transformed_vector)
                                    else:
                                        transformed_tensor = transformed_vector

                                    if transformed_tensor.dim() == 1:
                                        transformed_tensor = transformed_tensor.unsqueeze(0)

                                    input_tensor = transformed_tensor.cpu()
                                    
                                    # Choose backend: unified routes all vec2text to JXE
                                    actual_runner = 'jxe' if self.vec2text_backend == 'unified' else vec2text_name
                                    device_for_vec2text = subscriber_devices.get(vec2text_name) if subscriber_devices else None
                                    cascade_result = self._run_subscriber_subprocess(
                                        actual_runner,
                                        input_tensor,
                                        metadata={'original_texts': [text]},
                                        device_override=device_for_vec2text
                                    )
                                    
                                    processing_time = time.time() - start_time
                                    
                                    if cascade_result['status'] == 'error':
                                        error_msg = f"CASCADE ERROR: {cascade_result['error']}"
                                        text_results['subscribers'][cascade_name] = {
                                            'status': 'error',
                                            'error': error_msg,
                                            'time': processing_time
                                        }
                                    else:
                                        output_text = cascade_result['result'][0] if isinstance(cascade_result['result'], list) else cascade_result['result']
                                        re_encoded = self.encode_texts([output_text]).cpu()
                                        # Cosine to original input vector (pre-transform)
                                        cosine_to_original = torch.cosine_similarity(
                                            vector.squeeze(),
                                            re_encoded.squeeze(),
                                            dim=0
                                        ).item()
                                        # Cosine to transformed vector (post-transform)
                                        cosine_to_transformed = torch.cosine_similarity(
                                            input_tensor.squeeze(),
                                            re_encoded.squeeze(),
                                            dim=0
                                        ).item()
                                        text_results['subscribers'][cascade_name] = {
                                            'status': 'success',
                                            'output': output_text,
                                            'time': processing_time,
                                            'cosine': cosine_to_transformed,
                                            'cosine_to_original': cosine_to_original,
                                            'device': device_for_vec2text or 'cpu'
                                        }
                                except Exception as e:
                                    processing_time = time.time() - start_time
                                    text_results['subscribers'][cascade_name] = {
                                        'status': 'error',
                                        'error': f"CASCADE EXCEPTION: {str(e)}",
                                        'time': processing_time
                                    }
            
            # Cleanup: remove raw vectors from result payloads (humans can't read them)
            for _n, _s in list(text_results['subscribers'].items()):
                if isinstance(_s, dict) and 'output_vector' in _s:
                    _s.pop('output_vector', None)
            results['results'].append(text_results)
        
        return results
    
    def format_results_table(self, results: Dict[str, Any]) -> str:
        """Format results as a clean table"""
        rows = []
        
        for text_result in results['results']:
            idx = text_result['index']
            input_text = text_result['input_text']
            
            # Add input row
            rows.append([idx, 'INPUT', '-', '-', input_text])
            
            # Add subscriber results
            for name, sub_result in text_result['subscribers'].items():
                if sub_result.get('status') == 'error':
                    rows.append([
                        '', name, 'ERROR', '-', 
                        sub_result.get('error', 'Unknown error')[:80] + '...' if len(sub_result.get('error', '')) > 80 else sub_result.get('error', 'Unknown error')
                    ])
                elif sub_result.get('output_type') == 'vector':
                    # Use custom output_text if available, otherwise default
                    display_text = sub_result.get('output_text', '[Vector output]')
                    rows.append([
                        '', name,
                        f"{sub_result['time']:.2f}",
                        f"{sub_result['cosine_to_original']:.2f}",
                        display_text
                    ])
                else:
                    rows.append([
                        '', name,
                        f"{sub_result['time']:.2f}",
                        f"{sub_result['cosine']:.2f}",
                        sub_result.get('output', 'No output')
                    ])
        
        # Create table
        headers = ['#', 'Subscriber', 'Time (s)', 'Cosine', 'Output Text']
        table = tabulate(rows, headers=headers, tablefmt='grid', showindex=False)
        
        # Add header
        num_texts = results['metadata']['num_texts']
        subscribers = ', '.join(results['metadata']['subscribers'])
        steps = results['metadata']['steps']
        
        header = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ Text-Vector-Text Processing Results (Isolated Environments)                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Processing {num_texts} texts with subscribers: {subscribers}
Vec2text steps: {steps}

"""
        
        # Add summary
        encoding_time = results['metadata'].get('encoding_time', 0)
        summary = f"\nSummary:\n  GTR-T5 encoding time: {encoding_time:.2f}s ({num_texts} texts)\n"
        
        return header + table + summary


def main():
    parser = argparse.ArgumentParser(description='Text-Vector-Text Processing Pipeline (Isolated)')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-text', type=str, help='Single text to process')
    input_group.add_argument('--input-list', type=str, help='JSON list of texts')
    input_group.add_argument('--batch-file', type=str, help='File with texts (one per line)')
    
    # Processing options
    parser.add_argument('--subscribers', type=str, default='all',
                        help='Comma-separated list of subscribers (default: all)')
    parser.add_argument('--devices', type=str, default=None,
                        help='Comma-separated list of devices for each subscriber (cpu,mps,cuda). Must match subscriber count.')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of vec2text decoding steps (default: 1)')
    parser.add_argument('--vmmoe-checkpoint', type=str, help='Path to VMMoE model checkpoint')
    parser.add_argument('--mamba-checkpoint', type=str, help='Path to Mamba model checkpoint')
    parser.add_argument('--output-format', choices=['table', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--vec2text-backend', choices=['unified', 'isolated'], default='unified',
                        help='Vec2text backend for cascades: unified routes all vec2text to JXE; isolated uses each venv')
    
    args = parser.parse_args()
    
    # Parse input
    if args.input_text:
        texts = [args.input_text]
    elif args.input_list:
        try:
            texts = json.loads(args.input_list)
            if not isinstance(texts, list):
                raise ValueError("Input must be a list")
        except Exception as e:
            print(f"[ERROR] Failed to parse input list: {e}")
            sys.exit(1)
    else:  # batch_file
        try:
            with open(args.batch_file, 'r') as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[ERROR] Failed to read batch file: {e}")
            sys.exit(1)
    
    # Parse subscribers and devices
    if args.subscribers == 'all':
        subscribers = None
        subscriber_devices = None
    else:
        subscribers = [s.strip() for s in args.subscribers.split(',')]
        
        # Parse per-subscriber devices
        if args.devices:
            devices_list = [d.strip() for d in args.devices.split(',')]
            if len(devices_list) != len(subscribers):
                print(f"[ERROR] Device count ({len(devices_list)}) must match subscriber count ({len(subscribers)})")
                sys.exit(1)
            # Validate devices
            valid_devices = ['cpu', 'mps', 'cuda']
            for device in devices_list:
                if device not in valid_devices:
                    print(f"[ERROR] Invalid device '{device}'. Valid options: {valid_devices}")
                    sys.exit(1)
            subscriber_devices = dict(zip(subscribers, devices_list))
        else:
            subscriber_devices = None
    
    # Create orchestrator and process
    orchestrator = IsolatedVecTextVectOrchestrator(
        steps=args.steps,
        debug=args.debug,
        vmmoe_checkpoint=args.vmmoe_checkpoint,
        mamba_checkpoint=args.mamba_checkpoint,
        vec2text_backend=args.vec2text_backend
    )
    
    try:
        results = orchestrator.process(texts, subscribers, subscriber_devices)
        
        # Output results
        if args.output_format == 'table':
            print(orchestrator.format_results_table(results))
        else:
            print(json.dumps(results, indent=2, default=str))
            
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()