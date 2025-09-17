#!/usr/bin/env python3
""" 20250825T062650_3 """
"""
Main orchestrator for Text-Vector-Text processing pipeline
"""

import os
import sys
import json
import time
import argparse
from typing import Union, List, Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
from tabulate import tabulate
import torch
import subprocess
import tempfile
import pickle
from pathlib import Path

# Set environment variables for IELab compatibility
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


class VecTextVectOrchestrator:
    """Main orchestrator for text-vector-text pipeline"""
    
    def __init__(self, steps: int = 1, debug: bool = False, vmmoe_checkpoint: str = None):
        self.steps = steps
        self.debug = debug
        self.vmmoe_checkpoint = vmmoe_checkpoint or "output/vmmoe_concept_sequences_v2p1/best_model.pth"
        self.subscribers = {}
        self.text_encoder = None
        self._device = None
        
        if self.debug:
            print("[DEBUG] Initializing Text-Vector-Text Orchestrator")
            print(f"[DEBUG] Vec2text steps: {self.steps}")
            print(f"[DEBUG] VMMoE checkpoint: {self.vmmoe_checkpoint}")
            
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
        """Load GTR-T5 text encoder"""
        try:
            with timer("GTR-T5 loading", self.debug):
                # Add project root to path for imports
                from pathlib import Path
                project_root = Path(__file__).resolve().parents[2]
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from app.vect_text_vect.text_vect import TextToVectorEncoder
                self.text_encoder = TextToVectorEncoder(
                    device=str(self._device)
                )
        except Exception as e:
            print(f"[ERROR] Failed to load GTR-T5 encoder: {e}")
            sys.exit(1)
    
    def register_subscriber(self, name: str, subscriber):
        """Register a processing subscriber"""
        self.subscribers[name] = subscriber
        if self.debug:
            print(f"[DEBUG] Registered subscriber: {name}")
    
    def _load_subscribers(self, requested: List[str] = None):
        """Load requested subscribers"""
        available = {
            'jxe': ('app.vect_text_vect.subscriber_registry.jxe_vect2text', 'JXEVec2TextSubscriber'),
            'ielab': ('app.vect_text_vect.subscriber_registry.ielab_vec2text', 'IELabVec2TextSubscriber'),
            'vmmoe': ('app.vect_text_vect.subscriber_registry.vmmoe_vec2vec', 'VMMoEVec2VecSubscriber')
        }
        
        to_load = requested if requested else list(available.keys())
        
        for name in to_load:
            if name not in available:
                print(f"[WARNING] Unknown subscriber: {name}")
                continue
                
            module_name, class_name = available[name]
            try:
                if self.debug:
                    print(f"[DEBUG] Loading subscriber: {name}")
                # Ensure IELab loads in CPU-only mode without Accelerate meta tensors
                if name == 'ielab':
                    if self.debug:
                        print("[DEBUG]   Setting env for IELab: CPU-only, no accelerate")
                    os.environ['CUDA_VISIBLE_DEVICES'] = ''
                    os.environ['TRANSFORMERS_NO_ACCELERATE'] = '1'
                    os.environ['ACCELERATE_DISABLE_INIT_EMPTY_WEIGHTS'] = '1'
                    os.environ['TRANSFORMERS_DEFAULT_DEVICE'] = 'cpu'
                    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                
                # Dynamic import
                module = __import__(module_name, fromlist=[class_name])
                subscriber_class = getattr(module, class_name)
                
                # Special handling for ielab (CPU only)
                if name == 'ielab':
                    if self.debug:
                        print("[DEBUG]   Note: IELab requires CPU fallback - using cpu")
                    subscriber = subscriber_class(steps=self.steps, device='cpu', debug=self.debug)
                elif name == 'vmmoe':
                    # VMMoE doesn't need steps but needs checkpoint path
                    subscriber = subscriber_class(
                        checkpoint_path=self.vmmoe_checkpoint,
                        device=str(self._device), 
                        debug=self.debug
                    )
                else:
                    subscriber = subscriber_class(steps=self.steps, device=str(self._device), debug=self.debug)
                
                self.register_subscriber(name, subscriber)
                
            except Exception as e:
                error_msg = f"Failed to load {name}: {str(e)}"
                if self.debug:
                    print(f"[DEBUG] {error_msg}")
                    import traceback; traceback.print_exc()
                if name == 'ielab':
                    pr = Path(__file__).resolve().parents[2]
                    vpy = pr / "venv_ielab" / "bin" / "python"
                    wpy = pr / "app" / "vect_text_vect" / "subscriber_wrappers" / "ielab_wrapper.py"
                    class _IELabWrap:
                        def __init__(self, steps, vpy, wpy): self.steps, self.vpy, self.wpy = steps, str(vpy), str(wpy)
                        def process(self, vectors, metadata=None):
                            with tempfile.TemporaryDirectory() as td:
                                ip, op = os.path.join(td, "in.pkl"), os.path.join(td, "out.pkl")
                                with open(ip, "wb") as f: pickle.dump({"vectors": vectors.detach().cpu().numpy(), "steps": self.steps, "beam_width": 1}, f)
                                subprocess.run([self.vpy, self.wpy, ip, op], check=True)
                                with open(op, "rb") as f: res = pickle.load(f)
                                return res.get("result", [f"[IELab wrapper error: {res.get('error','unknown')}]"])
                    self.register_subscriber(name, _IELabWrap(self.steps, vpy, wpy))
                else:
                    self.subscribers[name] = {'error': error_msg}
    
    def process(self, 
                input_data: Union[str, List[str]], 
                subscribers: List[str] = None) -> Dict[str, Any]:
        """Process text through pipeline"""
        
        # Convert single string to list
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = input_data
            
        if not texts:
            raise ValueError("No input data provided")
        
        # Load subscribers
        self._load_subscribers(subscribers)
        
        results = {
            'metadata': {
                'num_texts': len(texts),
                'device': str(self._device),
                'subscribers': list(self.subscribers.keys()),
                'steps': self.steps
            },
            'texts': texts,
            'results': []
        }
        
        # Encode all texts
        encoding_start = time.time()
        try:
            vectors = self.text_encoder.encode(texts)
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
            
            vector = vectors[i:i+1]  # Keep batch dimension
            
            # Process through each subscriber
            for name, subscriber in self.subscribers.items():
                if isinstance(subscriber, dict) and 'error' in subscriber:
                    # Error subscriber
                    text_results['subscribers'][name] = {
                        'status': 'error',
                        'error': subscriber['error']
                    }
                    continue
                
                try:
                    start_time = time.time()
                    
                    if name == 'vmmoe':
                        # Vector-to-vector transformation
                        output = subscriber.process(vector)
                        processing_time = time.time() - start_time
                        
                        # Calculate cosine similarity
                        cosine = torch.cosine_similarity(
                            vector.squeeze(), 
                            output.squeeze(), 
                            dim=0
                        ).item()
                        
                        text_results['subscribers'][name] = {
                            'status': 'success',
                            'time': processing_time,
                            'cosine_to_original': cosine,
                            'decoded_by': {}
                        }
                        
                        # Process VMMoE output through vec2text subscribers
                        bridge_text = None  # for IELab bridge via JXE
                        for vec2text_name in ['jxe', 'ielab']:
                            if vec2text_name in self.subscribers and not isinstance(self.subscribers[vec2text_name], dict):
                                try:
                                    v2t_start = time.time()
                                    # Bridge: decode JXE then re-encode for IELab
                                    if vec2text_name == 'ielab' and bridge_text is not None:
                                        re_encoded = self.text_encoder.encode([bridge_text])
                                        decoded = self.subscribers[vec2text_name].process(
                                            re_encoded,
                                            metadata={'original_texts': [text]}
                                        )
                                    else:
                                        decoded = self.subscribers[vec2text_name].process(
                                            output,
                                            metadata={'original_texts': [text]}
                                        )
                                    v2t_time = time.time() - v2t_start
                                    
                                    # Re-encode and calculate cosine
                                    re_encoded = self.text_encoder.encode([decoded[0]])
                                    v2t_cosine = torch.cosine_similarity(
                                        vector.squeeze(),
                                        re_encoded.squeeze(),
                                        dim=0
                                    ).item()
                                    
                                    text_results['subscribers'][name]['decoded_by'][vec2text_name] = {
                                        'output': decoded[0],
                                        'time': v2t_time,
                                        'cosine': v2t_cosine
                                    }
                                    if vec2text_name == 'jxe' and decoded and isinstance(decoded, list):
                                        bridge_text = decoded[0]
                                except Exception as e:
                                    text_results['subscribers'][name]['decoded_by'][vec2text_name] = {
                                        'status': 'error',
                                        'error': str(e)
                                    }
                    else:
                        # Vec2text subscriber
                        output = subscriber.process(vector, metadata={'original_texts': [text]})
                        processing_time = time.time() - start_time
                        
                        # Re-encode and calculate cosine similarity
                        re_encoded = self.text_encoder.encode([output[0]])
                        cosine = torch.cosine_similarity(
                            vector.squeeze(),
                            re_encoded.squeeze(),
                            dim=0
                        ).item()
                        
                        text_results['subscribers'][name] = {
                            'status': 'success',
                            'output': output[0],
                            'time': processing_time,
                            'cosine': cosine
                        }
                        
                except Exception as e:
                    text_results['subscribers'][name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
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
                if name == 'vmmoe':
                    # Handle VMMoE and its decoded outputs
                    for v2t_name, v2t_result in sub_result.get('decoded_by', {}).items():
                        if v2t_result.get('status') == 'error':
                            rows.append([
                                '', f'vmmoe→{v2t_name}', 'ERROR', '-', 
                                v2t_result.get('error', 'Unknown error')
                            ])
                        else:
                            rows.append([
                                '', f'vmmoe→{v2t_name}', 
                                f"{v2t_result['time']:.2f}",
                                f"{v2t_result['cosine']:.2f}",
                                v2t_result['output']
                            ])
                else:
                    # Regular vec2text subscriber
                    if sub_result.get('status') == 'error':
                        rows.append([
                            '', name, 'ERROR', '-', 
                            sub_result.get('error', 'Unknown error')
                        ])
                    else:
                        rows.append([
                            '', name,
                            f"{sub_result['time']:.2f}",
                            f"{sub_result['cosine']:.2f}",
                            sub_result['output']
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
║ Text-Vector-Text Processing Results                                                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

Processing {num_texts} texts with subscribers: {subscribers}
Vec2text steps: {steps}

"""
        
        # Add summary
        encoding_time = results['metadata'].get('encoding_time', 0)
        total_time = time.time()  # Would need to track from start
        summary = f"\nSummary:\n  GTR-T5 encoding time: {encoding_time:.2f}s ({num_texts} texts)\n"
        
        # Calculate best reconstructions
        best_direct = {'name': None, 'cosine': 0}
        best_vmmoe = {'name': None, 'cosine': 0}
        
        for text_result in results['results']:
            for name, sub_result in text_result['subscribers'].items():
                if name != 'vmmoe' and sub_result.get('status') == 'success':
                    if sub_result['cosine'] > best_direct['cosine']:
                        best_direct = {'name': name, 'cosine': sub_result['cosine']}
                elif name == 'vmmoe':
                    for v2t_name, v2t_result in sub_result.get('decoded_by', {}).items():
                        if v2t_result.get('status') != 'error' and v2t_result.get('cosine', 0) > best_vmmoe['cosine']:
                            best_vmmoe = {'name': v2t_name, 'cosine': v2t_result['cosine']}
        
        if best_direct['name']:
            summary += f"  Best direct reconstruction: {best_direct['name']} (avg cosine: {best_direct['cosine']:.3f})\n"
        if best_vmmoe['name']:
            summary += f"  Best via VMMoE: {best_vmmoe['name']} (avg cosine: {best_vmmoe['cosine']:.3f})\n"
        
        return header + table + summary


def main():
    parser = argparse.ArgumentParser(description='Text-Vector-Text Processing Pipeline')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-text', type=str, help='Single text to process')
    input_group.add_argument('--input-list', type=str, help='JSON list of texts')
    input_group.add_argument('--batch-file', type=str, help='File with texts (one per line)')
    
    # Processing options
    parser.add_argument('--subscribers', type=str, default='all',
                        help='Comma-separated list of subscribers (default: all)')
    parser.add_argument('--steps', type=int, default=1,
                        help='Number of vec2text decoding steps (default: 1)')
    parser.add_argument('--vmmoe-checkpoint', type=str, 
                        default='output/vmmoe_concept_sequences_v2p1/best_model.pth',
                        help='Path to VMMoE checkpoint (default: output/vmmoe_concept_sequences_v2p1/best_model.pth)')
    parser.add_argument('--output-format', choices=['table', 'json'], default='table',
                        help='Output format (default: table)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
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
    
    # Parse subscribers
    if args.subscribers == 'all':
        subscribers = None
    else:
        subscribers = [s.strip() for s in args.subscribers.split(',')]
    
    # Create orchestrator and process
    orchestrator = VecTextVectOrchestrator(
        steps=args.steps, 
        debug=args.debug,
        vmmoe_checkpoint=args.vmmoe_checkpoint
    )
    
    try:
        results = orchestrator.process(texts, subscribers)
        
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