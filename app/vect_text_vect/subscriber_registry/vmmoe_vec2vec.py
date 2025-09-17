#!/usr/bin/env python3
# 20250820T195939_1
"""
VMMoE Vector-to-Vector transformation subscriber
"""

import torch
from typing import Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class VMMoEVec2VecSubscriber:
    """Transform vectors using VMMoE model"""
    
    def __init__(self, 
                 checkpoint_path: str = "output/vmmoe_stable_v1p5/20250814T230612_SN000077_VMamba_epoch0.pth",
                 normalize_output: bool = True,
                 device: str = None,
                 debug: bool = False):
        """Initialize VMMoE transformer"""
        self.checkpoint_path = checkpoint_path
        self.normalize_output = normalize_output
        self.debug = debug
        self.name = "vmmoe"
        self.output_type = "vector"
        
        # Device setup
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        if self.debug:
            print(f"[DEBUG] VMMoE using device: {self.device}")
            print(f"[DEBUG] Checkpoint: {self.checkpoint_path}")
            print(f"[DEBUG] Normalize output: {self.normalize_output}")
        
        # Load model
        try:
            from app.vmmoe.models.factory import load_vmmoe_for_inference
            
            if self.debug:
                print(f"[DEBUG] Loading VMMoE from checkpoint...")
                
            self.model = load_vmmoe_for_inference(
                checkpoint_path=self.checkpoint_path,
                device=str(self.device)
            )
            self.model.eval()
            
            if self.debug:
                print(f"[DEBUG] VMMoE loaded successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load VMMoE model: {e}")
    
    def process(self, vectors: torch.Tensor, metadata: Dict[str, Any] = None) -> torch.Tensor:
        """
        Transform vectors through VMMoE
        
        Args:
            vectors: [N, 768] tensor
            metadata: Optional metadata (unused)
            
        Returns:
            Transformed vectors [N, 768]
        """
        # Ensure correct shape: [B, T, D] where T=1
        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(1)  # [B, 1, 768]
        
        with torch.no_grad():
            try:
                # Ensure vectors are on the same device as the model
                try:
                    model_device = next(self.model.parameters()).device
                except StopIteration:
                    model_device = self.device
                vectors = vectors.to(model_device)
                output = self.model(vectors)
                
                # Handle tuple output
                if isinstance(output, tuple):
                    output = output[0]
                
                # Squeeze time dimension
                if output.dim() == 3:
                    output = output.squeeze(1)  # [B, 768]
                
                # Normalize if requested
                if self.normalize_output:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                
                return output
                
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] VMMoE processing error: {e}")
                raise