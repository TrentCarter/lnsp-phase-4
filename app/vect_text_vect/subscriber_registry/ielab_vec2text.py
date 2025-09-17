#!/usr/bin/env python3
# 20250825T061038_1
"""
IELab Vec2Text subscriber
"""

import torch
import vec2text
import transformers
from typing import List, Dict, Any


class IELabVec2TextSubscriber:
    """Improved vec2text implementation from IELab"""
    
    def __init__(self, 
                 steps: int = 20,
                 beam_width: int = 1,
                 device: str = None,
                 debug: bool = False):
        """Initialize IELab vec2text decoder"""
        self.steps = steps
        self.beam_width = beam_width
        self.debug = debug
        self.name = "ielab"
        self.output_type = "text"
        
        # IELab requires CPU due to MPS compatibility issues
        self.device = torch.device("cpu")
        
        if self.debug:
            print(f"[DEBUG] IELab Vec2Text using device: {self.device} (CPU required)")
            print(f"[DEBUG] Steps: {self.steps}, Beam width: {self.beam_width}")
        
        # Load models
        try:
            if self.debug:
                print("[DEBUG] Loading IELab inversion model...")
                
            self.inversion = vec2text.models.InversionModel.from_pretrained(
                "ielabgroup/vec2text_gtr-base-st_inversion",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,
            )
            
            if self.debug:
                print("[DEBUG] Loading IELab corrector model...")
                
            self.corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
                "ielabgroup/vec2text_gtr-base-st_corrector",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=False,
                device_map=None,
            )
            
            # Move to CPU
            self.inversion.to(self.device)
            self.corrector_model.to(self.device)
            
            # Setup trainers
            self.inversion_trainer = vec2text.trainers.InversionTrainer(
                model=self.inversion,
                train_dataset=None,
                eval_dataset=None,
                data_collator=transformers.DataCollatorForSeq2Seq(
                    self.inversion.tokenizer,
                    label_pad_token_id=-100,
                ),
            )
            
            self.corrector_model.config.dispatch_batches = None
            
            self.corrector = vec2text.trainers.Corrector(
                model=self.corrector_model,
                inversion_trainer=self.inversion_trainer,
                args=None,
                data_collator=vec2text.collator.DataCollatorForCorrection(
                    tokenizer=self.inversion_trainer.model.tokenizer
                ),
            )
            
            if self.debug:
                print("[DEBUG] IELab models loaded successfully")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load IELab models: {e}")
    
    def process(self, vectors: torch.Tensor, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Decode vectors using IELab models
        
        Args:
            vectors: [N, 768] tensor
            metadata: Optional metadata (unused)
            
        Returns:
            List of decoded texts
        """
        # Ensure vectors are on CPU and detached from any GPU computation graph
        vectors = vectors.detach().cpu().float()
        
        try:
            # Set default device to CPU for the operation
            with torch.cuda.device('cpu') if torch.cuda.is_available() else torch.no_grad():
                decoded_texts = vec2text.invert_embeddings(
                    embeddings=vectors,
                    corrector=self.corrector,
                    num_steps=self.steps,
                    sequence_beam_width=self.beam_width
                )
            return decoded_texts
            
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] IELab error details: {e}")
                import traceback
                traceback.print_exc()
            # Return error messages for each vector
            return [f"[IELab decode error: {e}]"] * vectors.shape[0]