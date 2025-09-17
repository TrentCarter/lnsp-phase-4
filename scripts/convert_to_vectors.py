# 20250903T1810_v1.0
import argparse
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Iterator

import numpy as np
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Text Extraction Logic ---
# Each function takes a dataset example (a dict) and returns a single string of text.

def get_text_from_alpaca(example: Dict) -> str:
    """Extracts text from the Alpaca dataset format."""
    instruction = example.get('instruction', '')
    inp = example.get('input', '')
    response = example.get('output', '')
    if inp:
        return f"{instruction}\n{inp}\n{response}"
    return f"{instruction}\n{response}"

def get_text_from_gsm8k(example: Dict) -> str:
    """Extracts text from the GSM8K dataset."""
    return example.get('question', '') + '\n' + example.get('answer', '')

def get_text_from_dolly(example: Dict) -> str:
    """Extracts text from the Dolly dataset."""
    return f"{example.get('instruction', '')}\n{example.get('context', '')}\n{example.get('response', '')}"

def get_text_from_tinystories(example: Dict) -> str:
    """Extracts text from the TinyStories dataset."""
    return example.get('text', '')

def get_text_from_hellaswag(example: Dict) -> str:
    """Extracts text from HellaSwag by combining context and the correct ending."""
    ctx = example.get('ctx', '')
    endings = example.get('endings', [])
    label = int(example.get('label', 0))
    if endings and 0 <= label < len(endings):
        return f"{ctx} {endings[label]}"
    return ctx

# --- Dataset Configuration ---
DATASET_PROCESSORS = {
    'alpaca': get_text_from_alpaca,
    'gsm8k': get_text_from_gsm8k,
    'dolly': get_text_from_dolly,
    'tiny_stories': get_text_from_tinystories,
    'hellaswag': get_text_from_hellaswag,
}

class VectorConverter:
    """Orchestrates the text-to-vector conversion process."""

    def __init__(self, input_dir: Path, output_dir: Path, chunk_size: int, model_name: str = 'sentence-transformers/gtr-t5-base'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        logging.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into phrases of N words."""
        words = text.split()
        return [' '.join(words[i:i + self.chunk_size]) for i in range(0, len(words), self.chunk_size)]

    def process_dataset(self, dataset_name: str, processor_fn):
        """Loads, chunks, and converts a single dataset."""
        dataset_path = self.input_dir / dataset_name
        if not dataset_path.exists():
            logging.warning(f"Dataset '{dataset_name}' not found at {dataset_path}. Skipping.")
            return

        logging.info(f"Processing dataset: {dataset_name}")
        dataset = load_from_disk(str(dataset_path))

        all_vectors = []
        for example in tqdm(dataset, desc=f"Converting {dataset_name}"):
            text = processor_fn(example)
            if not text:
                continue
            
            chunks = self.chunk_text(text)
            if chunks:
                vectors = self.model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
                all_vectors.append(vectors.cpu().numpy())

        if not all_vectors:
            logging.warning(f"No vectors were generated for {dataset_name}. Skipping save.")
            return

        # Save vectors
        output_subdir = self.output_dir / dataset_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_file = output_subdir / 'vectors.npy'
        
        # Note: This saves a list of numpy arrays, which is not ideal for large scale use.
        # A more robust solution would concatenate and save as a single large array with an index.
        # For this MVP, saving as a .npy of objects is acceptable.
        np.save(output_file, np.array(all_vectors, dtype=object), allow_pickle=True)
        logging.info(f"Saved {len(all_vectors)} vector sequences to {output_file}")

    def run(self):
        """Runs the conversion for all configured datasets."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Starting vector conversion. Output will be saved to {self.output_dir}")

        for name, processor in DATASET_PROCESSORS.items():
            self.process_dataset(name, processor)

        logging.info("Vector conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert text datasets to phrase-level vectors.")
    parser.add_argument('--input-dir', type=str, default='data/nemotron_datasets', help='Directory containing the downloaded Hugging Face datasets.')
    parser.add_argument('--output-dir', type=str, default='data/nemotron_vectors', help='Directory to save the output vector files.')
    parser.add_argument('--chunk-size', type=int, default=4, help='Number of words per vector chunk.')
    args = parser.parse_args()

    converter = VectorConverter(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        chunk_size=args.chunk_size
    )
    converter.run()

if __name__ == "__main__":
    main()
