#!/usr/bin/env python3
"""
Download lightweight datasets for Nemotron-VMMoE initial testing.
Focuses on smaller, high-quality datasets that can be downloaded quickly.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    print("Installing datasets...")
    os.system(f"{sys.executable} -m pip install datasets")
    sys.exit("Please restart after installation")


class LightweightDatasetDownloader:
    """Download smaller datasets for quick testing."""
    
    DATASETS = {
        # Small but high-quality datasets
        "alpaca": {
            "repo": "tatsu-lab/alpaca",
            "split": "train",
            "description": "52K instruction-following examples",
            "max_examples": None  # Take all (52k)
        },
        "gsm8k": {
            "repo": "openai/gsm8k", 
            "config": "main",
            "split": "train",
            "description": "7.5K grade school math problems",
            "max_examples": None  # Take all
        },
        "dolly": {
            "repo": "databricks/databricks-dolly-15k",
            "split": "train", 
            "description": "15K human-generated instructions",
            "max_examples": None  # Take all
        },
        "tiny_stories": {
            "repo": "roneneldan/TinyStories",
            "split": "train[:50000]",
            "description": "Simple coherent stories for training",
            "max_examples": 50000
        },
        "hellaswag": {
            "repo": "Rowan/hellaswag",
            "split": "train[:10000]",
            "description": "Commonsense reasoning",
            "max_examples": 10000
        }
    }
    
    def __init__(self, output_dir: str = "data/nemotron_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {}
        
    def download_dataset(self, name: str, info: Dict) -> bool:
        """Download a single dataset."""
        output_path = self.output_dir / name
        
        if output_path.exists():
            print(f"⏭️  {name} already exists")
            # Still count it
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(str(output_path))
                self.stats[name] = len(dataset)
            except:
                self.stats[name] = "existing"
            return True
            
        print(f"\n📥 Downloading {name}: {info['description']}")
        
        try:
            # Download dataset
            kwargs = {
                "path": info['repo'],
                "split": info['split'],
                "trust_remote_code": True
            }
            
            if 'config' in info and info['config']:
                kwargs['name'] = info['config']
                
            dataset = load_dataset(**kwargs)
            
            # Save to disk
            dataset.save_to_disk(str(output_path))
            self.stats[name] = len(dataset)
            
            print(f"✅ Downloaded {len(dataset):,} examples")
            return True
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            self.stats[name] = "failed"
            return False
    
    def create_sample_data(self):
        """Create sample data for immediate testing."""
        samples_path = self.output_dir / "sample_instructions.json"
        
        samples = [
            # Math examples
            {
                "instruction": "Solve: 3x + 7 = 22",
                "response": "To solve 3x + 7 = 22:\n1. Subtract 7 from both sides: 3x = 15\n2. Divide by 3: x = 5",
                "category": "math"
            },
            {
                "instruction": "What is 15% of 240?",
                "response": "15% of 240 = 0.15 × 240 = 36",
                "category": "math"
            },
            
            # Code examples
            {
                "instruction": "Write a Python function to check if a number is prime",
                "response": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
                "category": "code"
            },
            {
                "instruction": "Implement binary search in Python",
                "response": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                "category": "code"
            },
            
            # Reasoning examples
            {
                "instruction": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                "response": "No, we cannot conclude that some roses fade quickly. While all roses are flowers, the statement only says 'some flowers' fade quickly. These quick-fading flowers might be other types (like tulips or daisies) and not roses at all.",
                "category": "reasoning"
            },
            {
                "instruction": "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
                "response": "Let the ball cost x dollars. Then the bat costs (x + 1) dollars.\nTotal: x + (x + 1) = 1.10\n2x + 1 = 1.10\n2x = 0.10\nx = 0.05\nThe ball costs $0.05 (5 cents).",
                "category": "reasoning"
            }
        ]
        
        with open(samples_path, 'w') as f:
            json.dump(samples, f, indent=2)
            
        print(f"✅ Created {len(samples)} sample instructions")
        self.stats['samples'] = len(samples)
    
    def download_all(self):
        """Download all lightweight datasets."""
        print("=" * 60)
        print("🚀 LIGHTWEIGHT DATASET DOWNLOADER")
        print("=" * 60)
        
        # Download datasets
        for name, info in self.DATASETS.items():
            self.download_dataset(name, info)
        
        # Create samples
        self.create_sample_data()
        
        # Create metadata file
        metadata = {
            "datasets": self.stats,
            "total_examples": sum(v for v in self.stats.values() if isinstance(v, int)),
            "download_date": str(Path.cwd()),
            "purpose": "Nemotron-VMMoE training"
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        
        total = sum(v for v in self.stats.values() if isinstance(v, int))
        print(f"\nTotal examples: {total:,}")
        
        for name, count in self.stats.items():
            if isinstance(count, int):
                print(f"  ✅ {name}: {count:,} examples")
            else:
                print(f"  ⚠️  {name}: {count}")
        
        print("\n💡 Next Steps:")
        print("1. Convert to vectors: python scripts/convert_to_vectors.py")
        print("2. Start training: python train_nemotron_vmmoe.py")
        print("\n📝 Datasets provide coverage for:")
        print("  • Instruction following (Alpaca, Dolly)")
        print("  • Mathematical reasoning (GSM8K)")
        print("  • Language modeling (TinyStories)")
        print("  • Commonsense reasoning (HellaSwag)")


if __name__ == "__main__":
    downloader = LightweightDatasetDownloader()
    downloader.download_all()