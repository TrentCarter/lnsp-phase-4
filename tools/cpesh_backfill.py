#!/usr/bin/env python3
"""
CPESH Backfill Script - Regenerates CPESH for short chunks by batching them into context windows.
Pass 1: Merge contiguous chunks until 180-320 words
Pass 2: Generate CPESH for merged chunks
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib

class CPESHBackfill:
    def __init__(self,
                 data_path: str = "data/factoidwiki_1k.jsonl",
                 cpesh_output: str = "artifacts/cpesh_active.jsonl",
                 min_words: int = 180,
                 max_words: int = 320,
                 target_words: int = 250):
        self.data_path = data_path
        self.cpesh_output = cpesh_output
        self.min_words = min_words
        self.max_words = max_words
        self.target_words = target_words
        self.chunks = []
        self.merged_chunks = []

    def load_chunks(self):
        """Load all chunks from the dataset."""
        with open(self.data_path, 'r') as f:
            for line in f:
                self.chunks.append(json.loads(line))

    def merge_short_chunks(self) -> List[Dict]:
        """Merge adjacent short chunks until they reach target word count."""
        merged = []
        current_batch = {
            'doc_ids': [],
            'contents': [],
            'word_count': 0,
            'metadata': {}
        }

        for chunk in self.chunks:
            content = chunk.get('contents', '')
            word_count = len(content.split())

            # If adding this chunk would exceed max_words, save current batch
            if current_batch['word_count'] + word_count > self.max_words and current_batch['doc_ids']:
                if current_batch['word_count'] >= self.min_words:
                    merged.append(self._create_merged_entry(current_batch))
                current_batch = {'doc_ids': [], 'contents': [], 'word_count': 0, 'metadata': {}}

            # Add chunk to current batch
            current_batch['doc_ids'].append(chunk.get('id'))
            current_batch['contents'].append(content)
            current_batch['word_count'] += word_count

            # Preserve first chunk's metadata for title extraction
            if not current_batch['metadata'] and 'metadata' in chunk:
                current_batch['metadata'] = chunk['metadata']

            # If batch reaches target size, save it
            if current_batch['word_count'] >= self.target_words:
                merged.append(self._create_merged_entry(current_batch))
                current_batch = {'doc_ids': [], 'contents': [], 'word_count': 0, 'metadata': {}}

        # Save remaining batch if it meets minimum
        if current_batch['doc_ids'] and current_batch['word_count'] >= self.min_words:
            merged.append(self._create_merged_entry(current_batch))

        self.merged_chunks = merged
        return merged

    def _create_merged_entry(self, batch: Dict) -> Dict:
        """Create a merged entry from a batch of chunks."""
        return {
            'merged_id': self._generate_merged_id(batch['doc_ids']),
            'source_ids': batch['doc_ids'],
            'content': ' '.join(batch['contents']),
            'word_count': batch['word_count'],
            'metadata': batch['metadata'],
            'chunk_count': len(batch['doc_ids'])
        }

    def _generate_merged_id(self, doc_ids: List[str]) -> str:
        """Generate a unique ID for merged chunks."""
        combined = '-'.join(doc_ids)
        hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
        return f"merged-{hash_suffix}"

    def generate_cpesh(self, merged_entry: Dict) -> Optional[Dict]:
        """Generate CPESH for a merged chunk (placeholder for actual LLM call)."""
        content = merged_entry['content']

        # Extract key concepts (simplified - would use LLM in production)
        words = content.split()

        # Heuristic concept extraction
        concept = None
        if len(words) > 20:
            # Find the most significant noun phrase (simplified)
            concept = ' '.join(words[:5])  # First 5 words as concept

        # Generate probe question (simplified)
        probe = f"What is {concept}?" if concept else "What is this about?"

        # Expected answer (first sentence as simplified expected)
        sentences = content.split('.')
        expected = sentences[0] if sentences else content[:100]

        # Generate negatives (simplified)
        soft_negative = "Related concept"
        hard_negative = "Different domain concept"

        return {
            'doc_id': merged_entry['merged_id'],
            'source_ids': merged_entry['source_ids'],
            'cpesh': {
                'concept': concept,
                'probe': probe,
                'expected': expected,
                'soft_negative': soft_negative,
                'hard_negative': hard_negative,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'last_accessed': datetime.now(timezone.utc).isoformat(),
                'generation_method': 'backfill_v1'
            },
            'access_count': 0,
            'word_count': merged_entry['word_count'],
            'chunk_count': merged_entry['chunk_count']
        }

    def backfill_cpesh(self):
        """Run the complete backfill process."""
        print(f"Starting CPESH backfill...")
        print(f"Target: {self.min_words}-{self.max_words} words per chunk")

        # Load data
        print("Loading chunks...")
        self.load_chunks()
        print(f"Loaded {len(self.chunks)} chunks")

        # Analyze original chunks
        original_stats = self._analyze_chunks(self.chunks)
        print(f"\nOriginal chunks:")
        print(f"  Mean words: {original_stats['mean']:.1f}")
        print(f"  Min words: {original_stats['min']}")
        print(f"  Max words: {original_stats['max']}")
        print(f"  Short (<120w): {original_stats['short_count']}")

        # Merge short chunks
        print("\nMerging short chunks...")
        self.merge_short_chunks()
        print(f"Created {len(self.merged_chunks)} merged chunks")

        # Analyze merged chunks
        merged_stats = self._analyze_merged_chunks(self.merged_chunks)
        print(f"\nMerged chunks:")
        print(f"  Mean words: {merged_stats['mean']:.1f}")
        print(f"  Min words: {merged_stats['min']}")
        print(f"  Max words: {merged_stats['max']}")
        print(f"  In target range: {merged_stats['in_range']}/{len(self.merged_chunks)}")

        # Generate CPESH for merged chunks
        print("\nGenerating CPESH...")
        cpesh_entries = []
        for i, merged in enumerate(self.merged_chunks):
            if i % 100 == 0:
                print(f"  Processing {i}/{len(self.merged_chunks)}...")

            cpesh = self.generate_cpesh(merged)
            if cpesh:
                cpesh_entries.append(cpesh)

        # Save CPESH entries
        print(f"\nSaving {len(cpesh_entries)} CPESH entries to {self.cpesh_output}")
        with open(self.cpesh_output, 'w') as f:
            for entry in cpesh_entries:
                f.write(json.dumps(entry) + '\n')

        print(f"\n✅ Backfill complete!")
        print(f"  Generated {len(cpesh_entries)} CPESH entries")
        print(f"  Coverage: {len(cpesh_entries)}/{len(self.merged_chunks)} ({100*len(cpesh_entries)/len(self.merged_chunks):.1f}%)")

    def _analyze_chunks(self, chunks: List[Dict]) -> Dict:
        """Analyze word count statistics for original chunks."""
        word_counts = []
        short_count = 0

        for chunk in chunks:
            content = chunk.get('contents', '')
            words = len(content.split())
            word_counts.append(words)
            if words < 120:
                short_count += 1

        return {
            'mean': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min': min(word_counts) if word_counts else 0,
            'max': max(word_counts) if word_counts else 0,
            'short_count': short_count
        }

    def _analyze_merged_chunks(self, merged: List[Dict]) -> Dict:
        """Analyze word count statistics for merged chunks."""
        word_counts = [m['word_count'] for m in merged]
        in_range = sum(1 for wc in word_counts if self.min_words <= wc <= self.max_words)

        return {
            'mean': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min': min(word_counts) if word_counts else 0,
            'max': max(word_counts) if word_counts else 0,
            'in_range': in_range
        }

def main():
    backfiller = CPESHBackfill()
    backfiller.backfill_cpesh()

if __name__ == "__main__":
    main()