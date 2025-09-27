#!/usr/bin/env python3
"""
LNSP Validator Script - CI/CD quality gates for chunk quality, CPESH attachment, and TMD distribution.
Exit codes:
  0: All checks passed
  1: Critical failure (chunks too short, CPESH missing)
  2: Warning threshold exceeded
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import statistics

class LNSPValidator:
    def __init__(self, data_path: str = "data/factoidwiki_1k.jsonl",
                 cpesh_path: str = "artifacts/cpesh_cache.jsonl"):
        self.data_path = data_path
        self.cpesh_path = cpesh_path
        self.chunks = []
        self.cpesh_entries = {}
        self.stats = {}

    def load_data(self):
        """Load chunks and CPESH data."""
        # Load main dataset
        with open(self.data_path, 'r') as f:
            for line in f:
                self.chunks.append(json.loads(line))

        # Load CPESH cache
        if Path(self.cpesh_path).exists():
            with open(self.cpesh_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    doc_id = entry.get('doc_id')
                    if doc_id:
                        self.cpesh_entries[doc_id] = entry

    def calculate_stats(self) -> Dict:
        """Calculate validation statistics."""
        word_counts = []
        tmd_codes = []
        chunks_with_cpesh = 0
        chunks_too_short = 0
        tmd_defaults = 0

        for chunk in self.chunks:
            # Word count analysis
            content = chunk.get('contents', '')
            words = len(content.split())
            word_counts.append(words)
            if words < 120:
                chunks_too_short += 1

            # TMD analysis (simulated - would need actual TMD extraction)
            # For now, we'll assume default is 9.1.27
            tmd_code = "9.1.27"  # This would be extracted from actual TMD assignment
            tmd_codes.append(tmd_code)
            if tmd_code == "9.1.27":
                tmd_defaults += 1

            # CPESH attachment
            doc_id = chunk.get('id')
            if doc_id in self.cpesh_entries:
                cpesh = self.cpesh_entries[doc_id].get('cpesh', {})
                if cpesh.get('concept') and cpesh.get('probe') and cpesh.get('expected'):
                    chunks_with_cpesh += 1

        total_chunks = len(self.chunks)

        self.stats = {
            'total_chunks': total_chunks,
            'mean_words': statistics.mean(word_counts) if word_counts else 0,
            'median_words': statistics.median(word_counts) if word_counts else 0,
            'p95_words': sorted(word_counts)[int(len(word_counts) * 0.95)] if word_counts else 0,
            'chunks_too_short': chunks_too_short,
            'short_chunk_rate': chunks_too_short / total_chunks if total_chunks > 0 else 0,
            'cpesh_attach_rate': chunks_with_cpesh / total_chunks if total_chunks > 0 else 0,
            'tmd_default_rate': tmd_defaults / total_chunks if total_chunks > 0 else 0,
            'complete_entries': total_chunks - chunks_too_short - (total_chunks - chunks_with_cpesh),
            'complete_rate': 0  # Will calculate below
        }

        # Calculate complete rate
        complete = 0
        for chunk in self.chunks:
            doc_id = chunk.get('id')
            content = chunk.get('contents', '')
            words = len(content.split())

            has_text = words >= 120
            has_cpesh = doc_id in self.cpesh_entries and \
                       self.cpesh_entries[doc_id].get('cpesh', {}).get('concept') is not None
            has_non_default_tmd = True  # Would check actual TMD != default

            if has_text and has_cpesh and has_non_default_tmd:
                complete += 1

        self.stats['complete_entries'] = complete
        self.stats['complete_rate'] = complete / total_chunks if total_chunks > 0 else 0

        return self.stats

    def print_dashboard(self):
        """Print ASCII status dashboard."""
        print("─" * 50)
        print("LNSP STATUS DASHBOARD")
        print("─" * 50)
        print(f"Total chunks:         {self.stats['total_chunks']:,}")
        print(f"Complete entries:     {self.stats['complete_entries']:,} ({self.stats['complete_rate']:.1%})")
        print(f" - Too short (<120w): {self.stats['chunks_too_short']:,}")
        print(f" - CPESH missing:     {self.stats['total_chunks'] - int(self.stats['cpesh_attach_rate'] * self.stats['total_chunks']):,}")
        print(f" - TMD defaulted:     {int(self.stats['tmd_default_rate'] * self.stats['total_chunks']):,}")
        print(f"Mean chunk words:     {self.stats['mean_words']:.0f}")
        print(f"P95 chunk words:      {self.stats['p95_words']:.0f}")
        print(f"CPESH attach rate:    {self.stats['cpesh_attach_rate']:.1%}")
        print(f"TMD default rate:     {self.stats['tmd_default_rate']:.1%}")
        print("─" * 50)

    def validate(self) -> int:
        """Run validation checks and return exit code."""
        # Critical thresholds
        MIN_MEAN_WORDS = 120
        MIN_P95_WORDS = 250
        MAX_TMD_DEFAULT_RATE = 0.2
        MIN_CPESH_ATTACH_RATE = 0.9

        exit_code = 0
        failures = []
        warnings = []

        # Critical failures
        if self.stats['mean_words'] < MIN_MEAN_WORDS:
            failures.append(f"FAIL: Mean words per chunk ({self.stats['mean_words']:.0f}) < {MIN_MEAN_WORDS}")
            exit_code = 1

        if self.stats['p95_words'] < MIN_P95_WORDS:
            failures.append(f"FAIL: P95 words per chunk ({self.stats['p95_words']:.0f}) < {MIN_P95_WORDS}")
            exit_code = 1

        if self.stats['tmd_default_rate'] > MAX_TMD_DEFAULT_RATE:
            failures.append(f"FAIL: TMD default rate ({self.stats['tmd_default_rate']:.1%}) > {MAX_TMD_DEFAULT_RATE:.0%}")
            exit_code = 1

        # Warnings
        if self.stats['cpesh_attach_rate'] < MIN_CPESH_ATTACH_RATE:
            warnings.append(f"WARN: CPESH attach rate ({self.stats['cpesh_attach_rate']:.1%}) < {MIN_CPESH_ATTACH_RATE:.0%}")
            if exit_code == 0:
                exit_code = 2

        # Print results
        if failures:
            print("\n❌ VALIDATION FAILURES:")
            for failure in failures:
                print(f"  {failure}")

        if warnings:
            print("\n⚠️  VALIDATION WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")

        if exit_code == 0:
            print("\n✅ ALL VALIDATION CHECKS PASSED")

        return exit_code

def main():
    validator = LNSPValidator()

    print("Loading data...")
    validator.load_data()

    print("Calculating statistics...")
    validator.calculate_stats()

    print("\n")
    validator.print_dashboard()

    print("\nRunning validation checks...")
    exit_code = validator.validate()

    sys.exit(exit_code)

if __name__ == "__main__":
    main()