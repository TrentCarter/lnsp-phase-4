#!/usr/bin/env python3
"""
TMD Histogram and Analysis Script - Analyzes TMD distribution and identifies misrouted domains.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
sys.path.insert(0, 'src')

try:
    from utils.tmd import unpack_tmd, pack_tmd, format_tmd_code
except ImportError:
    print("Warning: TMD utilities not found. Using mock functions.")
    def unpack_tmd(bits): return (9, 1, 27)  # Mock default
    def pack_tmd(d, t, m): return (d << 12) | (t << 7) | (m << 1)
    def format_tmd_code(bits): return "9.1.27"

# TMD Domain mappings from schema
DOMAINS = {
    0: "Unknown",
    1: "Science",
    2: "Mathematics",
    3: "Technology",
    4: "Engineering",
    5: "Medicine",
    6: "Psychology",
    7: "Philosophy",
    8: "History",
    9: "Literature",
    10: "Art",
    11: "Economics",
    12: "Law",
    13: "Politics",
    14: "Education",
    15: "Environment"
}

TASKS = {
    0: "Unknown",
    1: "Fact Retrieval",
    2: "Definition Matching",
    3: "Analogical Reasoning",
    4: "Causal Inference",
    5: "Classification",
    6: "Entity Recognition",
    7: "Relationship Extraction",
    8: "Schema Adherence",
    9: "Summarization",
    10: "Paraphrasing",
    # ... (abbreviated for space)
}

MODIFIERS = {
    0: "Unknown",
    1: "Biochemical",
    2: "Evolutionary",
    3: "Computational",
    4: "Logical",
    5: "Ethical",
    6: "Historical",
    7: "Legal",
    8: "Philosophical",
    # ... (abbreviated for space)
    27: "Descriptive",
}

class TMDAnalyzer:
    def __init__(self, data_path: str = "data/factoidwiki_1k.jsonl"):
        self.data_path = data_path
        self.chunks = []
        self.tmd_assignments = []
        self.domain_confusion = defaultdict(list)

    def load_data(self):
        """Load chunks and extract/simulate TMD assignments."""
        with open(self.data_path, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks.append(chunk)

                # Extract TMD (would be from actual assignment system)
                tmd = self._extract_or_assign_tmd(chunk)
                self.tmd_assignments.append(tmd)

    def _extract_or_assign_tmd(self, chunk: Dict) -> Tuple[int, int, int]:
        """Extract or assign TMD based on content analysis."""
        content = chunk.get('contents', '').lower()

        # Simple heuristic assignment based on content
        domain = 8  # Default to History
        task = 1    # Default to Fact Retrieval
        modifier = 27  # Default to Descriptive

        # Domain detection heuristics
        if 'album' in content or 'music' in content or 'singer' in content:
            domain = 10  # Art
        elif 'magazine' in content or 'publication' in content:
            domain = 9   # Literature
        elif 'plot' in content or 'novel' in content or 'story' in content:
            domain = 9   # Literature
        elif 'science' in content or 'research' in content:
            domain = 1   # Science
        elif 'technology' in content or 'computer' in content:
            domain = 3   # Technology
        elif 'medicine' in content or 'health' in content:
            domain = 5   # Medicine

        # Task detection (simplified)
        if '?' in content:
            task = 2  # Definition Matching
        elif 'summary' in content:
            task = 9  # Summarization

        return (domain, task, modifier)

    def generate_histogram(self) -> Dict:
        """Generate TMD distribution histogram."""
        domain_counts = Counter()
        task_counts = Counter()
        modifier_counts = Counter()
        full_tmd_counts = Counter()

        for domain, task, modifier in self.tmd_assignments:
            domain_counts[domain] += 1
            task_counts[task] += 1
            modifier_counts[modifier] += 1
            full_tmd = format_tmd_code(pack_tmd(domain, task, modifier))
            full_tmd_counts[full_tmd] += 1

        return {
            'domains': domain_counts,
            'tasks': task_counts,
            'modifiers': modifier_counts,
            'full_codes': full_tmd_counts
        }

    def detect_misrouted(self) -> List[Dict]:
        """Detect potentially misrouted entries based on content vs TMD."""
        misrouted = []

        for i, chunk in enumerate(self.chunks):
            domain, task, modifier = self.tmd_assignments[i]
            content = chunk.get('contents', '').lower()

            # Check for obvious mismatches
            issues = []

            if domain == 8 and ('plot' in content or 'novel' in content):
                issues.append(f"Domain mismatch: History assigned but content suggests Literature")

            if domain == 8 and ('album' in content or 'music' in content):
                issues.append(f"Domain mismatch: History assigned but content suggests Art")

            if issues:
                misrouted.append({
                    'id': chunk.get('id'),
                    'assigned_tmd': format_tmd_code(pack_tmd(domain, task, modifier)),
                    'assigned_domain': DOMAINS.get(domain, 'Unknown'),
                    'issues': issues,
                    'content_preview': content[:100]
                })

        return misrouted

    def print_report(self):
        """Print comprehensive TMD analysis report."""
        histogram = self.generate_histogram()
        misrouted = self.detect_misrouted()

        print("=" * 70)
        print("TMD DISTRIBUTION ANALYSIS")
        print("=" * 70)

        # Domain distribution
        print("\nDOMAIN DISTRIBUTION:")
        print("-" * 40)
        total = len(self.tmd_assignments)
        for domain_id, count in sorted(histogram['domains'].items()):
            domain_name = DOMAINS.get(domain_id, f"Unknown({domain_id})")
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 2)
            print(f"{domain_name:15} {count:4} ({percentage:5.1f}%) {bar}")

        # Task distribution
        print("\nTASK DISTRIBUTION:")
        print("-" * 40)
        for task_id, count in sorted(histogram['tasks'].items())[:5]:  # Top 5
            task_name = TASKS.get(task_id, f"Unknown({task_id})")
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{task_name:20} {count:4} ({percentage:5.1f}%)")

        # Most common full TMD codes
        print("\nTOP TMD CODES:")
        print("-" * 40)
        for tmd_code, count in histogram['full_codes'].most_common(10):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{tmd_code:10} {count:4} ({percentage:5.1f}%)")

        # Default detection
        default_tmd = "9.1.27"  # History.FactRetrieval.Descriptive
        if default_tmd in histogram['full_codes']:
            default_count = histogram['full_codes'][default_tmd]
            default_rate = (default_count / total * 100) if total > 0 else 0
            print(f"\n⚠️  DEFAULT TMD RATE: {default_rate:.1f}% ({default_count}/{total})")
            if default_rate > 20:
                print("   WARNING: High default rate indicates TMD assignment issues!")

        # Misrouted entries
        if misrouted:
            print(f"\n🔄 POTENTIALLY MISROUTED ENTRIES: {len(misrouted)}")
            print("-" * 40)
            for entry in misrouted[:5]:  # Show first 5
                print(f"ID: {entry['id']}")
                print(f"  Assigned: {entry['assigned_domain']} ({entry['assigned_tmd']})")
                print(f"  Issues: {'; '.join(entry['issues'])}")
                print(f"  Content: {entry['content_preview']}...")
                print()

        # Summary statistics
        print("=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        print(f"Total chunks analyzed: {total}")
        print(f"Unique TMD codes: {len(histogram['full_codes'])}")
        print(f"Domain variance: {len(histogram['domains'])} domains used")
        print(f"Task variance: {len(histogram['tasks'])} tasks used")
        print(f"Modifier variance: {len(histogram['modifiers'])} modifiers used")

        # Recommendations
        print("\n📋 RECOMMENDATIONS:")
        print("-" * 40)
        if default_rate > 20:
            print("1. Fix TMD extractor - too many defaults")
        if len(histogram['domains']) < 3:
            print("2. Increase domain variance - limited to {len(histogram['domains'])} domains")
        if len(misrouted) > total * 0.1:
            print("3. Review domain assignment logic - high misrouting rate")
        if len(histogram['full_codes']) < 10:
            print("4. Increase TMD granularity - only {len(histogram['full_codes'])} unique codes")

def main():
    analyzer = TMDAnalyzer()

    print("Loading data...")
    analyzer.load_data()

    print(f"Analyzing {len(analyzer.chunks)} chunks...")
    analyzer.print_report()

if __name__ == "__main__":
    main()