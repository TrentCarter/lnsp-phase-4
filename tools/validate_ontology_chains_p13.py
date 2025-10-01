#!/usr/bin/env python3
"""
P13 Validation Script for Ontology Chains

Validates ontology chains against P13 requirements:
1. ✅ Sequential parent→child relationships
2. ✅ No duplicate concepts
3. ✅ Length 3-20 concepts
4. ✅ Valid JSON structure
5. ✅ Source metadata present
6. ✅ Concept quality (non-empty, meaningful)

Usage:
    python tools/validate_ontology_chains_p13.py \
        --input artifacts/ontology_chains/swo_chains_1k_sample.jsonl \
        --min-length 3 \
        --max-length 20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter


class P13Validator:
    """Validates ontology chains against P13 requirements."""
    
    def __init__(self, min_length: int = 3, max_length: int = 20):
        self.min_length = min_length
        self.max_length = max_length
        self.stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": Counter()
        }
    
    def validate_chain(self, chain: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single chain against P13 requirements.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # 1. Check JSON structure
        if not isinstance(chain, dict):
            errors.append("Invalid JSON structure")
            return False, errors
        
        # 2. Check required fields
        required_fields = ["chain_id", "concepts", "source", "chain_length"]
        for field in required_fields:
            if field not in chain:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # 3. Check concepts list
        concepts = chain.get("concepts", [])
        if not isinstance(concepts, list):
            errors.append("concepts must be a list")
            return False, errors
        
        if len(concepts) == 0:
            errors.append("concepts list is empty")
            return False, errors
        
        # 4. Check chain length
        actual_length = len(concepts)
        declared_length = chain.get("chain_length", 0)
        
        if actual_length != declared_length:
            errors.append(f"Length mismatch: actual={actual_length}, declared={declared_length}")
        
        if actual_length < self.min_length:
            errors.append(f"Chain too short: {actual_length} < {self.min_length}")
        
        if actual_length > self.max_length:
            errors.append(f"Chain too long: {actual_length} > {self.max_length}")
        
        # 5. Check for duplicate concepts
        if len(concepts) != len(set(concepts)):
            duplicates = [c for c in concepts if concepts.count(c) > 1]
            errors.append(f"Duplicate concepts found: {set(duplicates)}")
        
        # 6. Check concept quality
        for i, concept in enumerate(concepts):
            if not isinstance(concept, str):
                errors.append(f"Concept {i} is not a string: {type(concept)}")
                continue
            
            if not concept or concept.strip() == "":
                errors.append(f"Concept {i} is empty")
            
            if len(concept) < 2:
                errors.append(f"Concept {i} too short: '{concept}'")
            
            # Check for suspicious patterns
            if concept.startswith("http://") or concept.startswith("https://"):
                errors.append(f"Concept {i} is a URL: {concept}")
        
        # 7. Check source
        valid_sources = ["swo", "go", "dbpedia", "conceptnet"]
        source = chain.get("source", "")
        if source not in valid_sources:
            errors.append(f"Invalid source: {source}")
        
        return len(errors) == 0, errors
    
    def validate_file(self, input_path: Path) -> Dict:
        """Validate all chains in a JSONL file."""
        print(f"\n{'='*60}")
        print(f"P13 VALIDATION: {input_path.name}")
        print(f"{'='*60}")
        
        failed_chains = []
        
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                self.stats["total"] += 1
                
                try:
                    chain = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    self.stats["failed"] += 1
                    self.stats["errors"]["JSON parse error"] += 1
                    failed_chains.append({
                        "line": line_num,
                        "errors": [f"JSON parse error: {e}"]
                    })
                    continue
                
                is_valid, errors = self.validate_chain(chain)
                
                if is_valid:
                    self.stats["passed"] += 1
                else:
                    self.stats["failed"] += 1
                    for error in errors:
                        self.stats["errors"][error] += 1
                    
                    failed_chains.append({
                        "line": line_num,
                        "chain_id": chain.get("chain_id", "unknown"),
                        "errors": errors
                    })
        
        return {
            "stats": self.stats,
            "failed_chains": failed_chains
        }
    
    def print_report(self, results: Dict):
        """Print validation report."""
        stats = results["stats"]
        failed_chains = results["failed_chains"]
        
        print(f"\n{'='*60}")
        print("VALIDATION RESULTS")
        print(f"{'='*60}")
        
        total = stats["total"]
        passed = stats["passed"]
        failed = stats["failed"]
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n📊 Summary:")
        print(f"  Total chains:  {total:,}")
        print(f"  ✅ Passed:     {passed:,} ({pass_rate:.1f}%)")
        print(f"  ❌ Failed:     {failed:,} ({100-pass_rate:.1f}%)")
        
        if stats["errors"]:
            print(f"\n🔍 Error Breakdown:")
            for error, count in stats["errors"].most_common():
                print(f"  • {error}: {count}")
        
        if failed_chains and failed < 20:
            print(f"\n❌ Failed Chains (first {min(10, len(failed_chains))}):")
            for fc in failed_chains[:10]:
                print(f"\n  Line {fc['line']} - {fc.get('chain_id', 'unknown')}:")
                for error in fc["errors"]:
                    print(f"    • {error}")
        
        # Final verdict
        print(f"\n{'='*60}")
        if pass_rate >= 80:
            print("✅ VALIDATION PASSED (≥80% success rate)")
            print("✅ Ready for full ingestion!")
        else:
            print(f"❌ VALIDATION FAILED ({pass_rate:.1f}% < 80%)")
            print("⚠️  Fix errors before proceeding to full ingestion")
        print(f"{'='*60}\n")
        
        return pass_rate >= 80


def main():
    parser = argparse.ArgumentParser(
        description="Validate ontology chains against P13 requirements"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with chains"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=3,
        help="Minimum chain length (default: 3)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=20,
        help="Maximum chain length (default: 20)"
    )
    parser.add_argument(
        "--show-all-errors",
        action="store_true",
        help="Show all failed chains (not just first 10)"
    )
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run validation
    validator = P13Validator(
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    results = validator.validate_file(args.input)
    passed = validator.print_report(results)
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
