#!/usr/bin/env python3
"""
Data Migration Script: Apply all S3 fixes to existing data.

This script applies the root cause fixes:
1. Re-chunk with proper 180-320 word sizing
2. Generate missing CPESH with correct keys
3. Re-extract TMD using real content analysis
4. Validate output quality

Usage:
    python scripts/migrate_to_v2.py [--input INPUT_FILE] [--output OUTPUT_FILE]
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chunker_v2 import create_chunks, merge_short_chunks, analyze_chunks
from tmd_extractor_v2 import extract_tmd_from_text, analyze_tmd_distribution
from cpesh_fixer import generate_cpesh_context, analyze_key_alignment


def load_original_data(input_file: str) -> List[Dict[str, Any]]:
    """Load original problematic data."""
    chunks = []

    if not Path(input_file).exists():
        print(f"Warning: {input_file} not found")
        return chunks

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue

    return chunks


def rechunk_data(original_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Re-chunk data with proper sizing."""
    rechunked = []

    print("Re-chunking data with 180-320 word targets...")

    for orig_chunk in original_chunks:
        text = orig_chunk.get('text', orig_chunk.get('content', ''))
        if not text:
            continue

        # Create proper chunks from this text
        new_chunks = create_chunks(text, min_words=180, max_words=320)

        for new_chunk in new_chunks:
            # Preserve original metadata
            rechunked_chunk = {
                'text': new_chunk['text'],
                'word_count': new_chunk['word_count'],
                'chunk_id': new_chunk['chunk_id'],
                'chunk_index': len(rechunked),
                'source_id': orig_chunk.get('id', orig_chunk.get('source_id', '')),
                'original_chunk_id': orig_chunk.get('chunk_id', orig_chunk.get('id')),
                'migration_version': '2.0'
            }

            # Preserve any existing metadata
            for key in ['doc_id', 'title', 'url', 'timestamp', 'domain']:
                if key in orig_chunk:
                    rechunked_chunk[key] = orig_chunk[key]

            rechunked.append(rechunked_chunk)

    return rechunked


def apply_tmd_extraction(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply new TMD extraction to all chunks."""
    print("Applying TMD extraction v2...")

    for chunk in chunks:
        text = chunk.get('text', '')
        if text:
            tmd_result = extract_tmd_from_text(text)

            # Add TMD data to chunk
            chunk.update({
                'tmd_code': f"{tmd_result['domain_code']}.{tmd_result['task_code']}.{tmd_result['modifier_code']}",
                'domain_code': tmd_result['domain_code'],
                'task_code': tmd_result['task_code'],
                'modifier_code': tmd_result['modifier_code'],
                'domain': tmd_result['domain'],
                'task': tmd_result['task'],
                'modifier': tmd_result['modifier'],
                'tmd_confidence': tmd_result['confidence']
            })

    return chunks


def apply_cpesh_generation(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate CPESH contextual data for all chunks."""
    print("Generating CPESH contextual data...")

    for chunk in chunks:
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', '')

        if text and chunk_id:
            cpesh_data = generate_cpesh_context(text, chunk_id)
            chunk['cpesh'] = cpesh_data

    return chunks


def validate_migration(
    original_chunks: List[Dict[str, Any]],
    migrated_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Validate the migration results."""
    print("Validating migration results...")

    # Analyze chunk quality
    original_stats = analyze_chunks(original_chunks)
    migrated_stats = analyze_chunks(migrated_chunks)

    # Analyze TMD distribution
    tmd_results = []
    for chunk in migrated_chunks:
        if 'tmd_code' in chunk:
            tmd_results.append({
                'domain_code': chunk['domain_code'],
                'task_code': chunk['task_code'],
                'modifier_code': chunk['modifier_code'],
                'domain': chunk['domain'],
                'task': chunk['task'],
                'modifier': chunk['modifier'],
                'confidence': chunk.get('tmd_confidence', 0)
            })

    tmd_analysis = analyze_tmd_distribution(tmd_results)

    # Analyze CPESH coverage
    cpesh_analysis = analyze_key_alignment(migrated_chunks)

    # Calculate improvements
    chunk_improvement = {
        'mean_words': migrated_stats['mean_words'] - original_stats['mean_words'],
        'target_range_pct': migrated_stats['target_pct'] - original_stats['target_pct'],
        'short_chunks_reduction': original_stats['short_chunks'] - migrated_stats['short_chunks']
    }

    validation_report = {
        'migration_summary': {
            'original_chunks': len(original_chunks),
            'migrated_chunks': len(migrated_chunks),
            'chunk_change': len(migrated_chunks) - len(original_chunks)
        },
        'chunk_quality': {
            'before': original_stats,
            'after': migrated_stats,
            'improvement': chunk_improvement
        },
        'tmd_diversity': tmd_analysis,
        'cpesh_coverage': cpesh_analysis,
        'validation_status': 'PASS' if (
            migrated_stats['mean_words'] >= 180 and
            migrated_stats['target_pct'] >= 80 and
            tmd_analysis['unique_domains'] >= 5 and
            cpesh_analysis['alignment_rate'] >= 90
        ) else 'FAIL'
    }

    return validation_report


def save_results(chunks: List[Dict[str, Any]], output_file: str) -> None:
    """Save migrated chunks to output file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Migrate data with S3 fixes')
    parser.add_argument('--input', default='artifacts/cpesh_active.jsonl',
                       help='Input file path')
    parser.add_argument('--output', default='artifacts/cpesh_active_v2.jsonl',
                       help='Output file path')
    parser.add_argument('--report', default='artifacts/migration_report_v2.json',
                       help='Report file path')

    args = parser.parse_args()

    print("=== LNSP Data Migration to V2 ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print("-" * 50)

    # Step 1: Load original data
    print("Step 1: Loading original data...")
    original_chunks = load_original_data(args.input)
    print(f"Loaded {len(original_chunks)} original chunks")

    if not original_chunks:
        print("❌ No data to migrate")
        return

    # Step 2: Re-chunk with proper sizing
    print("\nStep 2: Re-chunking with 180-320 word targets...")
    rechunked = rechunk_data(original_chunks)
    print(f"Created {len(rechunked)} properly-sized chunks")

    # Step 3: Apply TMD extraction v2
    print("\nStep 3: Applying TMD extraction v2...")
    with_tmd = apply_tmd_extraction(rechunked)
    print(f"Applied TMD codes to {len(with_tmd)} chunks")

    # Step 4: Generate CPESH contextual data
    print("\nStep 4: Generating CPESH contextual data...")
    with_cpesh = apply_cpesh_generation(with_tmd)
    print(f"Generated CPESH for {len(with_cpesh)} chunks")

    # Step 5: Validate results
    print("\nStep 5: Validating migration...")
    validation_report = validate_migration(original_chunks, with_cpesh)

    # Step 6: Save results
    print(f"\nStep 6: Saving results to {args.output}...")
    save_results(with_cpesh, args.output)

    # Save validation report
    with open(args.report, 'w') as f:
        json.dump(validation_report, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("MIGRATION SUMMARY")
    print("=" * 50)

    status = validation_report['validation_status']
    print(f"Status: {status}")

    chunk_quality = validation_report['chunk_quality']
    print(f"Chunks: {len(original_chunks)} → {len(with_cpesh)}")
    print(f"Mean words: {chunk_quality['before']['mean_words']:.1f} → {chunk_quality['after']['mean_words']:.1f}")
    print(f"Target range: {chunk_quality['before']['target_pct']:.1f}% → {chunk_quality['after']['target_pct']:.1f}%")

    tmd_diversity = validation_report['tmd_diversity']
    print(f"TMD diversity: {tmd_diversity['unique_domains']} domains, {tmd_diversity['unique_tasks']} tasks")

    cpesh_coverage = validation_report['cpesh_coverage']
    print(f"CPESH coverage: {cpesh_coverage['alignment_rate']}%")

    if status == 'PASS':
        print("\n✅ Migration successful! Data quality targets met.")
        print(f"Ready for production use: {args.output}")
    else:
        print("\n⚠️  Migration completed with quality warnings.")
        print("Review validation report for details.")

    print(f"\nDetailed report: {args.report}")


if __name__ == "__main__":
    main()