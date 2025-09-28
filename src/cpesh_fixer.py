#!/usr/bin/env python3
"""
CPESH Fixer: Fix key alignment and generate missing CPESH data.

This addresses the 98.4% missing CPESH issue by ensuring proper key generation
and creating contextual enrichment for all chunks.
"""

import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


def normalize_text_for_key(text: str) -> str:
    """Normalize text for consistent key generation."""
    # Remove extra whitespace and normalize
    text = ' '.join(text.strip().split())
    # Remove special characters that might cause key mismatches
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()


def generate_chunk_key(text: str, index: int = 0) -> str:
    """Generate consistent chunk key from text."""
    normalized = normalize_text_for_key(text)
    content = f"{index}:{normalized[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def extract_contextual_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key terms for contextual enrichment."""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'its',
        'he', 'she', 'they', 'we', 'you', 'i', 'me', 'him', 'her', 'them',
        'us', 'my', 'your', 'his', 'her', 'their', 'our'
    }

    # Extract words, filter stop words, keep significant terms
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]

    # Count frequency and take most common
    from collections import Counter
    word_counts = Counter(keywords)

    return [word for word, count in word_counts.most_common(max_keywords)]


def generate_cpesh_context(text: str, chunk_id: str) -> Dict[str, Any]:
    """Generate CPESH contextual enrichment for a text chunk."""
    from datetime import datetime, timezone

    # Extract sentences for better CPESH generation
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Extract concept (main entity/topic)
    concept = lines[0][:200] if lines else sentences[0][:200] if sentences else text[:200]

    # Build probe question
    if concept:
        tokens = concept.split()
        if tokens and all(tok[:1].isupper() for tok in tokens[:2]):
            probe = f"Who is {concept}?"
        elif re.search(r'\b(year|when|date)\b', concept.lower()):
            probe = f"When did {concept} occur?"
        else:
            probe = f"What is {concept}?"
    else:
        probe = "What is the main concept described?"

    # Select expected answer (most relevant sentence)
    expected = sentences[0] if sentences else concept
    for sentence in sentences:
        if len(set(sentence.lower().split()) & set(concept.lower().split())) > 3:
            expected = sentence
            break

    # Generate soft negative (related but incorrect)
    soft_negative = None
    for sentence in sentences[1:]:
        if sentence != expected and len(sentence.split()) >= 4:
            soft_negative = sentence
            break
    if not soft_negative:
        soft_negative = f"A related fact from the source does not answer the probe about {concept[:50]}"

    # Hard negative pool (unrelated facts)
    hard_negatives = [
        "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
        "Binary search runs in logarithmic time on sorted collections.",
        "Quantum entanglement links particle states regardless of distance.",
        "The Krebs cycle generates ATP by oxidizing acetyl-CoA in mitochondria.",
        "Plate tectonics explains how Earth's crustal plates shift across the mantle.",
        "SNMP is a network protocol for monitoring and managing devices."
    ]
    # Select a hard negative based on concept hash
    hard_negative = hard_negatives[hash(concept) % len(hard_negatives)]

    # Return proper CPESH structure
    return {
        'concept': concept,
        'probe': probe,
        'expected': expected,
        'soft_negative': soft_negative,
        'hard_negative': hard_negative,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'last_accessed': datetime.now(timezone.utc).isoformat(),
        'generation_method': 'heuristic_v1',
        'insufficient_evidence': False
    }


def generate_cpesh_context_old(text: str, chunk_id: str) -> Dict[str, Any]:
    """Old version - kept for backward compatibility."""
    keywords = extract_contextual_keywords(text)

    # Extract entities (simple pattern matching)
    entities = {
        'dates': re.findall(r'\b\d{4}\b', text),
        'numbers': re.findall(r'\b\d+(?:\.\d+)?\b', text),
        'proper_nouns': re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    }

    # Generate semantic tags based on content
    semantic_tags = []
    if any(word in text.lower() for word in ['technology', 'computer', 'software']):
        semantic_tags.append('technology')
    if any(word in text.lower() for word in ['history', 'historical', 'century']):
        semantic_tags.append('history')
    if any(word in text.lower() for word in ['science', 'research', 'study']):
        semantic_tags.append('science')
    if any(word in text.lower() for word in ['art', 'artist', 'painting']):
        semantic_tags.append('art')

    # Create contextual relationships
    relationships = []
    if entities['dates']:
        relationships.append({
            'type': 'temporal',
            'values': entities['dates'][:3]  # Limit to 3 dates
        })
    if entities['proper_nouns']:
        relationships.append({
            'type': 'entity',
            'values': entities['proper_nouns'][:5]  # Limit to 5 entities
        })

    return {
        'chunk_id': chunk_id,
        'keywords': keywords,
        'entities': entities,
        'semantic_tags': semantic_tags,
        'relationships': relationships,
        'text_length': len(text),
        'word_count': len(text.split()),
        'enrichment_version': '2.0'
    }


def load_chunks_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []

    if not Path(file_path).exists():
        print(f"Warning: {file_path} not found")
        return chunks

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue

    return chunks


def analyze_key_alignment(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze key patterns and potential misalignments."""
    chunk_keys = set()
    cpesh_keys = set()
    key_patterns = {}

    for chunk in chunks:
        # Check for chunk identifiers
        if 'chunk_id' in chunk:
            chunk_keys.add(chunk['chunk_id'])
        if 'id' in chunk:
            chunk_keys.add(chunk['id'])

        # Check for CPESH data
        if 'cpesh' in chunk or 'context' in chunk or 'enrichment' in chunk:
            cpesh_keys.add(chunk.get('chunk_id', chunk.get('id', 'unknown')))

        # Analyze key patterns
        for key in ['chunk_id', 'id', 'source_id', 'cpe_id']:
            if key in chunk:
                pattern = type(chunk[key]).__name__
                key_patterns[key] = key_patterns.get(key, set())
                key_patterns[key].add(pattern)

    alignment_rate = len(chunk_keys & cpesh_keys) / len(chunk_keys) if chunk_keys else 0

    return {
        'total_chunks': len(chunks),
        'chunks_with_keys': len(chunk_keys),
        'chunks_with_cpesh': len(cpesh_keys),
        'aligned_keys': len(chunk_keys & cpesh_keys),
        'alignment_rate': round(alignment_rate * 100, 1),
        'missing_cpesh': len(chunk_keys - cpesh_keys),
        'orphaned_cpesh': len(cpesh_keys - chunk_keys),
        'key_patterns': {k: list(v) for k, v in key_patterns.items()}
    }


def fix_cpesh_alignment(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix CPESH alignment by generating missing contextual data."""
    fixed_chunks = []
    generated_count = 0

    for i, chunk in enumerate(chunks):
        fixed_chunk = chunk.copy()

        # Ensure consistent chunk ID
        if 'chunk_id' not in fixed_chunk:
            if 'id' in fixed_chunk:
                fixed_chunk['chunk_id'] = fixed_chunk['id']
            else:
                text = fixed_chunk.get('text', fixed_chunk.get('content', ''))
                fixed_chunk['chunk_id'] = generate_chunk_key(text, i)

        # Check if CPESH data exists
        has_cpesh = any(key in fixed_chunk for key in ['cpesh', 'context', 'enrichment', 'keywords'])

        if not has_cpesh:
            # Generate CPESH data
            text = fixed_chunk.get('text', fixed_chunk.get('content', ''))
            if text:
                cpesh_data = generate_cpesh_context(text, fixed_chunk['chunk_id'])
                fixed_chunk['cpesh'] = cpesh_data
                generated_count += 1

        fixed_chunks.append(fixed_chunk)

    return fixed_chunks, generated_count


def save_chunks_to_jsonl(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Save chunks to JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def create_cpesh_migration_report(
    original_chunks: List[Dict[str, Any]],
    fixed_chunks: List[Dict[str, Any]],
    generated_count: int
) -> Dict[str, Any]:
    """Create migration report."""
    original_analysis = analyze_key_alignment(original_chunks)
    fixed_analysis = analyze_key_alignment(fixed_chunks)

    return {
        'migration_summary': {
            'original_chunks': len(original_chunks),
            'fixed_chunks': len(fixed_chunks),
            'cpesh_generated': generated_count,
            'success_rate': round(generated_count / len(original_chunks) * 100, 1) if original_chunks else 0
        },
        'before': original_analysis,
        'after': fixed_analysis,
        'improvement': {
            'alignment_rate': round(fixed_analysis['alignment_rate'] - original_analysis['alignment_rate'], 1),
            'missing_cpesh': original_analysis['missing_cpesh'] - fixed_analysis['missing_cpesh']
        }
    }


def main():
    """Main CPESH fixing workflow."""
    input_file = "artifacts/cpesh_active.jsonl"
    output_file = "artifacts/cpesh_active_fixed.jsonl"
    report_file = "artifacts/cpesh_fix_report.json"

    print("CPESH Fixer - Addressing key alignment issues")
    print("-" * 50)

    # Load chunks
    print(f"Loading chunks from {input_file}...")
    chunks = load_chunks_from_jsonl(input_file)
    print(f"Loaded {len(chunks)} chunks")

    # Analyze current state
    print("\nAnalyzing current alignment...")
    original_analysis = analyze_key_alignment(chunks)
    print(f"Current CPESH attachment rate: {original_analysis['alignment_rate']}%")
    print(f"Missing CPESH: {original_analysis['missing_cpesh']} chunks")

    # Fix alignment
    print("\nFixing CPESH alignment...")
    fixed_chunks, generated_count = fix_cpesh_alignment(chunks)

    # Analyze results
    fixed_analysis = analyze_key_alignment(fixed_chunks)
    print(f"Generated CPESH for {generated_count} chunks")
    print(f"New CPESH attachment rate: {fixed_analysis['alignment_rate']}%")

    # Save results
    print(f"\nSaving fixed chunks to {output_file}...")
    save_chunks_to_jsonl(fixed_chunks, output_file)

    # Create report
    report = create_cpesh_migration_report(chunks, fixed_chunks, generated_count)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Migration report saved to {report_file}")
    print("\n✅ CPESH fixing complete!")
    print(f"Improvement: {report['improvement']['alignment_rate']}% increase in attachment rate")


if __name__ == "__main__":
    main()