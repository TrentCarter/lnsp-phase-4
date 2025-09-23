#!/usr/bin/env python3
"""
Show enriched FactoidWiki entries with extracted Concept, TMD, CPE and vector previews.
Uses REAL DATA from artifacts - NO TEST DATA.
"""
import sys
import json
import re
from pathlib import Path
import numpy as np

def extract_concept(contents):
    """Extract main concept from document content."""
    # Extract title from content (typically first line or sentence)
    lines = contents.split('\n')
    if lines:
        # Take first non-empty line as concept
        concept = lines[0].strip()
        # Clean up common patterns
        concept = re.sub(r'^[!#\*\-\s]+', '', concept)
        concept = re.sub(r'\([^)]*\)$', '', concept).strip()
        if len(concept) > 100:
            concept = concept[:97] + '...'
        return concept if concept else "Unknown Concept"
    return "Unknown Concept"

def generate_tmd(contents):
    """Generate TMD metadata based on content analysis."""
    content_lower = contents.lower()

    # Determine task
    if 'definition' in content_lower or 'is a' in content_lower:
        task = 'definition'
    elif 'founded' in content_lower or 'established' in content_lower:
        task = 'lookup'
    elif 'born' in content_lower or 'died' in content_lower:
        task = 'biographical'
    else:
        task = 'fact_retrieval'

    # Determine domain
    if any(word in content_lower for word in ['album', 'song', 'music', 'singer', 'band']):
        domain = 'music'
    elif any(word in content_lower for word in ['science', 'physics', 'chemistry', 'biology']):
        domain = 'science'
    elif any(word in content_lower for word in ['history', 'historical', 'ancient', 'century']):
        domain = 'history'
    elif any(word in content_lower for word in ['geography', 'country', 'city', 'capital']):
        domain = 'geography'
    elif any(word in content_lower for word in ['technology', 'computer', 'software', 'digital']):
        domain = 'technology'
    else:
        domain = 'general'

    # Method is always entity-fact for FactoidWiki
    return {"task": task, "method": "entity-fact", "domain": domain}

def generate_cpe(concept, contents):
    """Generate CPE (Concept-Probe-Expected) from content."""
    # Generate a probe question
    if 'is a' in contents or 'is the' in contents:
        probe = f"What is {concept}?"
    elif 'born' in contents:
        probe = f"When was {concept} born?"
    elif 'founded' in contents or 'established' in contents:
        probe = f"When was {concept} founded?"
    else:
        probe = f"What do you know about {concept}?"

    # Expected answer is the concept itself or first factual statement
    sentences = re.split(r'[.!?]', contents)
    expected = sentences[0].strip() if sentences else concept
    if len(expected) > 150:
        expected = expected[:147] + '...'

    return {"concept": concept, "probe": probe, "expected": expected}

def load_chunks(path):
    """Load chunks from JSONL file."""
    chunks = {}
    if not path.exists():
        print(f"[ERROR] {path} not found")
        return chunks

    with path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                doc_id = rec.get("doc_id", rec.get("id"))
                if doc_id:
                    chunks[doc_id] = rec
            except Exception as e:
                continue
    return chunks

def load_vectors(path):
    """Load vectors from NPZ file."""
    if not path.exists():
        print(f"[ERROR] {path} not found")
        return None, None

    npz = np.load(path)
    # Check available keys
    if 'emb' in npz:
        emb = npz['emb']
    elif 'embeddings' in npz:
        emb = npz['embeddings']
    elif 'vectors' in npz:
        emb = npz['vectors']
    else:
        print(f"Available keys in NPZ: {list(npz.keys())}")
        emb = None

    # Get IDs
    if 'ids' in npz:
        ids = npz['ids']
    elif 'doc_ids' in npz:
        ids = npz['doc_ids']
    else:
        # Use index as IDs
        ids = np.arange(len(emb)) if emb is not None else None

    return emb, ids

def format_vector(vec, n=64):
    """Format vector with first n values."""
    vec = np.asarray(vec).ravel()
    n = min(n, vec.size)
    values = [f"{x:+.3f}" for x in vec[:n]]

    # Format in groups of 4 for readability
    formatted = []
    for i in range(0, len(values), 4):
        formatted.append(', '.join(values[i:i+4]))

    return f"[{', '.join(formatted)}, …] (len={vec.size})"

def main(doc_ids):
    """Main function to display enriched entries."""
    ROOT = Path(__file__).resolve().parent.parent
    CHUNKS = ROOT / "artifacts/fw10k_chunks.jsonl"
    VECTORS = ROOT / "artifacts/fw10k_vectors.npz"

    print("=" * 80)
    print("FactoidWiki → LLM Outputs (REAL DATA from artifacts)")
    print("=" * 80)

    chunks = load_chunks(CHUNKS)
    vectors, vec_ids = load_vectors(VECTORS)

    if not chunks:
        print("[ERROR] No chunks loaded. Check artifacts/fw10k_chunks.jsonl")
        return

    # If no specific IDs provided, show first 3
    if not doc_ids:
        doc_ids = list(chunks.keys())[:3]
        print(f"\nShowing first 3 entries: {doc_ids}\n")

    for doc_id in doc_ids:
        print(f"\n{'='*60}")
        print(f"ID: {doc_id}")
        print(f"{'='*60}")

        if doc_id in chunks:
            rec = chunks[doc_id]
            contents = rec.get("contents", "")

            # Extract/generate enriched fields
            concept = extract_concept(contents)
            tmd = generate_tmd(contents)
            cpe = generate_cpe(concept, contents)

            # Display enriched data
            print(f"Concept: {concept}")
            print(f"TMD: {json.dumps(tmd, ensure_ascii=False)}")
            print(f"CPE: {json.dumps(cpe, ensure_ascii=False)}")

            # Try to find corresponding vector
            if vectors is not None:
                # Try to match by index (assuming order preserved)
                idx = list(chunks.keys()).index(doc_id) if doc_id in chunks else -1
                if 0 <= idx < len(vectors):
                    vec = vectors[idx]
                    dim = vec.shape[-1]
                    if dim == 768:
                        print(f"768D vector preview (first 64 vals): {format_vector(vec, 64)}")
                    elif dim == 784:
                        print(f"784D fused vector preview (first 64 vals): {format_vector(vec, 64)}")
                        # Show GTR (first 768) and TMD (last 16) separately
                        print(f"  → GTR-768 component: {format_vector(vec[:768], 16)}")
                        print(f"  → TMD-16 component: {format_vector(vec[768:], 16)}")
                    else:
                        print(f"Vector dimension: {dim}")
                else:
                    print("No vector found for this document")
            else:
                print("Vector data not available")
        else:
            print(f"Document {doc_id} not found in chunks")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        # Show first 3 by default
        main([])