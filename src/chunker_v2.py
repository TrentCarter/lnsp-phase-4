#!/usr/bin/env python3
"""
Chunker V2: Proper text chunking with 180-320 word targets.

This replaces the broken chunking that was producing 68-word snippets.
Aggregates sentences to reach target word count while preserving boundaries.
"""

import re
from typing import List, Dict, Any
import hashlib
import json


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using improved regex patterns."""
    # Handle common abbreviations and edge cases
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<PERIOD> ', text)
    text = re.sub(r'\b([A-Z])\.\s*([A-Z])\.\s*', r'\1<PERIOD> \2<PERIOD> ', text)
    text = re.sub(r'\.{3,}', '<ELLIPSIS>', text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore periods and ellipses
    sentences = [s.replace('<PERIOD>', '.').replace('<ELLIPSIS>', '...') for s in sentences]

    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def create_chunks(
    text: str,
    min_words: int = 180,
    max_words: int = 320,
    overlap_sentences: int = 1
) -> List[Dict[str, Any]]:
    """
    Create properly-sized chunks from text.

    Args:
        text: Input text to chunk
        min_words: Minimum words per chunk (default 180)
        max_words: Maximum words per chunk (default 320)
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of chunk dictionaries with text, word count, and position
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    chunk_index = 0

    while i < len(sentences):
        current_chunk = []
        current_words = 0

        # Build chunk by adding sentences
        j = i
        while j < len(sentences) and current_words < min_words:
            sentence = sentences[j]
            sentence_words = count_words(sentence)

            # Check if adding this sentence would exceed max
            if current_words + sentence_words > max_words and current_chunk:
                # Only break if we already have content
                break

            current_chunk.append(sentence)
            current_words += sentence_words
            j += 1

        # Handle edge cases
        if not current_chunk:
            # Single sentence too long, split it
            if i < len(sentences):
                words = sentences[i].split()
                if len(words) > max_words:
                    # Split long sentence into max_words chunks
                    for k in range(0, len(words), max_words):
                        chunk_words = words[k:k+max_words]
                        chunk_text = ' '.join(chunk_words)
                        chunks.append({
                            'text': chunk_text,
                            'word_count': len(chunk_words),
                            'chunk_index': chunk_index,
                            'start_sentence': i,
                            'end_sentence': i,
                            'chunk_id': generate_chunk_id(chunk_text, chunk_index)
                        })
                        chunk_index += 1
                    i += 1
                    continue
                else:
                    # Include the sentence even if it's short
                    current_chunk = [sentences[i]]
                    current_words = count_words(sentences[i])
                    j = i + 1

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_words,
                'chunk_index': chunk_index,
                'start_sentence': i,
                'end_sentence': j - 1,
                'chunk_id': generate_chunk_id(chunk_text, chunk_index)
            })
            chunk_index += 1

            # Move forward with overlap
            i = max(i + 1, j - overlap_sentences)
        else:
            i += 1

    return chunks


def generate_chunk_id(text: str, index: int) -> str:
    """Generate stable chunk ID from text content and index."""
    content = f"{index}:{text[:100]}"  # Use index and first 100 chars
    return hashlib.md5(content.encode()).hexdigest()[:16]


def merge_short_chunks(chunks: List[Dict[str, Any]], min_words: int = 180, max_words: int = 320) -> List[Dict[str, Any]]:
    """
    Merge consecutive short chunks to reach target size.
    Used for fixing existing data.
    """
    if not chunks:
        return []

    merged = []
    current_buffer = []
    current_words = 0

    for chunk in chunks:
        chunk_words = chunk.get('word_count', count_words(chunk.get('text', '')))

        # If this chunk alone is in range, add it directly
        if min_words <= chunk_words <= max_words:
            # First flush buffer if any
            if current_buffer:
                merged_text = ' '.join([c.get('text', '') for c in current_buffer])
                merged.append({
                    'text': merged_text,
                    'word_count': current_words,
                    'chunk_index': len(merged),
                    'chunk_id': generate_chunk_id(merged_text, len(merged)),
                    'merged_from': [c.get('chunk_id', '') for c in current_buffer]
                })
                current_buffer = []
                current_words = 0

            # Add this chunk as-is
            merged.append(chunk)

        # If adding this would exceed max, flush buffer first
        elif current_words + chunk_words > max_words and current_buffer:
            merged_text = ' '.join([c.get('text', '') for c in current_buffer])
            merged.append({
                'text': merged_text,
                'word_count': current_words,
                'chunk_index': len(merged),
                'chunk_id': generate_chunk_id(merged_text, len(merged)),
                'merged_from': [c.get('chunk_id', '') for c in current_buffer]
            })
            current_buffer = [chunk]
            current_words = chunk_words

        # Otherwise add to buffer
        else:
            current_buffer.append(chunk)
            current_words += chunk_words

            # Check if buffer reached minimum
            if current_words >= min_words:
                merged_text = ' '.join([c.get('text', '') for c in current_buffer])
                merged.append({
                    'text': merged_text,
                    'word_count': current_words,
                    'chunk_index': len(merged),
                    'chunk_id': generate_chunk_id(merged_text, len(merged)),
                    'merged_from': [c.get('chunk_id', '') for c in current_buffer]
                })
                current_buffer = []
                current_words = 0

    # Flush remaining buffer
    if current_buffer:
        merged_text = ' '.join([c.get('text', '') for c in current_buffer])
        merged.append({
            'text': merged_text,
            'word_count': current_words,
            'chunk_index': len(merged),
            'chunk_id': generate_chunk_id(merged_text, len(merged)),
            'merged_from': [c.get('chunk_id', '') for c in current_buffer]
        })

    return merged


def analyze_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chunk statistics for quality checking."""
    if not chunks:
        return {
            'total_chunks': 0,
            'mean_words': 0,
            'min_words': 0,
            'max_words': 0,
            'p95_words': 0,
            'short_chunks': 0,
            'target_range': 0
        }

    word_counts = [c.get('word_count', count_words(c.get('text', ''))) for c in chunks]
    word_counts.sort()

    total = len(word_counts)
    mean_words = sum(word_counts) / total if total > 0 else 0
    p95_index = int(total * 0.95)
    p95_words = word_counts[p95_index] if p95_index < total else word_counts[-1]

    short_chunks = sum(1 for w in word_counts if w < 120)
    target_range = sum(1 for w in word_counts if 180 <= w <= 320)

    return {
        'total_chunks': total,
        'mean_words': round(mean_words, 1),
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'p95_words': p95_words,
        'short_chunks': short_chunks,
        'short_pct': round(100 * short_chunks / total, 1) if total > 0 else 0,
        'target_range': target_range,
        'target_pct': round(100 * target_range / total, 1) if total > 0 else 0
    }


if __name__ == "__main__":
    # Test the chunker
    sample_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
    carbon dioxide as inputs. During photosynthesis, plants absorb light energy through
    chlorophyll molecules. This energy is used to convert carbon dioxide and water into
    glucose and oxygen. The glucose serves as food for the plant, while oxygen is released
    as a byproduct. This process is essential for life on Earth, as it produces the oxygen
    we breathe and forms the base of most food chains. Plants typically perform photosynthesis
    during daylight hours when sunlight is available. The rate of photosynthesis can be
    affected by factors such as light intensity, temperature, and carbon dioxide concentration.
    Different plants have adapted various strategies to maximize their photosynthetic efficiency
    in different environments. Some plants, called C4 plants, have evolved special mechanisms
    to concentrate carbon dioxide, making them more efficient in hot, dry conditions. Others,
    known as CAM plants, open their stomata at night to minimize water loss while still
    maintaining photosynthesis. Understanding photosynthesis is crucial for agriculture,
    climate science, and biotechnology research.
    """

    print("Testing Chunker V2")
    print("-" * 50)

    # Create chunks
    chunks = create_chunks(sample_text, min_words=180, max_words=320)

    # Analyze results
    stats = analyze_chunks(chunks)

    print(f"Created {stats['total_chunks']} chunks")
    print(f"Mean words: {stats['mean_words']}")
    print(f"Range: {stats['min_words']}-{stats['max_words']} words")
    print(f"In target range (180-320): {stats['target_pct']}%")
    print()

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk['word_count']} words")
        print(f"ID: {chunk['chunk_id']}")
        print(f"Text preview: {chunk['text'][:100]}...")
        print()