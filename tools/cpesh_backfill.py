#!/usr/bin/env python3
"""
CPESH Backfill Script - Regenerates CPESH for short chunks by batching them into context windows.
Pass 1: Merge contiguous chunks until 180-320 words
Pass 2: Generate CPESH for merged chunks
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?])\s+')

HARD_NEGATIVE_POOL = [
    "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
    "Plate tectonics explains how Earth's crustal plates shift across the mantle.",
    "Binary search runs in logarithmic time on sorted collections.",
    "The Krebs cycle generates ATP by oxidizing acetyl-CoA in mitochondria.",
    "Quantum entanglement links particle states regardless of distance.",
    "SNMP is a network protocol for monitoring and managing devices.",
]

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

        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        sentences = self._extract_sentences(content)

        concept = self._extract_concept(lines, sentences)
        probe = self._build_probe(concept)
        expected = self._select_expected(sentences, concept)
        soft_negative = self._select_soft_negative(sentences, concept, expected, content)
        hard_negative = self._select_hard_negative(concept)

        cpesh_payload = {
            'concept': concept,
            'probe': probe,
            'expected': expected,
            'soft_negative': soft_negative,
            'hard_negative': hard_negative,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'last_accessed': datetime.now(timezone.utc).isoformat(),
            'generation_method': 'backfill_v1'
        }

        return {
            'doc_id': merged_entry['merged_id'],
            'source_ids': merged_entry['source_ids'],
            'cpesh': cpesh_payload,
            'access_count': 0,
            'word_count': merged_entry['word_count'],
            'chunk_count': merged_entry['chunk_count']
        }

    def _extract_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        normalized = re.sub(r'\s+', ' ', text.replace('\n', ' ').strip())
        raw = [s.strip() for s in SENTENCE_SPLIT_RE.split(normalized) if s.strip()]
        cleaned: List[str] = []
        for sentence in raw:
            sentence = sentence.strip()
            if not sentence:
                continue
            cleaned.append(sentence)
        if not cleaned:
            cleaned = [seg.strip() for seg in text.split('\n') if seg.strip()]
        return cleaned

    def _extract_concept(self, lines: List[str], sentences: List[str]) -> str:
        if lines:
            candidate = lines[0]
        elif sentences:
            candidate = sentences[0]
        else:
            candidate = ""
        return candidate[:200]

    def _build_probe(self, concept: str) -> str:
        if not concept:
            return "What is the main concept described in this chunk?"

        concept_stripped = concept.strip()
        tokens = concept_stripped.split()
        if tokens and all(tok[:1].isupper() for tok in tokens[:2]):
            return f"Who is {concept_stripped}?"
        if re.search(r'\b(year|when|date)\b', concept_stripped.lower()):
            return f"When did {concept_stripped} occur?"
        return f"What is {concept_stripped}?"

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        return set(re.findall(r'\w+', text.lower()))

    def _select_expected(self, sentences: List[str], concept: str) -> str:
        if not sentences:
            return concept[:200] if concept else ""
        concept_tokens = self._tokenize(concept)
        best_sentence = None
        best_score = -1
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if len(tokens) < 4:
                continue
            overlap = len(tokens & concept_tokens)
            if overlap > best_score:
                best_sentence = sentence
                best_score = overlap
        if best_sentence:
            return best_sentence.strip()

        # Fall back to first reasonably informative sentence
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if len(tokens) >= 4:
                return sentence.strip()

        return sentences[0].strip()

    def _extract_proper_nouns(self, text: str) -> List[str]:
        if not text:
            return []
        nouns = []
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            value = match.group().strip()
            if value and value not in nouns:
                nouns.append(value)
        return nouns

    def _select_soft_negative(
        self,
        sentences: List[str],
        concept: str,
        expected: str,
        content: str,
    ) -> str:
        concept_tokens = self._tokenize(concept)
        expected_tokens = self._tokenize(expected)
        candidates: List[Tuple[int, str]] = []
        for sentence in sentences:
            if sentence == expected:
                continue
            tokens = self._tokenize(sentence)
            if not tokens or len(tokens) < 4:
                continue
            overlap = len(tokens & concept_tokens)
            if overlap == 0:
                continue
            if tokens == expected_tokens:
                continue
            candidates.append((overlap, sentence.strip()))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], len(item[1])))
            return candidates[0][1]

        proper_nouns = self._extract_proper_nouns(content)
        for noun in proper_nouns:
            if noun and noun.lower() not in concept.lower():
                return f"{noun} is discussed in the source but is not the correct answer to the probe about {concept}."

        keywords = self._top_keywords(content)
        if keywords:
            keyword = keywords[0]
            return f"The concept '{keyword}' appears in the source yet does not answer the probe about {concept}."

        return f"A related entity in the source is not the answer to the probe about {concept}."

    def _select_hard_negative(self, concept: str) -> str:
        concept_tokens = self._tokenize(concept)
        if not HARD_NEGATIVE_POOL:
            return "This statement is unrelated to the probe."

        scored = [
            (len(self._tokenize(candidate) & concept_tokens), candidate)
            for candidate in HARD_NEGATIVE_POOL
        ]
        scored.sort(key=lambda item: (item[0], len(item[1])))
        best_score = scored[0][0]
        best_candidates = [candidate for score, candidate in scored if score == best_score]
        if not best_candidates:
            return scored[0][1]
        idx = hash(concept) % len(best_candidates) if concept else 0
        return best_candidates[idx]

    def _top_keywords(self, text: str, limit: int = 5) -> List[str]:
        if not text:
            return []
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        stop_words = {
            'the', 'and', 'for', 'with', 'that', 'from', 'this', 'have', 'will',
            'their', 'about', 'there', 'which', 'into', 'would', 'while', 'where',
            'because', 'after', 'before', 'between', 'under', 'over', 'than',
            'been', 'being', 'through', 'during', 'without', 'against', 'among',
        }
        filtered = [w for w in words if w not in stop_words]
        if not filtered:
            return []
        counts = {}
        for word in filtered:
            counts[word] = counts.get(word, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [word for word, _ in ordered[:limit]]

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
