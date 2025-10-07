"""
STAGE 6: Schema-Aware LLM Smoothing

Generates natural language responses with MANDATORY concept ID citations.
All factual claims must cite concept IDs in (id:text) format.

Key features:
- Constrained generation (temp≤0.3, top_p≤0.8)
- Post-check validation (reject uncited sentences)
- Automatic regeneration if citation rate <90%

See: docs/PRDs/PRD_Inference_LVM_v2_PRODUCTION.md (lines 578-650)
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class ConceptPair:
    """Input concept + predicted next concept."""
    input_id: str
    input_text: str
    next_id: str
    next_text: str
    path: str  # "ANN", "GRAPH", "CROSS", "V2T"
    confidence: float


@dataclass
class SmoothedResponse:
    """LLM response with citation validation."""
    response_text: str
    cited_ids: Set[str]
    uncited_sentences: List[str]
    citation_rate: float
    regenerated: bool = False


class LLMSmoother:
    """
    Schema-aware LLM smoother with mandatory concept ID citations.

    Uses Llama 3.1:8b via Ollama for natural language generation.
    """

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "llama3.1:8b",
        temperature: float = 0.3,
        top_p: float = 0.8,
        min_citation_rate: float = 0.9
    ):
        self.endpoint = llm_endpoint
        self.model = llm_model
        self.temperature = temperature
        self.top_p = top_p
        self.min_citation_rate = min_citation_rate

    def smooth(
        self,
        query: str,
        concept_pairs: List[ConceptPair],
        max_attempts: int = 2
    ) -> SmoothedResponse:
        """
        Generate natural language response with concept ID citations.

        Args:
            query: Original user query
            concept_pairs: List of (input, next) concept pairs
            max_attempts: Max regeneration attempts if citations missing

        Returns:
            SmoothedResponse with validation results
        """
        valid_ids = self._extract_valid_ids(concept_pairs)

        for attempt in range(max_attempts):
            # Generate response
            prompt = self._build_prompt(query, concept_pairs, attempt=attempt)
            max_tokens = 60 + 20 * len(concept_pairs)

            response_text = self._call_llm(prompt, max_tokens)

            # Validate citations
            cited_ids = self._extract_cited_ids(response_text)
            uncited_sentences = self._find_uncited_sentences(response_text, cited_ids)

            total_sentences = len(self._split_sentences(response_text))
            citation_rate = (total_sentences - len(uncited_sentences)) / max(total_sentences, 1)

            # Check if acceptable
            if citation_rate >= self.min_citation_rate:
                logger.info(f"Citations valid: {citation_rate:.1%} rate (attempt {attempt+1})")
                return SmoothedResponse(
                    response_text=response_text,
                    cited_ids=cited_ids & valid_ids,  # Only valid IDs
                    uncited_sentences=uncited_sentences,
                    citation_rate=citation_rate,
                    regenerated=(attempt > 0)
                )

            logger.warning(f"Low citation rate {citation_rate:.1%} (attempt {attempt+1}), regenerating...")

        # Final attempt failed - return with warning
        logger.error(f"Failed to achieve {self.min_citation_rate:.0%} citation rate after {max_attempts} attempts")
        return SmoothedResponse(
            response_text=response_text,
            cited_ids=cited_ids & valid_ids,
            uncited_sentences=uncited_sentences,
            citation_rate=citation_rate,
            regenerated=True
        )

    def _build_prompt(self, query: str, pairs: List[ConceptPair], attempt: int = 0) -> str:
        """Build prompt with mandatory citation requirements."""
        # Format concept pairs
        pairs_text = self._format_concept_pairs(pairs)

        # Stricter wording for retry attempts
        strictness = [
            "CRITICAL REQUIREMENTS:",
            "⚠️  MANDATORY REQUIREMENTS (LOW CITATION RATE - STRICTER):",
            "🔴 FINAL ATTEMPT - EVERY SENTENCE MUST CITE:"
        ][min(attempt, 2)]

        prompt = f"""You are a knowledge system that predicts related concepts.

User Query: {query}

Retrieved Concepts with Predictions:
{pairs_text}

{strictness}
1. EVERY factual claim MUST cite a concept ID in (id:text) format.
2. Use the EXACT IDs provided above (e.g., "{pairs[0].input_id}:{pairs[0].input_text}").
3. Do NOT invent facts not supported by the retrieved concepts.
4. Keep response concise: 2-3 sentences maximum.

Example Format:
"({pairs[0].input_id}:Neural networks) are computational models that enable ({pairs[0].next_id}:deep learning), a key technique in artificial intelligence for pattern recognition."

Response (with mandatory citations):"""

        return prompt

    def _format_concept_pairs(self, pairs: List[ConceptPair]) -> str:
        """Format concept pairs for prompt."""
        lines = []
        for i, pair in enumerate(pairs, 1):
            lines.append(
                f"{i}. Input: ({pair.input_id}:{pair.input_text}) "
                f"→ Next: ({pair.next_id}:{pair.next_text}) "
                f"[confidence: {pair.confidence:.2f}, path: {pair.path}]"
            )
        return "\n".join(lines)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call Llama 3.1 via Ollama API."""
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "num_predict": max_tokens
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error generating response: {e}"

    def _extract_valid_ids(self, pairs: List[ConceptPair]) -> Set[str]:
        """Extract all valid concept IDs from pairs."""
        ids = set()
        for pair in pairs:
            ids.add(pair.input_id)
            ids.add(pair.next_id)
        return ids

    def _extract_cited_ids(self, text: str) -> Set[str]:
        """Extract all cited concept IDs from response."""
        # Pattern: (id:text) where id is UUID or similar
        pattern = r'\(([a-f0-9-]{8,}):([^)]+)\)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return {match[0] for match in matches}

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (can use nltk for production)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_uncited_sentences(self, text: str, cited_ids: Set[str]) -> List[str]:
        """Find sentences that lack concept ID citations."""
        sentences = self._split_sentences(text)
        uncited = []

        for sentence in sentences:
            # Check if sentence contains any citations
            has_citation = bool(re.search(r'\([a-f0-9-]{8,}:[^)]+\)', sentence, re.IGNORECASE))
            if not has_citation and len(sentence) > 10:  # Ignore very short sentences
                uncited.append(sentence)

        return uncited


# ============================================================================
# Demo / Testing
# ============================================================================

def demo():
    """Demonstrate LLM smoothing with citations."""
    print("="*60)
    print("STAGE 6: Schema-Aware LLM Smoothing Demo")
    print("="*60)

    smoother = LLMSmoother()

    # Mock concept pairs
    pairs = [
        ConceptPair(
            input_id="uuid-1234",
            input_text="machine learning",
            next_id="uuid-5678",
            next_text="neural networks",
            path="ANN",
            confidence=0.85
        ),
        ConceptPair(
            input_id="uuid-5678",
            input_text="neural networks",
            next_id="uuid-9abc",
            next_text="deep learning",
            path="GRAPH",
            confidence=0.78
        )
    ]

    query = "What are the key concepts in modern AI?"

    print(f"\nQuery: {query}")
    print(f"Concept pairs: {len(pairs)}")
    print("\nGenerating response with mandatory citations...\n")

    result = smoother.smooth(query, pairs)

    print("Response:")
    print(f"  {result.response_text}\n")
    print(f"Citation rate: {result.citation_rate:.1%}")
    print(f"Cited IDs: {result.cited_ids}")
    if result.uncited_sentences:
        print(f"⚠️  Uncited sentences: {result.uncited_sentences}")
    if result.regenerated:
        print(f"♻️  Response was regenerated")

    print("\n" + "="*60)


if __name__ == "__main__":
    demo()
