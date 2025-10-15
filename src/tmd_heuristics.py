"""
TMD Heuristics - Fast Task/Modifier Classification
===================================================

Provides fast keyword-based classification for Task (T) and Modifier (M) codes
when LNSP_TMD_MODE=hybrid. Domain (D) is still extracted via LLM.

Performance: ~0.1-0.5ms per chunk (vs ~200ms with LLM)
Accuracy: ~70-80% (vs ~95% with LLM)

Usage:
    from src.tmd_heuristics import classify_task, classify_modifier

    task_code = classify_task(chunk_text)
    modifier_code = classify_modifier(chunk_text)
"""

import re
from typing import Dict, List, Tuple


# ============================================================================
# Task Classification (32 types, 5 bits)
# ============================================================================

TASK_PATTERNS = {
    # Definitional tasks (0-7)
    0: {
        "keywords": ["is", "are", "means", "refers to", "defined as", "called", "known as"],
        "patterns": [r"\b(?:what|which)\s+(?:is|are)\b", r"\bdefin(?:e|ition)\b"],
        "name": "definition"
    },
    1: {
        "keywords": ["explain", "describe", "explanation", "description", "overview"],
        "patterns": [r"\bexplain(?:s|ed|ing)?\b", r"\bdescrib(?:e|es|ing)\b"],
        "name": "explanation"
    },
    2: {
        "keywords": ["example", "for instance", "such as", "including", "like"],
        "patterns": [r"\bfor\s+example\b", r"\bsuch\s+as\b"],
        "name": "example"
    },
    3: {
        "keywords": ["classification", "category", "type of", "kind of", "classified"],
        "patterns": [r"\btype\s+of\b", r"\bkind\s+of\b", r"\bclassif(?:y|ied|ication)\b"],
        "name": "classification"
    },

    # Comparative tasks (8-11)
    8: {
        "keywords": ["compare", "comparison", "contrast", "versus", "vs", "difference", "similar"],
        "patterns": [r"\bcompar(?:e|ison)\b", r"\bcontrast\b", r"\bvs\.?\b", r"\bversus\b"],
        "name": "comparison"
    },
    9: {
        "keywords": ["same as", "identical", "equivalent", "equal to", "synonymous"],
        "patterns": [r"\bsame\s+as\b", r"\bidentical\s+to\b", r"\bequivalent\b"],
        "name": "equivalence"
    },

    # Procedural tasks (12-15)
    12: {
        "keywords": ["how to", "steps", "procedure", "method", "process", "instructions"],
        "patterns": [r"\bhow\s+to\b", r"\bstep(?:s)?\b", r"\bprocedure\b", r"\bmethod\b"],
        "name": "procedure"
    },
    13: {
        "keywords": ["algorithm", "computation", "calculate", "compute", "formula"],
        "patterns": [r"\balgorithm\b", r"\bcalculat(?:e|ion)\b", r"\bcompute\b"],
        "name": "algorithm"
    },

    # Analytical tasks (16-23)
    16: {
        "keywords": ["analyze", "analysis", "examine", "investigate", "study"],
        "patterns": [r"\banalyz(?:e|ed|ing)\b", r"\banalysis\b", r"\bexamine\b"],
        "name": "analysis"
    },
    17: {
        "keywords": ["evaluate", "assessment", "judge", "appraise", "critique"],
        "patterns": [r"\bevaluat(?:e|ion)\b", r"\bassess(?:ment)?\b", r"\bcritique\b"],
        "name": "evaluation"
    },
    18: {
        "keywords": ["why", "reason", "cause", "because", "due to", "result of"],
        "patterns": [r"\bwhy\b", r"\breason\b", r"\bcause\b", r"\bbecause\b"],
        "name": "causation"
    },
    19: {
        "keywords": ["effect", "consequence", "impact", "result", "outcome"],
        "patterns": [r"\beffect\b", r"\bconsequence\b", r"\bimpact\b", r"\bresult\b"],
        "name": "consequence"
    },

    # Narrative tasks (24-27)
    24: {
        "keywords": ["history", "historical", "originated", "began", "started", "evolution"],
        "patterns": [r"\bhistor(?:y|ical)\b", r"\borigin(?:ated)?\b", r"\bevolution\b"],
        "name": "historical"
    },
    25: {
        "keywords": ["sequence", "chronology", "timeline", "order", "first", "then", "next"],
        "patterns": [r"\bfirst\b.*\bthen\b", r"\bchronolog(?:y|ical)\b", r"\bsequence\b"],
        "name": "sequential"
    },

    # Factual tasks (28-31)
    28: {
        "keywords": ["fact", "data", "statistic", "figure", "number", "measurement"],
        "patterns": [r"\b\d+\s*%\b", r"\b\d+\s+(?:people|years|miles)\b"],
        "name": "factual"
    },
    29: {
        "keywords": ["list", "enumerate", "items", "include", "contain"],
        "patterns": [r"\blist\b", r"\benumerat(?:e|ion)\b"],
        "name": "enumeration"
    },
}

# Default task for unknown patterns
DEFAULT_TASK = 0  # definition


def classify_task(text: str) -> int:
    """
    Classify text into one of 32 task types using keyword matching.

    Args:
        text: Input text (chunk)

    Returns:
        Task code (0-31)

    Performance: ~0.1-0.3ms per chunk
    """
    text_lower = text.lower()

    # Score each task based on keyword/pattern matches
    scores = {}
    for task_code, task_info in TASK_PATTERNS.items():
        score = 0

        # Keyword matching
        for keyword in task_info["keywords"]:
            if keyword in text_lower:
                score += 1

        # Pattern matching
        for pattern in task_info["patterns"]:
            if re.search(pattern, text_lower):
                score += 2  # Patterns weighted higher

        if score > 0:
            scores[task_code] = score

    # Return highest scoring task
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return DEFAULT_TASK


# ============================================================================
# Modifier Classification (32 types, 5 bits)
# ============================================================================

MODIFIER_PATTERNS = {
    # Temporal modifiers (0-7)
    0: {
        "keywords": ["currently", "now", "present", "today", "modern"],
        "patterns": [r"\bcurrent(?:ly)?\b", r"\bpresent\b", r"\btoday\b"],
        "name": "present"
    },
    1: {
        "keywords": ["past", "historical", "previously", "former", "ancient", "old"],
        "patterns": [r"\bpast\b", r"\bhistorical(?:ly)?\b", r"\bprevious(?:ly)?\b"],
        "name": "past"
    },
    2: {
        "keywords": ["future", "will", "upcoming", "planned", "projected"],
        "patterns": [r"\bfuture\b", r"\bwill\s+\w+\b", r"\bupcoming\b"],
        "name": "future"
    },

    # Certainty modifiers (8-11)
    8: {
        "keywords": ["certain", "definitely", "always", "must", "absolutely"],
        "patterns": [r"\bcertain(?:ly)?\b", r"\bdefin(?:ite|itely)\b", r"\balways\b"],
        "name": "certain"
    },
    9: {
        "keywords": ["probably", "likely", "might", "may", "possibly", "perhaps"],
        "patterns": [r"\bprobab(?:ly|le)\b", r"\blikely\b", r"\bmight\b", r"\bmay\b"],
        "name": "probable"
    },
    10: {
        "keywords": ["unknown", "unclear", "uncertain", "ambiguous", "debated"],
        "patterns": [r"\bunknown\b", r"\bunclear\b", r"\buncertain\b"],
        "name": "uncertain"
    },

    # Scope modifiers (12-15)
    12: {
        "keywords": ["general", "overall", "broad", "comprehensive", "universal"],
        "patterns": [r"\bgeneral(?:ly)?\b", r"\boverall\b", r"\bbroad(?:ly)?\b"],
        "name": "general"
    },
    13: {
        "keywords": ["specific", "particular", "detailed", "precise", "exact"],
        "patterns": [r"\bspecific(?:ally)?\b", r"\bparticular(?:ly)?\b", r"\bdetailed\b"],
        "name": "specific"
    },

    # Evaluative modifiers (16-19)
    16: {
        "keywords": ["good", "positive", "beneficial", "advantage", "successful"],
        "patterns": [r"\bgood\b", r"\bpositive\b", r"\bbeneficial\b", r"\badvantage\b"],
        "name": "positive"
    },
    17: {
        "keywords": ["bad", "negative", "harmful", "disadvantage", "problem"],
        "patterns": [r"\bbad\b", r"\bnegative\b", r"\bharmful\b", r"\bproblem\b"],
        "name": "negative"
    },
    18: {
        "keywords": ["neutral", "objective", "unbiased", "factual"],
        "patterns": [r"\bneutral\b", r"\bobjective\b", r"\bunbiased\b"],
        "name": "neutral"
    },

    # Complexity modifiers (20-23)
    20: {
        "keywords": ["simple", "basic", "easy", "straightforward", "elementary"],
        "patterns": [r"\bsimple\b", r"\bbasic\b", r"\beasy\b", r"\bstraightforward\b"],
        "name": "simple"
    },
    21: {
        "keywords": ["complex", "complicated", "advanced", "sophisticated", "intricate"],
        "patterns": [r"\bcomplex\b", r"\bcomplicated\b", r"\badvanced\b"],
        "name": "complex"
    },

    # Importance modifiers (24-27)
    24: {
        "keywords": ["important", "critical", "essential", "key", "vital"],
        "patterns": [r"\bimportant\b", r"\bcritical\b", r"\bessential\b", r"\bkey\b"],
        "name": "important"
    },
    25: {
        "keywords": ["minor", "trivial", "insignificant", "marginal"],
        "patterns": [r"\bminor\b", r"\btrivial\b", r"\binsignificant\b"],
        "name": "minor"
    },
}

# Default modifier for unknown patterns
DEFAULT_MODIFIER = 18  # neutral


def classify_modifier(text: str) -> int:
    """
    Classify text modifiers using keyword/pattern matching.

    Args:
        text: Input text (chunk)

    Returns:
        Modifier code (0-31)

    Performance: ~0.1-0.2ms per chunk
    """
    text_lower = text.lower()

    # Score each modifier based on keyword/pattern matches
    scores = {}
    for mod_code, mod_info in MODIFIER_PATTERNS.items():
        score = 0

        # Keyword matching
        for keyword in mod_info["keywords"]:
            if keyword in text_lower:
                score += 1

        # Pattern matching
        for pattern in mod_info["patterns"]:
            if re.search(pattern, text_lower):
                score += 2

        if score > 0:
            scores[mod_code] = score

    # Return highest scoring modifier
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]

    return DEFAULT_MODIFIER


def classify_tmd_hybrid(text: str, domain_code: int) -> Tuple[int, int, int]:
    """
    Hybrid TMD classification: Domain provided, Task/Modifier via heuristics.

    Args:
        text: Input text (chunk)
        domain_code: Pre-computed domain code from LLM (0-63)

    Returns:
        Tuple of (domain, task, modifier)

    Performance: ~0.2-0.5ms per chunk (vs ~200ms full LLM)
    """
    task_code = classify_task(text)
    modifier_code = classify_modifier(text)

    return (domain_code, task_code, modifier_code)


def get_task_name(task_code: int) -> str:
    """Get human-readable task name"""
    return TASK_PATTERNS.get(task_code, {}).get("name", "unknown")


def get_modifier_name(modifier_code: int) -> str:
    """Get human-readable modifier name"""
    return MODIFIER_PATTERNS.get(modifier_code, {}).get("name", "unknown")
