#!/usr/bin/env python3
"""
TMD Extractor V2: Real content analysis instead of defaulting to 9.1.27.

This replaces the broken TMD extraction that was defaulting everything.
Analyzes content keywords and patterns to determine proper domain/task/modifier codes.
"""

import re
from typing import Dict, Tuple, List, Any
from collections import Counter


# Domain patterns (16 domains, 1-16)
DOMAIN_PATTERNS = {
    1: {  # Technology
        'keywords': ['computer', 'software', 'internet', 'digital', 'algorithm', 'programming',
                    'technology', 'tech', 'AI', 'artificial intelligence', 'machine learning',
                    'database', 'network', 'code', 'development', 'app', 'system'],
        'patterns': [r'\b(API|HTTP|URL|CPU|GPU|RAM|SQL|HTML|CSS|JavaScript)\b'],
        'name': 'Technology'
    },
    2: {  # Science
        'keywords': ['experiment', 'research', 'hypothesis', 'theory', 'scientific', 'study',
                    'analysis', 'method', 'result', 'conclusion', 'discovery', 'physics',
                    'chemistry', 'biology', 'genetics', 'molecule', 'atom', 'evolution'],
        'patterns': [r'\b(DNA|RNA|CO2|H2O|theorem|formula|equation)\b'],
        'name': 'Science'
    },
    3: {  # Medicine
        'keywords': ['health', 'medical', 'patient', 'doctor', 'hospital', 'treatment',
                    'disease', 'symptom', 'diagnosis', 'therapy', 'medicine', 'drug',
                    'vaccine', 'surgery', 'clinic', 'healthcare', 'pharmaceutical'],
        'patterns': [r'\b(mg|ml|dose|syndrome|disorder|virus|bacteria)\b'],
        'name': 'Medicine'
    },
    4: {  # History
        'keywords': ['historical', 'century', 'ancient', 'medieval', 'war', 'empire',
                    'civilization', 'revolution', 'independence', 'battle', 'conquest',
                    'dynasty', 'era', 'period', 'archaeological', 'monument'],
        'patterns': [r'\b(\d{1,4}\s*(BC|AD|CE|BCE)|\d{4}s?)\b'],
        'name': 'History'
    },
    5: {  # Geography
        'keywords': ['mountain', 'river', 'ocean', 'continent', 'country', 'city',
                    'climate', 'geological', 'terrain', 'landscape', 'region',
                    'capital', 'border', 'coastline', 'valley', 'desert'],
        'patterns': [r'\b(km|miles|altitude|latitude|longitude|degrees)\b'],
        'name': 'Geography'
    },
    6: {  # Literature
        'keywords': ['novel', 'poem', 'author', 'writer', 'book', 'story', 'character',
                    'plot', 'narrative', 'poetry', 'fiction', 'literary', 'writing',
                    'publication', 'manuscript', 'verse', 'prose', 'genre'],
        'patterns': [r'\bpublished\s+in\s+\d{4}\b'],
        'name': 'Literature'
    },
    7: {  # Art
        'keywords': ['painting', 'sculpture', 'artist', 'museum', 'gallery', 'artwork',
                    'canvas', 'brush', 'color', 'style', 'artistic', 'creative',
                    'masterpiece', 'exhibition', 'portrait', 'landscape'],
        'patterns': [r'\b(oil\s+on\s+canvas|watercolor|acrylic)\b'],
        'name': 'Art'
    },
    8: {  # Sports
        'keywords': ['game', 'team', 'player', 'sport', 'championship', 'tournament',
                    'match', 'score', 'victory', 'competition', 'athlete', 'coach',
                    'season', 'league', 'stadium', 'field', 'court'],
        'patterns': [r'\b(\d+-\d+|points|goals|wins|losses)\b'],
        'name': 'Sports'
    },
    9: {  # Entertainment
        'keywords': ['movie', 'film', 'actor', 'actress', 'director', 'cinema',
                    'television', 'show', 'series', 'episode', 'music', 'song',
                    'album', 'concert', 'performance', 'entertainment'],
        'patterns': [r'\b(box\s+office|starring|featuring|soundtrack)\b'],
        'name': 'Entertainment'
    },
    10: {  # Business
        'keywords': ['company', 'business', 'corporation', 'profit', 'revenue',
                     'market', 'industry', 'economy', 'financial', 'investment',
                     'management', 'entrepreneur', 'startup', 'commerce'],
        'patterns': [r'\b(CEO|CFO|IPO|\$\d+|billion|million)\b'],
        'name': 'Business'
    },
    11: {  # Politics
        'keywords': ['government', 'political', 'president', 'minister', 'election',
                     'vote', 'democracy', 'parliament', 'congress', 'policy',
                     'law', 'legislation', 'constitution', 'citizen'],
        'patterns': [r'\b(elected|voted|policy|amendment)\b'],
        'name': 'Politics'
    },
    12: {  # Religion
        'keywords': ['religious', 'faith', 'church', 'temple', 'mosque', 'prayer',
                     'worship', 'spiritual', 'sacred', 'holy', 'divine',
                     'ceremony', 'ritual', 'belief', 'doctrine'],
        'patterns': [r'\b(God|Allah|Buddha|prophet|scripture)\b'],
        'name': 'Religion'
    },
    13: {  # Food
        'keywords': ['food', 'recipe', 'cooking', 'cuisine', 'restaurant',
                     'ingredient', 'flavor', 'taste', 'dish', 'meal',
                     'kitchen', 'chef', 'nutrition', 'diet'],
        'patterns': [r'\b(grams|cups|tablespoons|degrees|oven)\b'],
        'name': 'Food'
    },
    14: {  # Education
        'keywords': ['school', 'university', 'education', 'student', 'teacher',
                     'professor', 'learning', 'academic', 'degree', 'curriculum',
                     'scholarship', 'graduation', 'classroom', 'lecture'],
        'patterns': [r'\b(PhD|bachelor|master|diploma|GPA)\b'],
        'name': 'Education'
    },
    15: {  # Nature
        'keywords': ['nature', 'wildlife', 'animal', 'plant', 'forest', 'ecosystem',
                     'species', 'habitat', 'conservation', 'biodiversity',
                     'environment', 'natural', 'organic', 'botanical'],
        'patterns': [r'\b(species|genus|family|phylum)\b'],
        'name': 'Nature'
    },
    16: {  # Social
        'keywords': ['social', 'society', 'community', 'culture', 'tradition',
                     'custom', 'behavior', 'relationship', 'family', 'friendship',
                     'human', 'people', 'group', 'interaction'],
        'patterns': [r'\b(community|cultural|traditional|social)\b'],
        'name': 'Social'
    }
}

# Task patterns (32 tasks, 1-32)
TASK_PATTERNS = {
    1: {  # Fact Retrieval
        'keywords': ['is', 'was', 'are', 'were', 'located', 'born', 'died', 'founded',
                    'discovered', 'invented', 'created', 'established', 'occurred'],
        'patterns': [r'\b(when|where|who|what|which)\b', r'\bin\s+\d{4}\b'],
        'name': 'Fact Retrieval'
    },
    2: {  # Definition
        'keywords': ['definition', 'meaning', 'refers to', 'defined as', 'known as',
                    'called', 'termed', 'concept', 'principle'],
        'patterns': [r'\bis\s+(a|an|the)\s+\w+'],
        'name': 'Definition'
    },
    3: {  # Comparison
        'keywords': ['compared to', 'versus', 'different from', 'similar to', 'contrast',
                    'difference', 'similarity', 'alike', 'unlike', 'than'],
        'patterns': [r'\b(vs|versus|compared|than|while|whereas)\b'],
        'name': 'Comparison'
    },
    4: {  # Process Description
        'keywords': ['process', 'procedure', 'method', 'steps', 'stages', 'phases',
                    'sequence', 'workflow', 'algorithm', 'technique'],
        'patterns': [r'\b(first|then|next|finally|step\s+\d+)\b'],
        'name': 'Process Description'
    },
    5: {  # Cause and Effect
        'keywords': ['because', 'due to', 'caused by', 'results in', 'leads to',
                    'effect', 'consequence', 'impact', 'influence', 'reason'],
        'patterns': [r'\b(because|since|due\s+to|results?\s+in)\b'],
        'name': 'Cause and Effect'
    },
    6: {  # Classification
        'keywords': ['type', 'category', 'class', 'kind', 'variety', 'species',
                    'genre', 'classification', 'taxonomy', 'group'],
        'patterns': [r'\b(types?\s+of|kinds?\s+of|categories)\b'],
        'name': 'Classification'
    },
    7: {  # Measurement
        'keywords': ['measure', 'size', 'weight', 'height', 'length', 'area',
                    'volume', 'temperature', 'speed', 'distance', 'quantity'],
        'patterns': [r'\b(\d+\.?\d*\s*(cm|km|kg|meters?|feet|inches?))\b'],
        'name': 'Measurement'
    },
    8: {  # Analysis
        'keywords': ['analysis', 'examine', 'study', 'research', 'investigate',
                    'evaluate', 'assess', 'review', 'interpret', 'analyze'],
        'patterns': [r'\b(according\s+to|research\s+shows|studies\s+indicate)\b'],
        'name': 'Analysis'
    },
    9: {  # Summarization
        'keywords': ['summary', 'overview', 'brief', 'outline', 'synopsis',
                    'abstract', 'conclusion', 'key points', 'main ideas'],
        'patterns': [r'\b(in\s+summary|overall|key\s+points?)\b'],
        'name': 'Summarization'
    },
    10: {  # Instruction
        'keywords': ['how to', 'instructions', 'guide', 'tutorial', 'manual',
                     'directions', 'steps', 'procedure', 'method'],
        'patterns': [r'\b(how\s+to|step\s+by\s+step|follow\s+these)\b'],
        'name': 'Instruction'
    }
}

# Modifier patterns (32 modifiers, 1-32)
MODIFIER_PATTERNS = {
    27: {  # Descriptive (most common fallback)
        'keywords': ['descriptive', 'detailed', 'comprehensive', 'thorough'],
        'name': 'Descriptive'
    },
    1: {  # Factual
        'keywords': ['fact', 'factual', 'objective', 'accurate', 'precise',
                    'exact', 'specific', 'concrete', 'verifiable'],
        'name': 'Factual'
    },
    2: {  # Analytical
        'keywords': ['analytical', 'critical', 'systematic', 'methodical',
                    'logical', 'rational', 'scientific'],
        'name': 'Analytical'
    },
    3: {  # Historical
        'keywords': ['historical', 'chronological', 'temporal', 'past',
                    'ancient', 'traditional', 'heritage'],
        'name': 'Historical'
    },
    4: {  # Technical
        'keywords': ['technical', 'specialized', 'expert', 'professional',
                    'advanced', 'complex', 'sophisticated'],
        'name': 'Technical'
    },
    5: {  # Practical
        'keywords': ['practical', 'applied', 'useful', 'functional',
                    'hands-on', 'real-world', 'actionable'],
        'name': 'Practical'
    }
}


def score_domain(text: str) -> Dict[int, float]:
    """Score text against all domain patterns."""
    text_lower = text.lower()
    scores = {}

    for domain_id, domain_info in DOMAIN_PATTERNS.items():
        score = 0.0

        # Score keywords
        for keyword in domain_info['keywords']:
            if keyword.lower() in text_lower:
                score += 1.0

        # Score patterns
        for pattern in domain_info.get('patterns', []):
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 2.0  # Patterns get higher weight

        scores[domain_id] = score

    return scores


def score_task(text: str) -> Dict[int, float]:
    """Score text against all task patterns."""
    text_lower = text.lower()
    scores = {}

    for task_id, task_info in TASK_PATTERNS.items():
        score = 0.0

        # Score keywords
        for keyword in task_info['keywords']:
            if keyword.lower() in text_lower:
                score += 1.0

        # Score patterns
        for pattern in task_info.get('patterns', []):
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 2.0

        scores[task_id] = score

    return scores


def score_modifier(text: str) -> Dict[int, float]:
    """Score text against all modifier patterns."""
    text_lower = text.lower()
    scores = {}

    for modifier_id, modifier_info in MODIFIER_PATTERNS.items():
        score = 0.0

        # Score keywords
        for keyword in modifier_info['keywords']:
            if keyword.lower() in text_lower:
                score += 1.0

        scores[modifier_id] = score

    return scores


def extract_tmd_from_text(text: str) -> Dict[str, Any]:
    """
    Extract TMD codes from text using content analysis.

    Returns:
        dict with domain_code, task_code, modifier_code, and names
    """
    if not text or not text.strip():
        # Return default only for empty text
        return {
            'domain_code': 9,
            'task_code': 1,
            'modifier_code': 27,
            'domain': 'Literature',
            'task': 'Fact Retrieval',
            'modifier': 'Descriptive',
            'confidence': 0.0
        }

    # Score all categories
    domain_scores = score_domain(text)
    task_scores = score_task(text)
    modifier_scores = score_modifier(text)

    # Find best matches
    best_domain = max(domain_scores.items(), key=lambda x: x[1])
    best_task = max(task_scores.items(), key=lambda x: x[1])
    best_modifier = max(modifier_scores.items(), key=lambda x: x[1])

    # Use fallbacks if no good matches
    domain_code = best_domain[0] if best_domain[1] > 0 else 2  # Default to Science
    task_code = best_task[0] if best_task[1] > 0 else 1  # Default to Fact Retrieval
    modifier_code = best_modifier[0] if best_modifier[1] > 0 else 1  # Default to Factual

    # Calculate confidence
    total_score = best_domain[1] + best_task[1] + best_modifier[1]
    confidence = min(total_score / 10.0, 1.0)  # Normalize to 0-1

    return {
        'domain_code': domain_code,
        'task_code': task_code,
        'modifier_code': modifier_code,
        'domain': DOMAIN_PATTERNS[domain_code]['name'],
        'task': TASK_PATTERNS[task_code]['name'],
        'modifier': MODIFIER_PATTERNS[modifier_code]['name'],
        'confidence': round(confidence, 2),
        'scores': {
            'domain': dict(domain_scores),
            'task': dict(task_scores),
            'modifier': dict(modifier_scores)
        }
    }


def batch_extract_tmd(texts: List[str]) -> List[Dict[str, Any]]:
    """Extract TMD codes from multiple texts."""
    return [extract_tmd_from_text(text) for text in texts]


def analyze_tmd_distribution(tmd_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze distribution of TMD codes in batch results."""
    domain_counts = Counter()
    task_counts = Counter()
    modifier_counts = Counter()
    confidence_scores = []

    for result in tmd_results:
        domain_counts[f"{result['domain_code']}.{result['domain']}"] += 1
        task_counts[f"{result['task_code']}.{result['task']}"] += 1
        modifier_counts[f"{result['modifier_code']}.{result['modifier']}"] += 1
        confidence_scores.append(result['confidence'])

    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    return {
        'total_items': len(tmd_results),
        'unique_domains': len(domain_counts),
        'unique_tasks': len(task_counts),
        'unique_modifiers': len(modifier_counts),
        'avg_confidence': round(avg_confidence, 2),
        'domain_distribution': dict(domain_counts.most_common(10)),
        'task_distribution': dict(task_counts.most_common(10)),
        'modifier_distribution': dict(modifier_counts.most_common(10))
    }


if __name__ == "__main__":
    # Test the TMD extractor
    test_texts = [
        "Photosynthesis converts light energy into chemical energy in plants.",
        "Ada Lovelace is regarded as the first computer programmer.",
        "The Eiffel Tower was completed in 1889 for the Paris World's Fair.",
        "Mount Everest is the tallest mountain in the world at 8,848 meters.",
        "Shakespeare wrote Romeo and Juliet in the late 16th century.",
        "Python is a programming language used for web development and data analysis.",
        "The human heart pumps blood through the circulatory system.",
        "World War II ended in 1945 with the surrender of Germany and Japan."
    ]

    print("Testing TMD Extractor V2")
    print("-" * 50)

    results = batch_extract_tmd(test_texts)

    for i, (text, result) in enumerate(zip(test_texts, results)):
        print(f"Text {i+1}: {text[:50]}...")
        print(f"TMD: {result['domain_code']}.{result['task_code']}.{result['modifier_code']}")
        print(f"Domain: {result['domain']} (code {result['domain_code']})")
        print(f"Task: {result['task']} (code {result['task_code']})")
        print(f"Modifier: {result['modifier']} (code {result['modifier_code']})")
        print(f"Confidence: {result['confidence']}")
        print()

    # Analyze distribution
    analysis = analyze_tmd_distribution(results)
    print("Distribution Analysis:")
    print(f"Unique domains: {analysis['unique_domains']}")
    print(f"Unique tasks: {analysis['unique_tasks']}")
    print(f"Unique modifiers: {analysis['unique_modifiers']}")
    print(f"Average confidence: {analysis['avg_confidence']}")
    print()
    print("Top domains:", list(analysis['domain_distribution'].keys())[:3])
    print("Top tasks:", list(analysis['task_distribution'].keys())[:3])