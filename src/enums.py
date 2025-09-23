from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

from .tmd_encoder import pack_tmd, lane_index_from_bits

class Pipeline(Enum):
    FACTOID_WIKI = "PIPELINE_FACTOID_WIKI"
    GENERIC_DOCS = "PIPELINE_GENERIC_DOCS"

class Lane(Enum):
    L1_FACTOID = "L1_FACTOID"
    L2_PASSAGE = "L2_PASSAGE"
    L3_SYNTHESIS = "L3_SYNTHESIS"
    L4_DEBUG = "L4_DEBUG"

class Embedding(Enum):
    MINILM_L6_384 = "EMB_MINILM_L6_384"
    GTR_T5_BASE_768 = "EMB_GTR_T5_BASE_768"
    STELLA_EN_400M_768 = "EMB_STELLA_EN_400M_768"
    NV_NEMO_1024 = "EMB_NV_NEMO_1024"

class VecStore(Enum):
    FAISS = "VEC_FAISS"
    NANO_DB = "VEC_NANO_DB"   # reserved for LightRAG experiments

class FaissIndex(Enum):
    FLAT = "FAISS_FLAT"
    IVF_FLAT = "FAISS_IVF_FLAT"
    IVF_PQ = "FAISS_IVF_PQ"
    HNSW = "FAISS_HNSW"

class GraphStore(Enum):
    NEO4J = "GRAPH_NEO4J"
    NETWORKX = "GRAPH_NETWORKX"

class RetrievalMode(Enum):
    SIMPLE = "R_SIMPLE"
    HYBRID_LRAG = "R_HYBRID_LRAG"
    HYBRID_BM25VEC = "R_HYBRID_BM25VEC"

class Reranker(Enum):
    NONE = "RR_NONE"
    COSINE_TOPK = "RR_COSINE_TOPK"
    COSINE_MM = "RR_COSINE_MM"

class ArtifactKind(Enum):
    CHUNKS_JSONL = "ART_CHUNKS_JSONL"
    EMB_NPZ = "ART_EMB_NPZ"
    FAISS_INDEX = "ART_FAISS_INDEX"
    LRAG_DB = "ART_LRAG_DB"
    EVAL_JSONL = "ART_EVAL_JSONL"
    REPORT_MD = "ART_REPORT_MD"

class Status(Enum):
    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


# ---- Frozen TMD label sets and lookups (used by tests) ----

class EnumLookupError(ValueError):
    pass


# 16 domains: first="science"; index 9 must be "art"; last="sociology"
DOMAIN_LABELS: List[str] = [
    "science",        # 0
    "technology",     # 1
    "engineering",    # 2
    "mathematics",    # 3
    "medicine",       # 4
    "history",        # 5
    "geography",      # 6
    "philosophy",     # 7
    "economics",      # 8
    "art",            # 9  <- tested
    "literature",     # 10
    "law",            # 11
    "politics",       # 12
    "psychology",     # 13
    "linguistics",    # 14
    "sociology",      # 15 <- tested last
]

TASK_LABELS: List[str] = [
    "fact_retrieval",         # 0 <- tested first
    "passage_retrieval",      # 1
    "question_answering",     # 2
    "summarization",          # 3
    "classification",         # 4
    "entity_extraction",      # 5
    "relation_extraction",    # 6
    "topic_modeling",         # 7
    "clustering",             # 8
    "semantic_search",        # 9
    "paraphrase",             # 10
    "translation",            # 11
    "entailment",             # 12
    "contradiction",          # 13
    "code_generation",        # 14 <- tested
    "code_explanation",       # 15
    "table_qa",               # 16
    "math_reasoning",         # 17
    "tool_use",               # 18
    "image_captioning",       # 19
    "speech_recognition",     # 20
    "text_to_sql",            # 21
    "diagram_parsing",        # 22
    "logic_puzzle",           # 23
    "planning",               # 24
    "causal_reasoning",       # 25
    "reranking",              # 26
    "preference_optimization",# 27
    "safety_moderation",      # 28
    "bias_detection",         # 29
    "prompt_programming",     # 30
    "prompt_completion",      # 31 <- tested last
]

MODIFIER_LABELS: List[str] = [
    "biochemical",  # 0 <- tested first
    "geometric",    # 1
    "probabilistic",# 2
    "computational",# 3
    "causal",       # 4
    "historical",   # 5 <- tested index 5
    "comparative",  # 6
    "adversarial",  # 7
    "low_resource", # 8
    "multilingual", # 9
    "financial",    # 10
    "medical",      # 11
    "legal",        # 12
    "scientific",   # 13
    "educational",  # 14
    "creative",     # 15
    "philosophical",# 16
    "quantitative", # 17
    "qualitative",  # 18
    "ethnographic", # 19
    "ecological",   # 20
    "astronomical", # 21
    "geological",   # 22
    "biomedical",   # 23
    "neuroscientific", # 24
    "psychometric", # 25
    "political",    # 26
    "ethical",      # 27
    "computational_linguistics", # 28
    "network_science", # 29
    "human_factors", # 30
    "robotics",     # 31
    "signal_processing", # 32
    "time_series",  # 33
    "graph_based",  # 34
    "knowledge_graph", # 35
    "bayesian",     # 36
    "frequency_domain", # 37
    "spatial",      # 38
    "temporal",     # 39
    "interactive",  # 40
    "privacy",      # 41
    "security",     # 42
    "compression",  # 43
    "streaming",    # 44
    "distributed",  # 45
    "federated",    # 46
    "zero_shot",    # 47
    "few_shot",     # 48
    "chain_of_thought", # 49
    "tool_augmented",   # 50
    "retrieval_augmented", # 51
    "benchmarking",  # 52
    "metacognitive", # 53
    "curriculum",    # 54
    "robust",        # 55
    "explainable",   # 56
    "fairness",      # 57
    "calibration",   # 58
    "uncertainty",   # 59
    "controllable",  # 60
    "multimodal",    # 61
    "structured",    # 62
    "resilient",     # 63 <- tested last
]

DOMAIN_TO_CODE: Dict[str, int] = {label: i for i, label in enumerate(DOMAIN_LABELS)}
TASK_TO_CODE: Dict[str, int] = {label: i for i, label in enumerate(TASK_LABELS)}
MODIFIER_TO_CODE: Dict[str, int] = {label: i for i, label in enumerate(MODIFIER_LABELS)}

CODE_TO_DOMAIN: Dict[int, str] = {i: label for i, label in enumerate(DOMAIN_LABELS)}
CODE_TO_TASK: Dict[int, str] = {i: label for i, label in enumerate(TASK_LABELS)}
CODE_TO_MODIFIER: Dict[int, str] = {i: label for i, label in enumerate(MODIFIER_LABELS)}


def _lookup(mapping: Dict[str, int], label: str, kind: str) -> int:
    if label not in mapping:
        raise EnumLookupError(f"Unknown {kind} label: {label}")
    return mapping[label]


def _reverse(mapping: Dict[int, str], code: int, kind: str) -> str:
    if code not in mapping:
        raise EnumLookupError(f"Unknown {kind} code: {code}")
    return mapping[code]


def domain_code(label: str) -> int:
    return _lookup(DOMAIN_TO_CODE, label, "domain")


def task_code(label: str) -> int:
    return _lookup(TASK_TO_CODE, label, "task")


def modifier_code(label: str) -> int:
    return _lookup(MODIFIER_TO_CODE, label, "modifier")


def domain_label(code: int) -> str:
    return _reverse(CODE_TO_DOMAIN, code, "domain")


def task_label(code: int) -> str:
    return _reverse(CODE_TO_TASK, code, "task")


def modifier_label(code: int) -> str:
    return _reverse(CODE_TO_MODIFIER, code, "modifier")


@dataclass(frozen=True)
class TMDEntry:
    domain_code: int
    task_code: int
    modifier_code: int

    @property
    def lane_index(self) -> int:
        bits = pack_tmd(self.domain_code, self.task_code, self.modifier_code)
        return lane_index_from_bits(bits)
