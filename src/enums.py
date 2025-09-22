from enum import StrEnum, auto

class Lane(StrEnum):
    L0_RAW = auto()
    L1_FACTOID = auto()
    L2_GRAPH = auto()
    L3_SYNTH = auto()

class SourceType(StrEnum):
    WIKI = auto(); PDF = auto(); WEB = auto(); NOTE = auto(); MANUAL = auto(); UNKNOWN = auto()

class ChunkStatus(StrEnum):
    NEW = auto(); NORMALIZED = auto(); TMD_ENCODED = auto(); VECTORIZED = auto(); INDEXED = auto(); ERROR = auto()

class EmbedModel(StrEnum):
    GTR_T5_BASE_768 = auto()
    MINI_LM_L6_V2_384 = auto()
    BGE_SMALL_EN_V1_5_384 = auto()
    BGE_BASE_EN_V1_5_768 = auto()

class DistanceMetric(StrEnum):
    COSINE = auto(); IP = auto(); L2 = auto()

class RetrievalMode(StrEnum):
    DENSE = auto(); GRAPH = auto(); HYBRID = auto()

class KGEdgeType(StrEnum):
    MENTIONS = auto(); LINKS_TO = auto(); IS_A = auto(); PART_OF = auto(); ALIASES = auto(); DERIVES_FROM = auto()

class LightRAGMode(StrEnum):
    LOW_LEVEL = auto(); HIGH_LEVEL = auto(); HYBRID = auto()

class PromptTemplateType(StrEnum):
    QUERY = auto(); SUMMARIZE = auto(); EDGE_EXTRACT = auto(); EVAL = auto()

class EvalLabel(StrEnum):
    PASS = auto(); FAIL = auto(); FLAKY = auto()

def choices(enum_cls):  # convenience for FastAPI/OpenAPI
    return [e.value for e in enum_cls]

Uses StrEnum (Python 3.11). If you need pure strings for DB constraints, these are deterministic.
