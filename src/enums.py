from enum import Enum

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
