# LNSP Enums (frozen 2025-09-22)

## Lane
- L0_RAW          # raw text/files pre-normalization
- L1_FACTOID      # curated factoids (10k curated set)
- L2_GRAPH        # KG entity/edge nodes
- L3_SYNTH        # synthesized answers/summaries

## SourceType
- WIKI - PDF - WEB - NOTE - MANUAL - UNKNOWN

## ChunkStatus
- NEW - NORMALIZED - TMD_ENCODED - VECTORIZED - INDEXED - ERROR

## EmbedModel
- GTR_T5_BASE_768
- MINI_LM_L6_V2_384
- BGE_SMALL_EN_V1_5_384
- BGE_BASE_EN_V1_5_768

## DistanceMetric
- COSINE - IP - L2

## RetrievalMode
- DENSE - GRAPH - HYBRID

## KGEdgeType
- MENTIONS - LINKS_TO - IS_A - PART_OF - ALIASES - DERIVES_FROM

## LightRAGMode
- LOW_LEVEL - HIGH_LEVEL - HYBRID

## PromptTemplateType
- QUERY - SUMMARIZE - EDGE_EXTRACT - EVAL

## EvalLabel
- PASS - FAIL - FLAKY
