# LNSP Prompt Templates (CPESH-Enhanced)

> **Output policy**: Respond with **only** a single JSON object that conforms to the Output JSON
> schema of the current section. Do not include prose, explanations, or code fences. If any required
> field cannot be justified from the provided inputs/context, set `insufficient_evidence` to `true`
> and set that field to `null`.
>
> **Scoring convention**: All scores use **cosine similarity** on L2-normalized 768D vectors
> (FAISS IP ≡ cosine). If the word *distance* appears anywhere, interpret it as
> `distance = 1 − cosine_similarity`.

## QUERY (lane-aware with contrastive routing)
You are the LNSP retriever. Task: return **IDs** + brief rationales with contrastive awareness.
Inputs:
- lane: {L1_FACTOID | L2_GRAPH | L3_SYNTH}
- user_query: <text>
- constraints: {top_k:int, similarity_metric=cosine, retrieval_mode, hard_neg_threshold_sim:0.7}
- hints: optional entity list
- cpesh_check: boolean (enable contrastive filtering)

Rules:
- If lane=L1_FACTOID -> prefer dense; return top_k by cosine, include scores.
- If lane=L2_GRAPH -> expand via KGEdgeType in [MENTIONS, LINKS_TO, IS_A] up to 2 hops.
- If lane=L3_SYNTH -> return a balanced mix: 70% dense / 30% graph.
- If cpesh_check=true -> filter results where `hard_neg_sim >= hard_neg_threshold_sim`.

Output JSON:
{
  "items":[
    {
      "id": "...",                // stable doc_id, not positional
      "score": 0.883,
      "why":  "...",
      "soft_neg_sim": 0.65,       // cosine similarity to soft negative (or null)
      "hard_neg_sim": 0.32        // cosine similarity to hard negative (or null)
    }
  ],
  "lane":"...",
  "mode":"...",
  "filtered_count": 0,
  "diagnostics": {"cpesh_used": false, "n_graph_expanded": 0, "n_filtered": 0}
}

## CPESH_EXTRACT
> **Return JSON only**. Use `null` for unknowns; set `insufficient_evidence=true` if fields cannot be justified.
Given a factoid, extract concept-probe-expected with soft and hard negatives:

Instructions:
1. Identify the core concept from the factoid.
2. Generate a probe question that targets this concept.
3. Extract the expected answer from the factoid.
4. Create a **soft negative**: same domain/sibling concept, semantically related but **incorrect**.
5. Create a **hard negative**: different domain or misleadingly unrelated concept.
6. Do **not** invent content beyond provided inputs/context. If a negative equals (or paraphrases) the expected,
   set that negative to `null` and set `insufficient_evidence=true`.

Output JSON:
{
  "concept": "...",
  "probe": "...",
  "expected": "...",
  "soft_negative": "...",         // Semantically adjacent but incorrect
  "hard_negative": "...",         // Unrelated or misleading
  "insufficient_evidence": false
}

Examples:
- Soft negative: "Calvin cycle" when expected is "Light-dependent reactions"
- Hard negative: "Mitochondrial respiration" for a photosynthesis concept

## EDGE_EXTRACT
> **Return JSON only**. Use `null` for unknowns; set `insufficient_evidence=true` if fields cannot be justified.
> Allowed `rel` values: `MENTIONS`, `LINKS_TO`, `IS_A`, `CAUSES`, `REQUIRES`, `ENABLES`.
> If none fit, return with `insufficient_evidence=true`.
Given a factoid, extract entities and relations:
Output JSON:
{ "entities": ["..."], "edges":[{"src":"...","rel":"IS_A","dst":"..."}] }

## TMD_EXTRACT
> **Return JSON only**. Use `null` for unknowns; set `insufficient_evidence=true` if fields cannot be justified.
Classify the factoid into TMD and lane.

**Output JSON**
```json
{
  "domain": "<one of: Science, Technology, History, Geography, Biology, Chemistry, Physics, Arts, Culture, Other>",
  "task": "<one of: Fact Retrieval, Definition, Comparison, Explanation, Procedure>",
  "modifier": "<e.g., Biochemical, Thermodynamics, Quantitative, Taxonomic>",
  "tmd_code": "D.T.M",
  "lane_index": 0,
  "insufficient_evidence": false
}
```
Rules:
- Infer strictly from the provided text/context; if unclear, set `insufficient_evidence=true` and `tmd_code=null`.
- `tmd_code` must be formatted `D.T.M` and map to the offline 16-bit encoder (the model must not invent unseen codes).

## SUMMARIZE
> **Return JSON only**. Use `null` for unknowns; set `insufficient_evidence=true` if fields cannot be justified.
Given N factoids, produce a **4-sentence** neutral summary. No speculation, no bullet points, no citations.

## CONTRASTIVE_TRAINING_BATCH
> **Return JSON only**. Use `null` for unknowns; set `insufficient_evidence=true` if fields cannot be justified.
Generate training triplets/quadruplets from CPESH data for contrastive learning:

Input: Array of CPESH objects
Output JSON:
{
  "triplets": [
    {
      "anchor": "concept_text",
      "positive": "expected_answer",
      "negative": "soft_negative",
      "margin": 0.3
    }
  ],
  "quadruplets": [
    {
      "anchor": "concept_text",
      "positive": "expected_answer",
      "soft_neg": "soft_negative",
      "hard_neg": "hard_negative",
      "soft_margin": 0.3,
      "hard_margin": 0.7
    }
  ]
}

## CPESH_EXAMPLES

### Science Domain
```json
{
  "concept": "ATP synthesis in cellular respiration",
  "probe": "Where is most ATP produced during cellular respiration?",
  "expected": "Mitochondrial electron transport chain",
  "soft_negative": "Krebs cycle",
  "hard_negative": "Chloroplast thylakoid membrane"
}
```

### Technology Domain
```json
{
  "concept": "REST API architectural constraints",
  "probe": "What makes an API RESTful?",
  "expected": "Stateless client-server communication with uniform interface",
  "soft_negative": "HTTP-based web services",
  "hard_negative": "SQL database queries"
}
```

### History Domain
```json
{
  "concept": "Industrial Revolution catalyst",
  "probe": "What primarily triggered the Industrial Revolution in Britain?",
  "expected": "Steam engine innovation and coal availability",
  "soft_negative": "Agricultural revolution",
  "hard_negative": "Renaissance art movement"
}
```

## SEMANTIC_GPS_ROUTING
Route queries based on contrastive distances:

```python
def semantic_gps_route(query_embedding, cpesh_data):
    # Calculate distances
    expected_sim = cosine_similarity(query_embedding, cpesh_data.expected_embedding)
    soft_neg_sim = cosine_similarity(query_embedding, cpesh_data.soft_negative_embedding)
    hard_neg_sim = cosine_similarity(query_embedding, cpesh_data.hard_negative_embedding)

    # Routing decision tree
    if hard_neg_sim > 0.7:
        return {"route": "REJECT", "reason": "Too similar to hard negative"}
    elif soft_neg_sim > expected_sim:
        return {"route": "CLARIFY", "reason": "Ambiguous - closer to soft negative"}
    elif expected_sim > 0.82:
        return {"route": "DIRECT", "confidence": "HIGH"}
    elif expected_sim > 0.65:
        return {"route": "ENSEMBLE", "confidence": "MEDIUM"}
    else:
        return {"route": "EXPLORATORY", "confidence": "LOW"}
```
