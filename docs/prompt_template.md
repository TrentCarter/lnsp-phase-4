# LNSP Prompt Templates (CPESH-Enhanced)

## QUERY (lane-aware with contrastive routing)
You are the LNSP retriever. Task: return **IDs** + brief rationales with contrastive awareness.
Inputs:
- lane: {L1_FACTOID | L2_GRAPH | L3_SYNTH}
- user_query: <text>
- constraints: {top_k:int, distance_metric, retrieval_mode, hard_neg_threshold:0.3}
- hints: optional entity list
- cpesh_check: boolean (enable contrastive filtering)

Rules:
- If lane=L1_FACTOID -> prefer dense; return top_k by cosine, include scores.
- If lane=L2_GRAPH -> expand via KGEdgeType in [MENTIONS, LINKS_TO, IS_A] up to 2 hops.
- If lane=L3_SYNTH -> return a balanced mix: 70% dense / 30% graph.
- If cpesh_check=true -> filter results where hard_negative_distance < threshold

Output JSON:
{
  "items":[{"id": "...", "score": 0.883, "why": "...", "soft_neg_dist": 0.65, "hard_neg_dist": 0.32}],
  "lane":"...",
  "mode":"...",
  "filtered_count": 0
}

## CPESH_EXTRACT
Given a factoid, extract concept-probe-expected with soft and hard negatives:

Instructions:
1. Identify the core concept from the factoid
2. Generate a probe question that targets this concept
3. Extract the expected answer from the factoid
4. Create a soft negative: semantically related but incorrect (same domain, wrong detail)
5. Create a hard negative: unrelated or misleading (different domain or concept)

Output JSON:
{
  "concept": "...",
  "probe": "...",
  "expected": "...",
  "soft_negative": "...",  // Semantically adjacent but incorrect
  "hard_negative": "...",  // Unrelated or misleading
  "confidence": 0.95
}

Examples:
- Soft negative: "Calvin cycle" when expected is "Light-dependent reactions"
- Hard negative: "Mitochondrial respiration" for a photosynthesis concept

## EDGE_EXTRACT
Given a factoid, extract entities and relations:
Output JSON:
{ "entities": ["..."], "edges":[{"src":"...","rel":"IS_A","dst":"..."}] }

## SUMMARIZE
Given N factoids, produce a 4-sentence neutral summary. No speculation.

## CONTRASTIVE_TRAINING_BATCH
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
