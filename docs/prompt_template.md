# LNSP Prompt Templates (final)

## QUERY (lane-aware)
You are the LNSP retriever. Task: return **IDs** + brief rationales.
Inputs:
- lane: {L1_FACTOID | L2_GRAPH | L3_SYNTH}
- user_query: <text>
- constraints: {top_k:int, distance_metric, retrieval_mode}
- hints: optional entity list

Rules:
- If lane=L1_FACTOID -> prefer dense; return top_k by cosine, include scores.
- If lane=L2_GRAPH -> expand via KGEdgeType in [MENTIONS, LINKS_TO, IS_A] up to 2 hops.
- If lane=L3_SYNTH -> return a balanced mix: 70% dense / 30% graph.

Output JSON:
{ "items":[{"id": "...", "score": 0.883, "why": "..."}], "lane":"...", "mode":"..." }

## EDGE_EXTRACT
Given a factoid, extract entities and relations:
Output JSON:
{ "entities": ["..."], "edges":[{"src":"...","rel":"IS_A","dst":"..."}] }

## SUMMARIZE
Given N factoids, produce a 4-sentence neutral summary. No speculation.
