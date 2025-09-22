import numpy as np

from integrations import Triple, ingest_triples
from integrations.lightrag import (
    LightRAGConfig,
    LightRAGGraphBuilderAdapter,
    LightRAGHybridRetriever,
)


class TestLightRAGGraphAdapter:
    def test_confidence_injection(self):
        config = LightRAGConfig(
            graph_enabled=True,
            relation_confidence_floor=0.42,
            relation_source_tag="lightrag",
        )
        adapter = LightRAGGraphBuilderAdapter.from_config(config)

        cpe_record = {
            "cpe_id": "cpe-1",
            "concept_text": "Photosynthesis converts light",
        }

        relations = adapter.enhance_relations(
            cpe_record,
            [
                {
                    "subj": "photosynthesis",
                    "pred": "produces",
                    "obj": "oxygen",
                }
            ],
        )

        assert len(relations) == 1
        relation = relations[0]
        assert relation["confidence"] >= 0.42
        assert relation["source"] == "lightrag"
        assert relation["text"].startswith("photosynthesis produces")


class TestLightRAGHybridRetriever:
    def test_search_with_fallback(self):
        config = LightRAGConfig(query_enabled=True, query_weight=0.5)
        adapter = LightRAGHybridRetriever.from_config(config, dim=4)

        adapter.register_document([1.0, 0.0, 0.0, 0.0], {"cpe_id": "a"})
        adapter.register_document([0.0, 1.0, 0.0, 0.0], {"cpe_id": "b"})

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        fallback = [
            {"cpe_id": "b", "score": 0.75, "rank": 1, "retriever": "faiss"}
        ]

        results = adapter.search(query_vector=query, top_k=2, fallback_candidates=fallback)

        assert results
        assert results[0]["cpe_id"] == "a"
        assert any(candidate["cpe_id"] == "b" for candidate in results)


def test_ingest_triples_lane_filtering():
    triples = [
        Triple(src_cpe_id="src", dst_cpe_id="dst1", properties={"lane_index": 10}),
        Triple(src_cpe_id="src", dst_cpe_id="dst2", properties={"lane_index": 20}),
    ]

    collected = []

    def writer(triple: Triple) -> None:
        collected.append(triple)

    count = ingest_triples(triples, lane_index=12, lane_window=5, writer=writer)

    assert count == 1
    assert len(collected) == 1
    assert collected[0].dst_cpe_id == "dst1"
