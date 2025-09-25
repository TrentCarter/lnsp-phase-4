"""Regression tests for CPESH cache timestamp behaviour."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, TYPE_CHECKING

import numpy as np
import pytest

from src.schemas import CPESH
from src.utils import timestamps as ts_utils

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from src.api.retrieve import RetrievalContext


@pytest.fixture()
def retrieval_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "RetrievalContext":
    """Provide a RetrievalContext with lightweight stubs and isolated cache file."""

    cache_path = tmp_path / "cpesh_cache.jsonl"
    monkeypatch.setenv("LNSP_CPESH_CACHE", str(cache_path))
    monkeypatch.setenv("LNSP_CPESH_MAX_K", "2")
    monkeypatch.setenv("LNSP_CPESH_TIMEOUT_S", "1.5")

    monkeypatch.setattr(sys, "version_info", (3, 11, 0, "final", 0), raising=False)

    import types

    stub_settings = types.ModuleType("pydantic_settings")

    class _BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    stub_settings.BaseSettings = _BaseSettings
    monkeypatch.setitem(sys.modules, "pydantic_settings", stub_settings)
    retrieve_module = importlib.import_module("src.api.retrieve")
    ctx_cls = retrieve_module.RetrievalContext

    class DummyFaissDB:
        def __init__(self, *_, **__):
            self.cpe_ids = []
            self.doc_ids = []
            self.concept_texts = []

        def load(self, *_args, **_kwargs) -> bool:
            return True

        def search_legacy(self, *_args, **_kwargs):  # pragma: no cover - unused in these tests
            return []

    class DummyAdapter:
        @classmethod
        def from_config(cls, *_, **__):
            return cls()

    class DummyConfig:
        @classmethod
        def from_env(cls):
            return cls()

    class DummyEmbedder:
        def __init__(self, *_, **__):
            self._vec = np.zeros(768, dtype=np.float32)

        def encode(self, texts, batch_size: int | None = None):  # noqa: D401 - simple stub
            if not texts:
                return np.zeros((0, self._vec.shape[0]), dtype=np.float32)
            return np.tile(self._vec, (len(texts), 1))

    monkeypatch.setattr(retrieve_module, "FaissDB", DummyFaissDB)
    monkeypatch.setattr(retrieve_module, "LightRAGHybridRetriever", DummyAdapter)
    monkeypatch.setattr(retrieve_module, "LightRAGConfig", DummyConfig)
    monkeypatch.setattr(retrieve_module, "EmbeddingBackend", DummyEmbedder)

    monkeypatch.setattr(ctx_cls, "_validate_npz_schema", lambda self, path: None)
    monkeypatch.setattr(ctx_cls, "_validate_faiss_dimension", lambda self: None)
    monkeypatch.setattr(ctx_cls, "_load_faiss_index", lambda self: None)

    ctx = ctx_cls(npz_path=str(tmp_path / "dummy.npz"))
    yield ctx
    ctx.close()


def test_put_cpesh_to_cache_records_timestamps(retrieval_context: "RetrievalContext") -> None:
    cpesh = CPESH(concept="Helium", expected="It is a noble gas")
    retrieval_context.put_cpesh_to_cache("doc-1", cpesh)

    entry = retrieval_context.cpesh_cache["doc-1"]
    created = entry["cpesh"].get("created_at")
    accessed = entry["cpesh"].get("last_accessed")

    assert entry["access_count"] == 1
    assert ts_utils.parse_iso_timestamp(created) is not None
    assert ts_utils.parse_iso_timestamp(accessed) is not None


def test_get_cpesh_updates_last_accessed(retrieval_context: "RetrievalContext") -> None:
    cpesh = CPESH(concept="Helium", expected="It is a noble gas")
    retrieval_context.put_cpesh_to_cache("doc-2", cpesh)
    initial_entry = retrieval_context.cpesh_cache["doc-2"]
    initial_last = initial_entry["cpesh"]["last_accessed"]

    time.sleep(0.01)
    fetched = retrieval_context.get_cpesh_from_cache("doc-2")
    assert fetched is not None

    updated_entry = retrieval_context.cpesh_cache["doc-2"]
    updated_last = updated_entry["cpesh"]["last_accessed"]

    initial_dt = ts_utils.parse_iso_timestamp(initial_last)
    updated_dt = ts_utils.parse_iso_timestamp(updated_last)
    assert initial_dt is not None
    assert updated_dt is not None
    assert updated_dt >= initial_dt
    assert updated_last == fetched.last_accessed
    assert updated_entry["access_count"] == 2


def test_iso_timestamp_round_trip() -> None:
    iso_now = ts_utils.get_iso_timestamp()
    parsed = ts_utils.parse_iso_timestamp(iso_now)
    assert parsed is not None
    assert parsed.tzinfo is not None

    round_tripped = parsed.isoformat()
    reparsed = ts_utils.parse_iso_timestamp(round_tripped)
    assert reparsed == parsed


def test_ingest_retrieve_audit_trail(retrieval_context: "RetrievalContext") -> None:
    doc_ids = [f"doc-{idx}" for idx in range(5)]
    created_snapshots: Dict[str, str] = {}

    for doc_id in doc_ids:
        cpesh = CPESH(concept=f"Concept {doc_id}", expected="Answer")
        retrieval_context.put_cpesh_to_cache(doc_id, cpesh)
        created_snapshots[doc_id] = retrieval_context.cpesh_cache[doc_id]["cpesh"]["created_at"]

    for doc_id in doc_ids:
        time.sleep(0.001)
        cpesh = retrieval_context.get_cpesh_from_cache(doc_id)
        assert cpesh is not None
        assert cpesh.created_at is not None
        assert cpesh.last_accessed is not None
        created_dt = ts_utils.parse_iso_timestamp(cpesh.created_at)
        last_dt = ts_utils.parse_iso_timestamp(cpesh.last_accessed)
        assert created_dt is not None
        assert last_dt is not None
        assert last_dt >= created_dt

    retrieval_context._save_cpesh_cache()

    cache_file = Path(retrieval_context.cpesh_cache_path)
    assert cache_file.exists()

    records = [json.loads(line) for line in cache_file.read_text().splitlines() if line.strip()]
    assert len(records) == len(doc_ids)

    for record in records:
        doc_id = record["doc_id"]
        assert doc_id in doc_ids
        payload = record.get("cpesh", {})
        assert ts_utils.parse_iso_timestamp(payload.get("created_at")) is not None
        assert ts_utils.parse_iso_timestamp(payload.get("last_accessed")) is not None
        assert payload.get("created_at") == created_snapshots[doc_id]
        assert record.get("access_count", 0) >= 3
