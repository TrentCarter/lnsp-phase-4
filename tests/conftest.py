"""Pytest configuration for path setup and FAISS environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pytest

# Set thread limits before any imports to prevent FAISS crashes on macOS ARM64
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("FAISS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_PATH / "src"

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Test if FAISS is importable
try:
    import faiss  # noqa
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "faiss: marks tests that require FAISS")
    config.addinivalue_line("markers", "heavy: marks slow/heavy tests")


def pytest_collection_modifyitems(config, items):
    """Skip FAISS tests if FAISS is not available."""
    if HAS_FAISS:
        return

    skip_faiss = pytest.mark.skip(reason="FAISS not importable on this runtime")
    for item in items:
        if "faiss" in item.keywords:
            item.add_marker(skip_faiss)


# Make FAISS availability accessible to tests
pytest.HAS_FAISS = HAS_FAISS
