"""Pytest configuration for path setup."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_PATH = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT_PATH / "src"

if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
