"""
CPESH Data Store

This module provides a permanent training datastore facade for CPESH entries.
It supports an active tier for immediate use with future tiers (Warm, Cold)
for scalability.
"""

import json
import os
from typing import Iterator, Dict, Any
from pathlib import Path


class CPESHDataStore:
    """
    Permanent training datastore facade (Active tier for S5).

    Future tiers (Warm .jsonl.gz, Cold Parquet, SQLite index) can slot behind this.
    This provides a clean interface for storing and retrieving CPESH entries.
    """

    def __init__(self, active_path: str = "artifacts/cpesh_active.jsonl"):
        """
        Initialize the CPESH datastore.

        Args:
            active_path: Path to the active tier storage file
        """
        self.active_path = active_path
        self.active_dir = os.path.dirname(active_path)

        # Ensure the directory exists
        os.makedirs(self.active_dir, exist_ok=True)

    def append(self, entry: Dict[str, Any]) -> None:
        """
        Append a CPESH entry to the active tier.

        Args:
            entry: CPESH entry dictionary containing quality, cosine, expected_vec, etc.
        """
        with open(self.active_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def iter_active(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all entries in the active tier.

        Returns:
            Iterator of CPESH entry dictionaries
        """
        if not os.path.exists(self.active_path):
            return iter([])

        with open(self.active_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue

    def get_by_doc_id(self, doc_id: str) -> Dict[str, Any] | None:
        """
        Get a CPESH entry by document ID.

        Args:
            doc_id: Document identifier

        Returns:
            CPESH entry if found, None otherwise
        """
        for entry in self.iter_active():
            if entry.get("doc_id") == doc_id:
                return entry
        return None

    def count(self) -> int:
        """
        Count the number of entries in the active tier.

        Returns:
            Number of entries
        """
        count = 0
        for _ in self.iter_active():
            count += 1
        return count

    def clear(self) -> None:
        """
        Clear all entries from the active tier.
        """
        if os.path.exists(self.active_path):
            os.remove(self.active_path)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the datastore.

        Returns:
            Dictionary with statistics
        """
        entries = list(self.iter_active())
        return {
            "total_entries": len(entries),
            "file_path": self.active_path,
            "file_size": os.path.getsize(self.active_path) if os.path.exists(self.active_path) else 0
        }
