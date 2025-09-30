#!/usr/bin/env python3
"""
Download Gene Ontology (GO).

Usage:
    python scripts/data_downloading/download_go.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_downloading.download_base import OntologyDownloader, logger


class GODownloader(OntologyDownloader):
    """Download Gene Ontology."""

    def __init__(self):
        super().__init__("go")
        self.obo_url = "http://current.geneontology.org/ontology/go.obo"

    def download(self) -> bool:
        """Download GO OBO file."""
        logger.info("=" * 80)
        logger.info("GENE ONTOLOGY (GO) DOWNLOADER")
        logger.info("=" * 80)

        obo_path = self.base_dir / "go.obo"
        success = self.download_file(
            self.obo_url,
            obo_path,
            expected_size_mb=150
        )

        if not success:
            logger.error("❌ Download failed")
            return False

        # Compute checksum
        checksum = self.compute_checksum(obo_path)
        size_mb = obo_path.stat().st_size / (1024 * 1024)

        # Update dataset map
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=size_mb,
            checksum=checksum
        )

        logger.info("=" * 80)
        logger.info("✅ GO download complete!")
        logger.info(f"   File: {obo_path}")
        logger.info(f"   Size: {size_mb:.1f} MB")
        logger.info("=" * 80)
        return True


if __name__ == "__main__":
    downloader = GODownloader()
    success = downloader.download()
    sys.exit(0 if success else 1)
