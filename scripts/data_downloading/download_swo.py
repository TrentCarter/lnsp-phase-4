#!/usr/bin/env python3
"""
Download Software Ontology (SWO).

Usage:
    python scripts/data_downloading/download_swo.py
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.data_downloading.download_base import OntologyDownloader, logger


class SWODownloader(OntologyDownloader):
    """Download Software Ontology."""

    def __init__(self):
        super().__init__("swo")
        self.owl_url = "https://raw.githubusercontent.com/allysonlister/swo/master/swo.owl"

    def download(self) -> bool:
        """Download SWO OWL file."""
        logger.info("=" * 80)
        logger.info("SOFTWARE ONTOLOGY (SWO) DOWNLOADER")
        logger.info("=" * 80)

        owl_path = self.base_dir / "swo.owl"
        success = self.download_file(
            self.owl_url,
            owl_path,
            expected_size_mb=50
        )

        if not success:
            logger.error("❌ Download failed")
            return False

        # Compute checksum
        checksum = self.compute_checksum(owl_path)
        size_mb = owl_path.stat().st_size / (1024 * 1024)

        # Update dataset map
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=size_mb,
            checksum=checksum
        )

        logger.info("=" * 80)
        logger.info("✅ SWO download complete!")
        logger.info(f"   File: {owl_path}")
        logger.info(f"   Size: {size_mb:.1f} MB")
        logger.info("=" * 80)
        return True


if __name__ == "__main__":
    downloader = SWODownloader()
    success = downloader.download()
    sys.exit(0 if success else 1)
