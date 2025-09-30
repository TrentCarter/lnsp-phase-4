#!/usr/bin/env python3
"""
Download ConceptNet 5.7 assertions dataset.

ConceptNet is a semantic network with common-sense relationships.
We'll extract parent-child chains from hierarchical relations like:
- /r/IsA (is a type of)
- /r/PartOf (is part of)
- /r/HasA (has a)

Expected chains: ~20K
Expected quality: 82%
"""

import logging
from pathlib import Path
from datetime import datetime
from download_base import OntologyDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConceptNetDownloader(OntologyDownloader):
    """Download ConceptNet assertions."""

    # ConceptNet 5.7 assertions (CSV format)
    # Contains all semantic relationships
    CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"

    def __init__(self):
        super().__init__("conceptnet")
        self.output_file = self.base_dir / "conceptnet-assertions-5.7.0.csv.gz"

    def download(self) -> bool:
        """Download ConceptNet assertions file."""
        logger.info("=" * 60)
        logger.info("ConceptNet 5.7 Downloader")
        logger.info("=" * 60)

        # Download main assertions file (~350MB compressed, ~1.5GB uncompressed)
        success = self.download_file(
            url=self.CONCEPTNET_URL,
            output_path=self.output_file,
            expected_size_mb=350  # Compressed size
        )

        if not success:
            return False

        # Compute checksum
        checksum = self.compute_checksum(self.output_file)

        # Update dataset map
        file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=round(file_size_mb, 2),
            checksum=checksum
        )

        logger.info("=" * 60)
        logger.info("✅ ConceptNet download complete!")
        logger.info("=" * 60)
        logger.info(f"  File: {self.output_file}")
        logger.info(f"  Size: {file_size_mb:.1f} MB (compressed)")
        logger.info(f"  Format: CSV (gzipped)")
        logger.info("  Relations to extract: /r/IsA, /r/PartOf, /r/HasA")
        logger.info("  Expected chains: ~20K")
        logger.info("=" * 60)

        return True


if __name__ == "__main__":
    downloader = ConceptNetDownloader()
    success = downloader.download()
    exit(0 if success else 1)
