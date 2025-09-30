#!/usr/bin/env python3
"""
Download DBpedia ontology and instance types.

DBpedia extracts structured data from Wikipedia.
We'll use the ontology hierarchy and dbo:type relations to build parent-child chains.

Files to download:
1. dbpedia_2016-10.owl - Ontology structure (classes + properties)
2. instance_types_en.ttl.bz2 - Instance-to-class mappings

Expected chains: ~30K
Expected quality: 92%
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


class DBpediaDownloader(OntologyDownloader):
    """Download DBpedia ontology and instance types."""

    # DBpedia 2016-10 ontology (stable version)
    ONTOLOGY_URL = "https://downloads.dbpedia.org/2016-10/dbpedia_2016-10.owl"

    # Instance types (English) - links instances to ontology classes
    # Using 2016-10 for consistency
    INSTANCE_TYPES_URL = "https://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_en.ttl.bz2"

    def __init__(self):
        super().__init__("dbpedia")
        self.ontology_file = self.base_dir / "dbpedia_2016-10.owl"
        self.instance_types_file = self.base_dir / "instance_types_en.ttl.bz2"

    def download(self) -> bool:
        """Download DBpedia ontology and instance types."""
        logger.info("=" * 60)
        logger.info("DBpedia 2016-10 Downloader")
        logger.info("=" * 60)

        # Download ontology file (~2MB)
        logger.info("\n[1/2] Downloading ontology structure...")
        success_ontology = self.download_file(
            url=self.ONTOLOGY_URL,
            output_path=self.ontology_file,
            expected_size_mb=2
        )

        if not success_ontology:
            return False

        # Download instance types (~1.2GB compressed, ~7GB uncompressed)
        logger.info("\n[2/2] Downloading instance types...")
        logger.info("⚠️  This is a large file (~1.2GB), may take 10-20 minutes")
        success_instances = self.download_file(
            url=self.INSTANCE_TYPES_URL,
            output_path=self.instance_types_file,
            expected_size_mb=1200
        )

        if not success_instances:
            return False

        # Compute checksums
        checksum_ontology = self.compute_checksum(self.ontology_file)
        checksum_instances = self.compute_checksum(self.instance_types_file)

        # Update dataset map
        total_size_mb = (
            self.ontology_file.stat().st_size +
            self.instance_types_file.stat().st_size
        ) / (1024 * 1024)

        # Store both checksums in dataset map
        combined_checksum = f"{checksum_ontology[:16]}+{checksum_instances[:16]}"

        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=round(total_size_mb, 2),
            checksum=combined_checksum
        )

        logger.info("=" * 60)
        logger.info("✅ DBpedia download complete!")
        logger.info("=" * 60)
        logger.info(f"  Ontology: {self.ontology_file}")
        logger.info(f"    Size: {self.ontology_file.stat().st_size / (1024*1024):.1f} MB")
        logger.info(f"  Instances: {self.instance_types_file}")
        logger.info(f"    Size: {self.instance_types_file.stat().st_size / (1024*1024):.1f} MB (compressed)")
        logger.info(f"  Total: {total_size_mb:.1f} MB")
        logger.info("  Format: OWL/XML + Turtle (bzip2)")
        logger.info("  Expected chains: ~30K")
        logger.info("=" * 60)

        return True


if __name__ == "__main__":
    downloader = DBpediaDownloader()
    success = downloader.download()
    exit(0 if success else 1)
