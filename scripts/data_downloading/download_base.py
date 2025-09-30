#!/usr/bin/env python3
"""
Base downloader class for ontology datasets.
All specific downloaders inherit from this.
"""

import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OntologyDownloader:
    """Base class for downloading ontology datasets."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.base_dir = Path("data/datasets/ontology_datasets") / dataset_name
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_map_path = Path("data/dataset_map.json")

    def download_file(
        self,
        url: str,
        output_path: Path,
        chunk_size: int = 8192,
        expected_size_mb: Optional[int] = None
    ) -> bool:
        """
        Download file with progress tracking and validation.

        Args:
            url: Download URL
            output_path: Local save path
            chunk_size: Download chunk size (bytes)
            expected_size_mb: Expected file size for validation

        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return True

        logger.info(f"Downloading {url}")
        logger.info(f"Saving to: {output_path}")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        # Progress update every 10MB
                        if downloaded_size % (10 * 1024 * 1024) < chunk_size:
                            mb_downloaded = downloaded_size / (1024 * 1024)
                            if total_size > 0:
                                percent = (downloaded_size / total_size) * 100
                                logger.info(f"  Downloaded {mb_downloaded:.1f} MB ({percent:.1f}%)")
                            else:
                                logger.info(f"  Downloaded {mb_downloaded:.1f} MB")

            actual_size_mb = downloaded_size / (1024 * 1024)
            logger.info(f"✅ Download complete: {actual_size_mb:.1f} MB")

            # Validate size
            if expected_size_mb:
                if abs(actual_size_mb - expected_size_mb) > (expected_size_mb * 0.1):
                    logger.warning(
                        f"⚠️  Size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB"
                    )

            return True

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        logger.info(f"Computing checksum for {file_path.name}...")
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        checksum = sha256.hexdigest()
        logger.info(f"  Checksum: {checksum[:16]}...")
        return checksum

    def update_dataset_map(
        self,
        downloaded_at: str,
        download_size_mb: float,
        checksum: str
    ):
        """Update dataset_map.json with download metadata."""
        if not self.dataset_map_path.exists():
            logger.error("dataset_map.json not found!")
            return

        with open(self.dataset_map_path, 'r') as f:
            dataset_map = json.load(f)

        if self.dataset_name in dataset_map["sources"]:
            dataset_map["sources"][self.dataset_name]["downloaded_at"] = downloaded_at
            dataset_map["sources"][self.dataset_name]["download_size_mb"] = download_size_mb
            dataset_map["sources"][self.dataset_name]["checksum"] = checksum
            dataset_map["updated_at"] = datetime.now().isoformat()

            with open(self.dataset_map_path, 'w') as f:
                json.dump(dataset_map, f, indent=2)

            logger.info(f"✅ Updated dataset_map.json for {self.dataset_name}")
        else:
            logger.warning(f"⚠️  {self.dataset_name} not found in dataset_map.json")

    def download(self) -> bool:
        """
        Download dataset. Override in subclasses.

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement download()")


if __name__ == "__main__":
    print("This is a base class. Use specific downloaders (e.g., download_swo.py)")
