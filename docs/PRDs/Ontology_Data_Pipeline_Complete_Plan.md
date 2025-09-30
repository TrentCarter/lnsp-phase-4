# Complete Ontology Data Pipeline Plan

**Date**: 2025-09-30
**Goal**: Download, ingest, validate, and train on 5 high-quality STEM ontology datasets
**Timeline**: 3-4 weeks total

---

## Architecture Overview

```
Step 0: Dataset Catalog
  └─> dataset_map.json (tracks all sources, splits, metadata)

Step 1: Download Scripts (5x datasets)
  ├─> scripts/data_downloading/download_dbpedia.py
  ├─> scripts/data_downloading/download_wikidata.py
  ├─> scripts/data_downloading/download_swo.py
  ├─> scripts/data_downloading/download_go.py
  └─> scripts/data_downloading/download_conceptnet.py

  Downloads to:
  └─> data/datasets/ontology_datasets/
      ├─> dbpedia/
      ├─> wikidata/
      ├─> swo/
      ├─> go/
      └─> conceptnet/

Step 2: Test Ingestion (1K from each)
  └─> src/ingest_ontology.py --source <name> --limit 1000

  Outputs:
  ├─> artifacts/ontology_samples/dbpedia_1k.jsonl
  ├─> artifacts/ontology_samples/wikidata_1k.jsonl
  ├─> artifacts/ontology_samples/swo_1k.jsonl
  ├─> artifacts/ontology_samples/go_1k.jsonl
  └─> artifacts/ontology_samples/conceptnet_1k.jsonl

Step 3: Early Validation
  └─> src/pipeline/p13_ontology_validation.py

  Outputs:
  └─> artifacts/validation_reports/
      ├─> dbpedia_1k_validation.json
      ├─> wikidata_1k_validation.json
      ├─> swo_1k_validation.json
      ├─> go_1k_validation.json
      └─> conceptnet_1k_validation.json

Step 4: Full Ingestion (125K total)
  └─> make ontology-ingest-all

Step 5: Sequential Training
  └─> src/lvm/train_sequential_curriculum.py
      (Trains on 5x 1K sets in order of quality)
```

---

## Step 0: Dataset Catalog (dataset_map.json)

### Purpose
- **Central manifest** of all datasets, splits, and metadata
- **Training order** tracking (curriculum learning)
- **Quality metrics** per source
- **Data lineage** (provenance tracking)

### Schema

```json
{
  "version": "1.0",
  "created_at": "2025-09-30T12:00:00Z",
  "updated_at": "2025-09-30T12:00:00Z",
  "total_chains": 125000,
  "total_concepts": 875000,
  "sources": {
    "dbpedia": {
      "name": "DBpedia Ontology (STEM subset)",
      "homepage": "https://www.dbpedia.org/",
      "license": "CC BY-SA 3.0",
      "format": "RDF/Turtle",
      "download_url": "https://databus.dbpedia.org/dbpedia/ontology/2023.09.01/ontology--DEV_type=parsed_sorted.nt",
      "local_path": "data/datasets/ontology_datasets/dbpedia/",
      "downloaded_at": "2025-09-30T13:00:00Z",
      "download_size_mb": 1250,
      "concepts_count": 4580000,
      "stem_concepts_count": 500000,
      "target_chains": 30000,
      "generated_chains": 0,
      "quality_metrics": {
        "expected_coherence": 0.85,
        "expected_pass_rate": 0.92,
        "domain_coverage": ["technology", "science", "mathematics", "engineering"]
      },
      "splits": {
        "test_1k": {
          "path": "artifacts/ontology_samples/dbpedia_1k.jsonl",
          "start_idx": 0,
          "end_idx": 1000,
          "chains": 1000,
          "concepts": 7000,
          "coherence": null,
          "pass_rate": null,
          "validation_report": "artifacts/validation_reports/dbpedia_1k_validation.json"
        },
        "train": {
          "path": "artifacts/ontology_chains/dbpedia_train.jsonl",
          "start_idx": 1000,
          "end_idx": 30000,
          "chains": 29000,
          "concepts": 203000,
          "coherence": null,
          "pass_rate": null
        }
      },
      "curriculum_priority": 2,
      "training_stage": "stage_2_mixed_quality"
    },
    "wikidata": {
      "name": "Wikidata (Science & Technology)",
      "homepage": "https://www.wikidata.org/",
      "license": "CC0 (public domain)",
      "format": "JSON",
      "download_url": "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz",
      "local_path": "data/datasets/ontology_datasets/wikidata/",
      "downloaded_at": null,
      "download_size_mb": 120000,
      "concepts_count": 100000000,
      "stem_concepts_count": 10000000,
      "target_chains": 20000,
      "generated_chains": 0,
      "quality_metrics": {
        "expected_coherence": 0.82,
        "expected_pass_rate": 0.88,
        "domain_coverage": ["technology", "science", "mathematics", "engineering", "computer science"]
      },
      "splits": {
        "test_1k": {
          "path": "artifacts/ontology_samples/wikidata_1k.jsonl",
          "start_idx": 0,
          "end_idx": 1000,
          "chains": 1000,
          "concepts": 6500,
          "coherence": null,
          "pass_rate": null,
          "validation_report": "artifacts/validation_reports/wikidata_1k_validation.json"
        },
        "train": {
          "path": "artifacts/ontology_chains/wikidata_train.jsonl",
          "start_idx": 1000,
          "end_idx": 20000,
          "chains": 19000,
          "concepts": 123500,
          "coherence": null,
          "pass_rate": null
        }
      },
      "curriculum_priority": 3,
      "training_stage": "stage_2_mixed_quality"
    },
    "swo": {
      "name": "Software Ontology",
      "homepage": "http://www.ebi.ac.uk/swo/",
      "license": "CC BY 4.0",
      "format": "OWL",
      "download_url": "https://raw.githubusercontent.com/allysonlister/swo/master/release/swo.owl",
      "local_path": "data/datasets/ontology_datasets/swo/",
      "downloaded_at": null,
      "download_size_mb": 50,
      "concepts_count": 5000,
      "stem_concepts_count": 5000,
      "target_chains": 15000,
      "generated_chains": 0,
      "quality_metrics": {
        "expected_coherence": 0.90,
        "expected_pass_rate": 0.95,
        "domain_coverage": ["technology", "engineering", "computer science"]
      },
      "splits": {
        "test_1k": {
          "path": "artifacts/ontology_samples/swo_1k.jsonl",
          "start_idx": 0,
          "end_idx": 1000,
          "chains": 1000,
          "concepts": 8000,
          "coherence": null,
          "pass_rate": null,
          "validation_report": "artifacts/validation_reports/swo_1k_validation.json"
        },
        "train": {
          "path": "artifacts/ontology_chains/swo_train.jsonl",
          "start_idx": 1000,
          "end_idx": 15000,
          "chains": 14000,
          "concepts": 112000,
          "coherence": null,
          "pass_rate": null
        }
      },
      "curriculum_priority": 1,
      "training_stage": "stage_1_clean_data"
    },
    "go": {
      "name": "Gene Ontology + Protein Ontology",
      "homepage": "http://geneontology.org/",
      "license": "CC BY 4.0",
      "format": "OBO",
      "download_url": "http://current.geneontology.org/ontology/go.obo",
      "local_path": "data/datasets/ontology_datasets/go/",
      "downloaded_at": null,
      "download_size_mb": 150,
      "concepts_count": 44000,
      "stem_concepts_count": 44000,
      "target_chains": 40000,
      "generated_chains": 0,
      "quality_metrics": {
        "expected_coherence": 0.90,
        "expected_pass_rate": 0.94,
        "domain_coverage": ["science", "medicine", "biology"]
      },
      "splits": {
        "test_1k": {
          "path": "artifacts/ontology_samples/go_1k.jsonl",
          "start_idx": 0,
          "end_idx": 1000,
          "chains": 1000,
          "concepts": 12000,
          "coherence": null,
          "pass_rate": null,
          "validation_report": "artifacts/validation_reports/go_1k_validation.json"
        },
        "train": {
          "path": "artifacts/ontology_chains/go_train.jsonl",
          "start_idx": 1000,
          "end_idx": 40000,
          "chains": 39000,
          "concepts": 468000,
          "coherence": null,
          "pass_rate": null
        }
      },
      "curriculum_priority": 1,
      "training_stage": "stage_1_clean_data"
    },
    "conceptnet": {
      "name": "ConceptNet (Technical subset)",
      "homepage": "https://conceptnet.io/",
      "license": "CC BY-SA 4.0",
      "format": "CSV",
      "download_url": "https://github.com/commonsense/conceptnet5/raw/master/data/conceptnet-assertions-5.7.0.csv.gz",
      "local_path": "data/datasets/ontology_datasets/conceptnet/",
      "downloaded_at": null,
      "download_size_mb": 350,
      "concepts_count": 8000000,
      "stem_concepts_count": 2000000,
      "target_chains": 20000,
      "generated_chains": 0,
      "quality_metrics": {
        "expected_coherence": 0.77,
        "expected_pass_rate": 0.82,
        "domain_coverage": ["technology", "science", "common_sense"]
      },
      "splits": {
        "test_1k": {
          "path": "artifacts/ontology_samples/conceptnet_1k.jsonl",
          "start_idx": 0,
          "end_idx": 1000,
          "chains": 1000,
          "concepts": 4500,
          "coherence": null,
          "pass_rate": null,
          "validation_report": "artifacts/validation_reports/conceptnet_1k_validation.json"
        },
        "train": {
          "path": "artifacts/ontology_chains/conceptnet_train.jsonl",
          "start_idx": 1000,
          "end_idx": 20000,
          "chains": 19000,
          "concepts": 85500,
          "coherence": null,
          "pass_rate": null
        }
      },
      "curriculum_priority": 4,
      "training_stage": "stage_3_full_dataset"
    }
  },
  "curriculum_stages": {
    "stage_1_clean_data": {
      "description": "High-quality expert-curated ontologies",
      "sources": ["swo", "go"],
      "total_chains": 55000,
      "expected_coherence": 0.90,
      "training_epochs": 3,
      "learning_rate": 0.0001,
      "notes": "Learn basic sequential patterns with clean data"
    },
    "stage_2_mixed_quality": {
      "description": "Large-scale community-curated ontologies",
      "sources": ["dbpedia", "wikidata"],
      "total_chains": 50000,
      "expected_coherence": 0.84,
      "training_epochs": 4,
      "learning_rate": 0.00005,
      "notes": "Generalization to noisy data"
    },
    "stage_3_full_dataset": {
      "description": "Common-sense reasoning and edge cases",
      "sources": ["conceptnet"],
      "total_chains": 20000,
      "expected_coherence": 0.77,
      "training_epochs": 3,
      "learning_rate": 0.00002,
      "notes": "Robustness training with crowdsourced data"
    }
  },
  "validation_thresholds": {
    "min_pass_rate": 0.80,
    "min_coherence": 0.75,
    "min_chain_length": 5,
    "max_chain_length": 20
  },
  "provenance": {
    "pipeline_version": "1.0",
    "lnsp_version": "0.5.0",
    "downloaded_by": "automated_pipeline",
    "notes": "Replaces FactoidWiki 5K baseline due to low quality (48.9% pass rate)"
  }
}
```

---

## Step 1: Download Scripts

### Base Download Script Template

```python
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

logging.basicConfig(level=logging.INFO)
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

        logger.info(f"Downloading {url} to {output_path}")

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
                        if downloaded_size % (10 * 1024 * 1024) == 0:
                            mb_downloaded = downloaded_size / (1024 * 1024)
                            logger.info(f"Downloaded {mb_downloaded:.1f} MB")

            actual_size_mb = downloaded_size / (1024 * 1024)
            logger.info(f"Download complete: {actual_size_mb:.1f} MB")

            # Validate size
            if expected_size_mb:
                if abs(actual_size_mb - expected_size_mb) > (expected_size_mb * 0.1):
                    logger.warning(
                        f"Size mismatch: expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB"
                    )

            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

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

            logger.info(f"Updated dataset_map.json for {self.dataset_name}")
        else:
            logger.warning(f"{self.dataset_name} not found in dataset_map.json")

    def download(self) -> bool:
        """
        Download dataset. Override in subclasses.

        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement download()")
```

### 1. DBpedia Downloader

```python
#!/usr/bin/env python3
"""
Download DBpedia ontology dataset.

Usage:
    python scripts/data_downloading/download_dbpedia.py
"""

from pathlib import Path
import gzip
import logging
from download_base import OntologyDownloader

logger = logging.getLogger(__name__)


class DBpediaDownloader(OntologyDownloader):
    """Download DBpedia STEM ontology subset."""

    def __init__(self):
        super().__init__("dbpedia")
        self.ontology_url = "https://databus.dbpedia.org/dbpedia/ontology/2023.09.01/ontology--DEV_type=parsed_sorted.nt"
        self.mappings_url = "https://databus.dbpedia.org/dbpedia/mappings/instance-types/2023.09.01/instance-types_lang=en_specific.ttl.bz2"

    def download(self) -> bool:
        """Download DBpedia ontology and mappings."""
        logger.info("=" * 80)
        logger.info("DBPEDIA DOWNLOADER")
        logger.info("=" * 80)

        # Download ontology file
        ontology_path = self.base_dir / "ontology.nt"
        success = self.download_file(
            self.ontology_url,
            ontology_path,
            expected_size_mb=50
        )

        if not success:
            return False

        # Download instance types (mappings)
        mappings_path = self.base_dir / "instance_types.ttl.bz2"
        success = self.download_file(
            self.mappings_url,
            mappings_path,
            expected_size_mb=1200
        )

        if not success:
            return False

        # Compute checksums
        ontology_checksum = self.compute_checksum(ontology_path)
        logger.info(f"Ontology checksum: {ontology_checksum}")

        # Update dataset map
        total_size_mb = (ontology_path.stat().st_size + mappings_path.stat().st_size) / (1024 * 1024)
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=total_size_mb,
            checksum=ontology_checksum
        )

        logger.info("✅ DBpedia download complete")
        return True


if __name__ == "__main__":
    downloader = DBpediaDownloader()
    success = downloader.download()
    exit(0 if success else 1)
```

### 2. Wikidata Downloader

```python
#!/usr/bin/env python3
"""
Download Wikidata STEM subset.

Note: Full Wikidata dump is 120GB+. This script downloads
a filtered STEM subset using SPARQL queries.

Usage:
    python scripts/data_downloading/download_wikidata.py
"""

import json
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
from download_base import OntologyDownloader

logger = logging.getLogger(__name__)


class WikidataDownloader(OntologyDownloader):
    """Download Wikidata STEM concepts via SPARQL."""

    def __init__(self):
        super().__init__("wikidata")
        self.sparql_endpoint = "https://query.wikidata.org/sparql"

    def fetch_stem_concepts(self, root_qid: str, limit: int = 10000) -> list:
        """
        Fetch STEM concepts from Wikidata using SPARQL.

        Args:
            root_qid: Root Wikidata ID (e.g., Q8029 for "algorithm")
            limit: Max results

        Returns:
            List of concept dicts
        """
        sparql = SPARQLWrapper(self.sparql_endpoint)
        sparql.setReturnFormat(JSON)

        query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?parent ?parentLabel
        WHERE {{
          ?item wdt:P31/wdt:P279* wd:{root_qid} .
          ?item rdfs:label ?itemLabel .
          ?item schema:description ?itemDescription .
          OPTIONAL {{
            ?item wdt:P279 ?parent .
            ?parent rdfs:label ?parentLabel .
          }}
          FILTER(LANG(?itemLabel) = "en")
          FILTER(LANG(?itemDescription) = "en")
          FILTER(LANG(?parentLabel) = "en")
        }}
        LIMIT {limit}
        """

        sparql.setQuery(query)
        results = sparql.query().convert()

        concepts = []
        for result in results["results"]["bindings"]:
            concepts.append({
                "qid": result["item"]["value"].split("/")[-1],
                "label": result["itemLabel"]["value"],
                "description": result.get("itemDescription", {}).get("value", ""),
                "parent_qid": result.get("parent", {}).get("value", "").split("/")[-1],
                "parent_label": result.get("parentLabel", {}).get("value", "")
            })

        return concepts

    def download(self) -> bool:
        """Download Wikidata STEM concepts."""
        logger.info("=" * 80)
        logger.info("WIKIDATA DOWNLOADER")
        logger.info("=" * 80)

        stem_roots = [
            ("Q8029", "algorithm"),
            ("Q7397", "software"),
            ("Q12483", "statistics"),
            ("Q395", "mathematics"),
            ("Q2329", "chemistry"),
            ("Q413", "physics"),
            ("Q420", "biology")
        ]

        all_concepts = []
        for qid, label in stem_roots:
            logger.info(f"Fetching concepts for: {label} ({qid})")
            try:
                concepts = self.fetch_stem_concepts(qid, limit=3000)
                logger.info(f"  Found {len(concepts)} concepts")
                all_concepts.extend(concepts)
            except Exception as e:
                logger.error(f"  Error: {e}")
                continue

        # Save to JSONL
        output_path = self.base_dir / "wikidata_stem_concepts.jsonl"
        with open(output_path, 'w') as f:
            for concept in all_concepts:
                f.write(json.dumps(concept) + '\n')

        logger.info(f"Saved {len(all_concepts)} concepts to {output_path}")

        # Update dataset map
        size_mb = output_path.stat().st_size / (1024 * 1024)
        checksum = self.compute_checksum(output_path)
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=size_mb,
            checksum=checksum
        )

        logger.info("✅ Wikidata download complete")
        return True


if __name__ == "__main__":
    downloader = WikidataDownloader()
    success = downloader.download()
    exit(0 if success else 1)
```

### 3. Software Ontology (SWO) Downloader

```python
#!/usr/bin/env python3
"""
Download Software Ontology (SWO).

Usage:
    python scripts/data_downloading/download_swo.py
"""

from download_base import OntologyDownloader


class SWODownloader(OntologyDownloader):
    """Download Software Ontology."""

    def __init__(self):
        super().__init__("swo")
        self.owl_url = "https://raw.githubusercontent.com/allysonlister/swo/master/release/swo.owl"

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

        logger.info("✅ SWO download complete")
        return True


if __name__ == "__main__":
    downloader = SWODownloader()
    success = downloader.download()
    exit(0 if success else 1)
```

### 4. Gene Ontology (GO) Downloader

```python
#!/usr/bin/env python3
"""
Download Gene Ontology (GO).

Usage:
    python scripts/data_downloading/download_go.py
"""

from download_base import OntologyDownloader


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

        logger.info("✅ GO download complete")
        return True


if __name__ == "__main__":
    downloader = GODownloader()
    success = downloader.download()
    exit(0 if success else 1)
```

### 5. ConceptNet Downloader

```python
#!/usr/bin/env python3
"""
Download ConceptNet assertions.

Usage:
    python scripts/data_downloading/download_conceptnet.py
"""

import gzip
from download_base import OntologyDownloader


class ConceptNetDownloader(OntologyDownloader):
    """Download ConceptNet assertions."""

    def __init__(self):
        super().__init__("conceptnet")
        self.csv_url = "https://github.com/commonsense/conceptnet5/raw/master/data/conceptnet-assertions-5.7.0.csv.gz"

    def download(self) -> bool:
        """Download ConceptNet CSV file."""
        logger.info("=" * 80)
        logger.info("CONCEPTNET DOWNLOADER")
        logger.info("=" * 80)

        csv_path = self.base_dir / "conceptnet-assertions-5.7.0.csv.gz"
        success = self.download_file(
            self.csv_url,
            csv_path,
            expected_size_mb=350
        )

        if not success:
            return False

        # Decompress for faster processing later
        decompressed_path = self.base_dir / "conceptnet-assertions-5.7.0.csv"
        logger.info(f"Decompressing to {decompressed_path}...")

        with gzip.open(csv_path, 'rb') as f_in:
            with open(decompressed_path, 'wb') as f_out:
                f_out.write(f_in.read())

        logger.info("Decompression complete")

        # Compute checksum
        checksum = self.compute_checksum(decompressed_path)
        size_mb = decompressed_path.stat().st_size / (1024 * 1024)

        # Update dataset map
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=size_mb,
            checksum=checksum
        )

        logger.info("✅ ConceptNet download complete")
        return True


if __name__ == "__main__":
    downloader = ConceptNetDownloader()
    success = downloader.download()
    exit(0 if success else 1)
```

---

## Step 2: Test Ingestion (1K from each)

### Test Ingestion Script

```python
#!/usr/bin/env python3
"""
Test ingestion of 1K chains from each ontology source.

Usage:
    python src/test_ingest_1k.py --source dbpedia
    python src/test_ingest_1k.py --all
"""

import argparse
import json
import logging
from pathlib import Path
from src.ingest_ontology import OntologyIngester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ingest_source(source_name: str, limit: int = 1000) -> dict:
    """
    Test ingest 1K chains from a single source.

    Args:
        source_name: One of [dbpedia, wikidata, swo, go, conceptnet]
        limit: Number of chains to ingest

    Returns:
        Statistics dict
    """
    logger.info("=" * 80)
    logger.info(f"TEST INGESTION: {source_name.upper()} (1K chains)")
    logger.info("=" * 80)

    ingester = OntologyIngester(source=source_name)

    # Generate chains
    chains = ingester.generate_chains(limit=limit)

    if not chains:
        logger.error(f"Failed to generate chains from {source_name}")
        return {"success": False}

    # Save to JSONL
    output_dir = Path("artifacts/ontology_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{source_name}_1k.jsonl"

    with open(output_path, 'w') as f:
        for chain in chains:
            f.write(json.dumps(chain) + '\n')

    logger.info(f"Saved {len(chains)} chains to {output_path}")

    # Compute statistics
    total_concepts = sum(len(chain["concept_chain"]) for chain in chains)
    avg_length = total_concepts / len(chains) if chains else 0
    avg_coherence = sum(
        sum(chain["coherence_scores"]) / len(chain["coherence_scores"])
        for chain in chains if chain.get("coherence_scores")
    ) / len(chains) if chains else 0

    stats = {
        "success": True,
        "source": source_name,
        "chains": len(chains),
        "total_concepts": total_concepts,
        "avg_chain_length": avg_length,
        "avg_coherence": avg_coherence,
        "output_path": str(output_path)
    }

    logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
    return stats


def test_ingest_all() -> dict:
    """Test ingest 1K from all 5 sources."""
    sources = ["dbpedia", "wikidata", "swo", "go", "conceptnet"]

    all_stats = {}
    for source in sources:
        stats = test_ingest_source(source, limit=1000)
        all_stats[source] = stats

    # Save combined report
    report_path = Path("artifacts/ontology_samples/test_ingest_report.json")
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"Test ingestion complete. Report: {report_path}")
    return all_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["dbpedia", "wikidata", "swo", "go", "conceptnet"])
    parser.add_argument("--all", action="store_true", help="Test all sources")
    parser.add_argument("--limit", type=int, default=1000, help="Number of chains per source")

    args = parser.parse_args()

    if args.all:
        test_ingest_all()
    elif args.source:
        test_ingest_source(args.source, limit=args.limit)
    else:
        print("Error: Must specify --source or --all")
        exit(1)
```

---

## Step 3: Early Data Validation

### Ontology-Specific P13 Validation

```python
#!/usr/bin/env python3
"""
P13 Echo Validation for ontology samples.

Validates 1K samples from each source before full ingestion.

Usage:
    python src/pipeline/p13_ontology_validation.py --source dbpedia
    python src/pipeline/p13_ontology_validation.py --all
"""

import argparse
import json
from pathlib import Path
from src.pipeline.p13_echo_validation import P13EchoValidator

def validate_ontology_sample(source_name: str) -> dict:
    """
    Validate 1K sample from ontology source.

    Args:
        source_name: Source to validate

    Returns:
        Validation report dict
    """
    sample_path = Path(f"artifacts/ontology_samples/{source_name}_1k.jsonl")

    if not sample_path.exists():
        print(f"Error: {sample_path} not found. Run test ingestion first.")
        return {"error": "sample not found"}

    validator = P13EchoValidator(threshold=0.82)

    # Load sample entries
    entries = []
    with open(sample_path, 'r') as f:
        for line in f:
            chain = json.loads(line)
            # Extract CPESH entries from chain
            for i, concept in enumerate(chain["concept_chain"]):
                if i < len(chain["concept_chain"]) - 1:
                    entries.append({
                        "cpe_id": f"{source_name}-{chain['seq_id']}-{i}",
                        "probe_question": chain.get("probe", f"What follows {concept}?"),
                        "concept_text": concept,
                        "domain_code": 2,  # Technology default
                        "task_code": 4,    # Taxonomy
                        "modifier_code": 46,  # Hierarchical
                        "tmd_lane": f"{source_name}-taxonomy-hierarchical"
                    })

    # Validate
    results = validator.validate_batch(entries)

    # Generate report
    report = validator.generate_report(results)

    # Save report
    report_dir = Path("artifacts/validation_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{source_name}_1k_validation.json"

    with open(report_path, 'w') as f:
        json.dump(report.__dict__, f, indent=2)

    print(f"✅ Validation complete: {report_path}")
    print(f"   Pass rate: {report.pass_rate:.2%}")
    print(f"   Mean coherence: {report.mean_echo_score:.4f}")

    return report.__dict__


def validate_all_samples():
    """Validate all 5 ontology samples."""
    sources = ["dbpedia", "wikidata", "swo", "go", "conceptnet"]

    all_reports = {}
    for source in sources:
        print(f"\n{'='*80}")
        print(f"Validating {source.upper()}")
        print(f"{'='*80}")
        report = validate_ontology_sample(source)
        all_reports[source] = report

    # Summary table
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"{'Source':<15} {'Pass Rate':<12} {'Mean Score':<12} {'Status':<10}")
    print("-"*80)

    for source, report in all_reports.items():
        if "error" in report:
            print(f"{source:<15} {'ERROR':<12} {'-':<12} {'❌':<10}")
        else:
            pass_rate = report["pass_rate"]
            mean_score = report["mean_echo_score"]
            status = "✅" if pass_rate >= 0.80 else "⚠️" if pass_rate >= 0.70 else "❌"
            print(f"{source:<15} {pass_rate:<12.2%} {mean_score:<12.4f} {status:<10}")

    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["dbpedia", "wikidata", "swo", "go", "conceptnet"])
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.all:
        validate_all_samples()
    elif args.source:
        validate_ontology_sample(args.source)
    else:
        print("Error: Must specify --source or --all")
        exit(1)
```

---

## Makefile Commands

```makefile
# Ontology Data Pipeline Makefile

# Step 0: Initialize dataset map
.PHONY: ontology-init
ontology-init:
	@echo "Initializing dataset_map.json..."
	cp docs/PRDs/dataset_map_template.json data/dataset_map.json
	@echo "✅ Dataset map initialized"

# Step 1: Download all datasets
.PHONY: ontology-download-all
ontology-download-all:
	@echo "Downloading all ontology datasets..."
	python scripts/data_downloading/download_dbpedia.py
	python scripts/data_downloading/download_wikidata.py
	python scripts/data_downloading/download_swo.py
	python scripts/data_downloading/download_go.py
	python scripts/data_downloading/download_conceptnet.py
	@echo "✅ All downloads complete"

# Step 2: Test ingest 1K from each source
.PHONY: ontology-test-ingest
ontology-test-ingest:
	@echo "Test ingesting 1K chains from each source..."
	python src/test_ingest_1k.py --all
	@echo "✅ Test ingestion complete"

# Step 3: Validate 1K samples
.PHONY: ontology-validate-samples
ontology-validate-samples:
	@echo "Validating 1K samples from each source..."
	python src/pipeline/p13_ontology_validation.py --all
	@echo "✅ Validation complete"

# Step 4: Full ingestion (125K chains)
.PHONY: ontology-ingest-full
ontology-ingest-full:
	@echo "Full ingestion of 125K ontology chains..."
	python src/ingest_ontology_full.py
	@echo "✅ Full ingestion complete"

# Combined pipeline
.PHONY: ontology-pipeline
ontology-pipeline: ontology-init ontology-download-all ontology-test-ingest ontology-validate-samples
	@echo "✅ Ontology pipeline complete. Ready for full ingestion."

# Clean up
.PHONY: ontology-clean
ontology-clean:
	rm -rf data/datasets/ontology_datasets/*/
	rm -rf artifacts/ontology_samples/
	rm -rf artifacts/validation_reports/
	@echo "✅ Cleaned ontology artifacts"
```

---

## Timeline & Execution Plan

### Week 1: Download & Test (Days 1-7)

**Day 1** (4 hours):
```bash
# Create dataset_map.json
cp docs/PRDs/dataset_map_template.json data/dataset_map.json

# Create download scripts
# (already provided above)
```

**Day 2-3** (8 hours):
```bash
# Download all datasets
make ontology-download-all

# Expected results:
# - DBpedia: 1.2GB
# - Wikidata: 20GB (filtered)
# - SWO: 50MB
# - GO: 150MB
# - ConceptNet: 350MB
```

**Day 4-5** (12 hours):
```bash
# Test ingest 1K from each
make ontology-test-ingest

# Expected outputs:
# - artifacts/ontology_samples/dbpedia_1k.jsonl (1000 chains)
# - artifacts/ontology_samples/wikidata_1k.jsonl (1000 chains)
# - artifacts/ontology_samples/swo_1k.jsonl (1000 chains)
# - artifacts/ontology_samples/go_1k.jsonl (1000 chains)
# - artifacts/ontology_samples/conceptnet_1k.jsonl (1000 chains)
```

**Day 6-7** (8 hours):
```bash
# Validate samples
make ontology-validate-samples

# Expected pass rates:
# - SWO: 94%+ ✅
# - GO: 92%+ ✅
# - DBpedia: 90%+ ✅
# - Wikidata: 86%+ ✅
# - ConceptNet: 80%+ ✅
```

### Week 2-3: Full Ingestion (Days 8-21)

```bash
# Full ingestion (125K chains)
make ontology-ingest-full

# Takes ~10-14 days depending on:
# - LLM throughput (500ms/concept)
# - Embedding speed (50ms/concept)
# - Database write speed (10ms/concept)
```

### Week 4: Training (Days 22-28)

```bash
# Sequential curriculum training
python src/lvm/train_sequential_curriculum.py \
  --stage1-sources swo,go \
  --stage2-sources dbpedia,wikidata \
  --stage3-sources conceptnet \
  --epochs-per-stage 3,4,3 \
  --model-out artifacts/lvm/latent_mamba_ontology_v1.pt
```

---

## Success Criteria

### Download Phase
- ✅ All 5 datasets downloaded without errors
- ✅ File sizes match expected ranges (±10%)
- ✅ Checksums computed and stored
- ✅ dataset_map.json updated with metadata

### Test Ingestion Phase
- ✅ 1K chains generated from each source
- ✅ Avg chain length 5-15 concepts
- ✅ All chains have sequential structure
- ✅ No duplicate concepts within chains

### Validation Phase
- ✅ SWO/GO: ≥90% pass rate
- ✅ DBpedia/Wikidata: ≥85% pass rate
- ✅ ConceptNet: ≥80% pass rate
- ✅ Mean coherence: ≥0.80 across all sources

### Full Ingestion Phase
- ✅ 125K total chains generated
- ✅ No data corruption (checksums validated)
- ✅ dataset_map.json tracks all splits
- ✅ Ready for P15 training

---

**Document Status**: Complete Implementation Plan
**Next Action**: Create dataset_map.json template and download_base.py
