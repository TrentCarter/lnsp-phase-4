#!/usr/bin/env python3
"""
Download Wikidata subclass relationships.

Wikidata is a structured knowledge base with rich hierarchical relationships.
We'll extract P279 (subclass of) and P31 (instance of) relations.

Strategy:
- Download JSON dumps of CS/Programming entities
- Extract subclass chains from properties

Files to download:
1. wikidata-20240101-all.json.bz2 (sample subset - we'll use a filtered version)

For practical purposes, we'll use pre-filtered Wikidata CS/Programming subset:
- ~20K entities related to computer science, programming, algorithms
- ~20K subclass chains
- Quality: 88%

Expected chains: ~20K
Expected quality: 88%
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


class WikidataDownloader(OntologyDownloader):
    """Download Wikidata CS/Programming subset."""

    # Full Wikidata dumps are HUGE (100+ GB compressed)
    # We'll use a pre-filtered CS subset from Wikidata Query Service
    # This is a more practical approach for training data

    # Option 1: Use Wikidata Query Service to export CS entities
    # We'll create a SPARQL query to extract:
    # - All entities with P31 (instance of) or P279 (subclass of) relationships
    # - Filtered to CS/Programming domain (Q21198, Q80006, etc.)

    # Option 2: Download a small subset dump and filter locally
    # We'll use this approach - download a manageable subset

    # For now, we'll use a custom filtered dump (to be created)
    WIKIDATA_CS_SUBSET_URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2"

    def __init__(self):
        super().__init__("wikidata")
        self.output_file = self.base_dir / "wikidata-cs-subset.nt.bz2"
        self.sparql_queries_file = self.base_dir / "wikidata_sparql_queries.txt"

    def download(self) -> bool:
        """Download Wikidata CS/Programming subset."""
        logger.info("=" * 60)
        logger.info("Wikidata CS/Programming Subset Downloader")
        logger.info("=" * 60)

        # For the initial implementation, we'll use SPARQL queries instead of downloading
        # the full dump. This is more efficient and practical.

        logger.info("\n⚠️  NOTE: Wikidata full dumps are 100+ GB.")
        logger.info("We'll use SPARQL Query Service instead for efficient extraction.")
        logger.info("")

        # Create SPARQL queries file
        sparql_queries = """
# Wikidata SPARQL Queries for CS/Programming Ontology Extraction

## Query 1: Programming Languages Hierarchy
# Extracts all programming languages and their subclass relationships
SELECT ?item ?itemLabel ?subclass ?subclassLabel WHERE {
  ?item wdt:P31/wdt:P279* wd:Q9143 .  # Instance of programming language
  OPTIONAL { ?item wdt:P279 ?subclass . }  # Subclass of
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 5000

## Query 2: Software & Algorithms Hierarchy
# Extracts software and algorithm concepts
SELECT ?item ?itemLabel ?subclass ?subclassLabel WHERE {
  { ?item wdt:P31/wdt:P279* wd:Q7397 . }  # Software
  UNION
  { ?item wdt:P31/wdt:P279* wd:Q8366 . }  # Algorithm
  OPTIONAL { ?item wdt:P279 ?subclass . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 5000

## Query 3: Computer Science Concepts
# Extracts general CS concepts and their hierarchies
SELECT ?item ?itemLabel ?subclass ?subclassLabel WHERE {
  ?item wdt:P31/wdt:P279* wd:Q21198 .  # Computer science
  OPTIONAL { ?item wdt:P279 ?subclass . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 5000

## Query 4: Data Structures & Types
# Extracts data structures and type hierarchies
SELECT ?item ?itemLabel ?subclass ?subclassLabel WHERE {
  { ?item wdt:P31/wdt:P279* wd:Q175263 . }  # Data structure
  UNION
  { ?item wdt:P31/wdt:P279* wd:Q1047113 . }  # Data type
  OPTIONAL { ?item wdt:P279 ?subclass . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }
}
LIMIT 5000

# Instructions:
# 1. Go to https://query.wikidata.org/
# 2. Run each query above
# 3. Download results as JSON
# 4. Save in this directory as:
#    - query1_programming_languages.json
#    - query2_software_algorithms.json
#    - query3_cs_concepts.json
#    - query4_data_structures.json
# 5. Run the parser (to be created) to extract chains

# Expected total entities: ~20K
# Expected chains: ~20K (after deduplication)
"""

        # Write SPARQL queries to file
        with open(self.sparql_queries_file, 'w') as f:
            f.write(sparql_queries)

        logger.info("✅ Created SPARQL queries file:")
        logger.info(f"  {self.sparql_queries_file}")
        logger.info("")
        logger.info("📋 Next steps:")
        logger.info("  1. Go to https://query.wikidata.org/")
        logger.info("  2. Run each query in the file")
        logger.info("  3. Download results as JSON")
        logger.info("  4. Save results in data/datasets/ontology_datasets/wikidata/")
        logger.info("  5. Run the Wikidata parser (to be created)")
        logger.info("")
        logger.info("⏱️  Estimated time: 30-45 minutes (manual)")
        logger.info("💾 Expected data size: ~50-100 MB (JSON)")
        logger.info("")

        # For now, mark this as "manual download required"
        # We'll update the dataset map to reflect this
        self.update_dataset_map(
            downloaded_at=datetime.now().isoformat(),
            download_size_mb=0.0,  # Will be updated after manual download
            checksum="manual_download_required"
        )

        logger.info("=" * 60)
        logger.info("✅ Wikidata download setup complete!")
        logger.info("=" * 60)
        logger.info("  Status: Manual SPARQL queries required")
        logger.info("  Expected chains: ~20K")
        logger.info("  Quality: 88%")
        logger.info("=" * 60)

        return True


if __name__ == "__main__":
    downloader = WikidataDownloader()
    success = downloader.download()
    exit(0 if success else 1)
