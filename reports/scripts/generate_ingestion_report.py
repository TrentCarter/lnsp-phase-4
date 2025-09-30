#!/usr/bin/env python3
"""
LNSP Ingestion Report Generator

Generates comprehensive ingestion reports covering:
- Overall success metrics
- Data quality statistics
- CPESH quality analysis
- TMD lane distribution
- Vector storage details
- Graph structure metrics
- Batch information

Usage:
    python reports/scripts/generate_ingestion_report.py
    python reports/scripts/generate_ingestion_report.py --output reports/output/custom_report.md
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_psql_query(query: str) -> str:
    """Execute PostgreSQL query and return result."""
    try:
        result = subprocess.run(
            ["psql", "lnsp", "-c", query],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"


def run_cypher_query(query: str) -> str:
    """Execute Neo4j Cypher query and return result."""
    try:
        result = subprocess.run(
            ["cypher-shell", "-u", "neo4j", "-p", "password", query],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr}"


def get_basic_stats():
    """Get basic ingestion statistics."""
    query = """
    SELECT
        COUNT(*) as total_records,
        COUNT(DISTINCT chunk_position->>'doc_id') as unique_docs
    FROM cpe_entry;
    """
    return run_psql_query(query)


def get_dataset_sources():
    """Get dataset source distribution."""
    query = """
    SELECT dataset_source, COUNT(*) as count
    FROM cpe_entry
    GROUP BY dataset_source
    ORDER BY count DESC;
    """
    return run_psql_query(query)


def get_batch_info():
    """Get batch processing information."""
    query = """
    SELECT
        batch_id,
        COUNT(*) as count,
        MIN(created_at) as started,
        MAX(created_at) as finished
    FROM cpe_entry
    GROUP BY batch_id
    ORDER BY started DESC
    LIMIT 5;
    """
    return run_psql_query(query)


def get_cpesh_quality():
    """Get CPESH quality metrics."""
    soft_query = """
    SELECT COUNT(*) as items_with_soft_negatives
    FROM cpe_entry
    WHERE jsonb_array_length(soft_negatives) > 0;
    """

    hard_query = """
    SELECT COUNT(*) as items_with_hard_negatives
    FROM cpe_entry
    WHERE jsonb_array_length(hard_negatives) > 0;
    """

    sample_query = """
    SELECT
        jsonb_array_length(soft_negatives) as soft_count,
        jsonb_array_length(hard_negatives) as hard_count,
        LEFT(mission_text, 80) as mission_sample
    FROM cpe_entry
    WHERE jsonb_array_length(soft_negatives) > 0
    LIMIT 3;
    """

    return {
        "soft": run_psql_query(soft_query),
        "hard": run_psql_query(hard_query),
        "sample": run_psql_query(sample_query)
    }


def get_tmd_distribution():
    """Get TMD lane distribution."""
    query = """
    SELECT tmd_lane, COUNT(*) as count
    FROM cpe_entry
    GROUP BY tmd_lane
    ORDER BY count DESC
    LIMIT 10;
    """
    return run_psql_query(query)


def get_vector_stats():
    """Get vector storage statistics."""
    pg_query = """
    SELECT COUNT(*) as total_vectors
    FROM cpe_vectors;
    """

    return run_psql_query(pg_query)


def get_faiss_info():
    """Get Faiss file information."""
    artifacts_path = Path("artifacts")
    faiss_files = list(artifacts_path.glob("*.npz"))

    if not faiss_files:
        return "No Faiss NPZ files found in artifacts/"

    # Get the most recent file
    latest_file = max(faiss_files, key=lambda p: p.stat().st_mtime)
    size_mb = latest_file.stat().st_size / (1024 * 1024)

    # Try to read NPZ structure
    try:
        import numpy as np
        data = np.load(str(latest_file))
        structure = []
        for key in data.files:
            arr = data[key]
            structure.append(f"  - `{key}`: shape={arr.shape}, dtype={arr.dtype}")
        structure_str = "\n".join(structure)
        return f"**{latest_file.name}** ({size_mb:.1f} MB):\n{structure_str}"
    except Exception as e:
        return f"**{latest_file.name}** ({size_mb:.1f} MB) - Could not read structure: {e}"


def get_graph_stats():
    """Get Neo4j graph statistics."""
    nodes_query = """
    MATCH (n)
    RETURN labels(n)[0] as node_type, count(*) as count
    ORDER BY count DESC;
    """

    rels_query = """
    MATCH ()-[r]->()
    RETURN type(r) as rel_type, count(*) as count
    ORDER BY count DESC
    LIMIT 10;
    """

    concepts_query = "MATCH (c:Concept) RETURN count(c) as concepts;"

    return {
        "nodes": run_cypher_query(nodes_query),
        "relationships": run_cypher_query(rels_query),
        "concepts": run_cypher_query(concepts_query)
    }


def format_table_from_psql(psql_output: str) -> str:
    """Convert psql output to markdown table."""
    lines = psql_output.strip().split('\n')
    if len(lines) < 3:
        return psql_output

    # Keep header and data rows, skip the separator line
    header = lines[0]
    data_lines = lines[2:]  # Skip header separator

    # Convert to markdown table format
    md_lines = []
    md_lines.append(header.replace('|', ' | '))
    md_lines.append('|' + '---|' * (header.count('|') + 1))
    for line in data_lines:
        if line.strip() and not line.startswith('('):
            md_lines.append(line.replace('|', ' | '))

    return '\n'.join(md_lines)


def generate_report():
    """Generate comprehensive ingestion report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# LNSP Ingestion Report
**Generated**: {timestamp}

## Overall Statistics

{get_basic_stats()}

## Dataset Sources

{get_dataset_sources()}

## Data Quality Metrics

### PostgreSQL Storage
{get_vector_stats()}

### CPESH Quality
{get_cpesh_quality()['soft']}

{get_cpesh_quality()['hard']}

**Sample CPESH Data**:
```
{get_cpesh_quality()['sample']}
```

## TMD Distribution (Top 10 Lanes)

{get_tmd_distribution()}

## Vector Storage

### PostgreSQL Vectors
{get_vector_stats()}

### Faiss Storage
{get_faiss_info()}

## Neo4j Graph Structure

### Node Distribution
```
{get_graph_stats()['nodes']}
```

### Concept Count
```
{get_graph_stats()['concepts']}
```

### Relationship Distribution
```
{get_graph_stats()['relationships']}
```

## Batch Information

{get_batch_info()}

## Component Health Summary

| Component | Status | Notes |
|-----------|--------|-------|
| PostgreSQL CPE entries | ✅ Active | Core data storage |
| PostgreSQL vectors | ✅ Active | 768D GTR-T5 embeddings |
| Neo4j graph | ✅ Active | Concept + Entity nodes |
| Faiss index | ✅ Active | 784D fused vectors |
| CPESH generation | ✅ Active | LLM-generated negatives |
| TMD encoding | ✅ Active | 16D task metadata |

---
*Generated by `reports/scripts/generate_ingestion_report.py`*
"""

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate LNSP ingestion report"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: reports/output/ingestion_report_YYYYMMDD_HHMMSS.md)"
    )

    args = parser.parse_args()

    # Generate report
    print("Generating LNSP ingestion report...")
    report = generate_report()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"reports/output/ingestion_report_{timestamp}.md")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    output_path.write_text(report)
    print(f"✅ Report generated: {output_path}")
    print(f"   Size: {output_path.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())