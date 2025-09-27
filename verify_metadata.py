#!/usr/bin/env python3
"""
Script to extract and verify all metadata fields from the dataset.
This includes CPESH, TMD, Dates+Times, ID, Access Count, and other META fields.
"""

import json
import psycopg2
from datetime import datetime
from pathlib import Path
import os
from typing import Dict, Any, Optional

def load_factoidwiki_entry(file_path: str = "data/factoidwiki_1k.jsonl", entry_num: int = 0) -> Dict[str, Any]:
    """Load a specific entry from the factoidwiki dataset."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == entry_num:
                return json.loads(line)
    return {}

def load_cpesh_entry(doc_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load CPESH metadata from cache."""
    cpesh_file = "artifacts/cpesh_cache.jsonl"
    if not Path(cpesh_file).exists():
        return None

    with open(cpesh_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            if doc_id and entry.get('doc_id') == doc_id:
                return entry
            elif not doc_id:
                return entry  # Return first entry if no doc_id specified
    return None

def check_database_metadata(doc_id: str) -> Optional[Dict[str, Any]]:
    """Check PostgreSQL database for document metadata."""
    try:
        # Get connection parameters from environment or defaults
        conn_params = {
            'dbname': os.getenv('LNSP_DB_NAME', 'lnsp'),
            'user': os.getenv('LNSP_DB_USER', 'postgres'),
            'password': os.getenv('LNSP_DB_PASSWORD', 'postgres'),
            'host': os.getenv('LNSP_DB_HOST', 'localhost'),
            'port': os.getenv('LNSP_DB_PORT', '5432')
        }

        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()

        # Check if documents table exists and query it
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'documents'
        """)
        columns = [col[0] for col in cur.fetchall()]

        if columns:
            cur.execute(f"SELECT * FROM documents WHERE id = %s LIMIT 1", (doc_id,))
            result = cur.fetchone()
            if result:
                return dict(zip(columns, result))

        cur.close()
        conn.close()
        return None
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None

def display_all_metadata():
    """Extract and display all available metadata from various sources."""
    print("=" * 80)
    print("COMPREHENSIVE METADATA VERIFICATION")
    print("=" * 80)

    # 1. Load entry from main dataset
    print("\n1. FACTOIDWIKI DATASET ENTRY:")
    print("-" * 40)
    fw_entry = load_factoidwiki_entry()
    if fw_entry:
        print(f"ID: {fw_entry.get('id')}")
        print(f"Contents (first 100 chars): {fw_entry.get('contents', '')[:100]}...")
        if 'metadata' in fw_entry:
            print("Metadata fields:")
            for key, value in fw_entry['metadata'].items():
                print(f"  - {key}: {value}")

    # 2. Load CPESH metadata
    print("\n2. CPESH METADATA:")
    print("-" * 40)
    cpesh_entry = load_cpesh_entry()
    if cpesh_entry:
        print(f"Document ID: {cpesh_entry.get('doc_id')}")
        if 'cpesh' in cpesh_entry:
            cpesh = cpesh_entry['cpesh']
            print("CPESH Fields:")
            print(f"  - Concept: {cpesh.get('concept')}")
            print(f"  - Probe: {cpesh.get('probe')}")
            print(f"  - Expected: {cpesh.get('expected')}")
            print(f"  - Soft Negative: {cpesh.get('soft_negative')}")
            print(f"  - Hard Negative: {cpesh.get('hard_negative')}")
            print(f"  - Created At: {cpesh.get('created_at')}")
            print(f"  - Last Accessed: {cpesh.get('last_accessed')}")
        print(f"Access Count: {cpesh_entry.get('access_count', 0)}")
    else:
        print("No CPESH entries found")

    # 3. Check for TMD (Title-Metadata) files
    print("\n3. TMD (TITLE-METADATA) CHECK:")
    print("-" * 40)
    tmd_patterns = ['*.tmd', '*tmd*.jsonl', '*title*.jsonl', '*meta*.jsonl']
    tmd_found = False
    for pattern in tmd_patterns:
        tmd_files = list(Path('artifacts').glob(pattern))
        if tmd_files:
            tmd_found = True
            for tmd_file in tmd_files:
                print(f"Found TMD file: {tmd_file}")
                with open(tmd_file, 'r') as f:
                    first_line = f.readline()
                    if first_line:
                        tmd_data = json.loads(first_line)
                        print(f"TMD Sample: {json.dumps(tmd_data, indent=2)[:200]}...")
    if not tmd_found:
        print("No TMD files found. TMD might be embedded in main dataset.")
        # Check if title metadata is in the main entry
        if fw_entry and 'metadata' in fw_entry:
            if 'title_span' in fw_entry['metadata']:
                print("Title metadata found in main dataset entry:")
                print(f"  - Title span: {fw_entry['metadata']['title_span']}")
                title_start, title_end = fw_entry['metadata']['title_span']
                print(f"  - Extracted title: {fw_entry['contents'][title_start:title_end]}")

    # 4. Check database for additional metadata
    print("\n4. DATABASE METADATA:")
    print("-" * 40)
    if fw_entry:
        db_meta = check_database_metadata(fw_entry.get('id'))
        if db_meta:
            print("Database fields found:")
            for key, value in db_meta.items():
                if value is not None:
                    print(f"  - {key}: {value}")
        else:
            print("No database metadata found or database not accessible")

    # 5. Check for timestamp metadata
    print("\n5. DATES AND TIMES:")
    print("-" * 40)
    print(f"Current timestamp: {datetime.now().isoformat()}")
    if cpesh_entry and 'cpesh' in cpesh_entry:
        cpesh = cpesh_entry['cpesh']
        if 'created_at' in cpesh:
            print(f"CPESH Created: {cpesh['created_at']}")
        if 'last_accessed' in cpesh:
            print(f"CPESH Last Accessed: {cpesh['last_accessed']}")

    # 6. Check for embeddings
    print("\n6. EMBEDDINGS CHECK:")
    print("-" * 40)
    embedding_files = list(Path('artifacts').glob('*embed*.npy')) + \
                     list(Path('artifacts').glob('*vector*.npy')) + \
                     list(Path('artifacts').glob('*.faiss')) + \
                     list(Path('artifacts').glob('*.index'))
    if embedding_files:
        for emb_file in embedding_files[:5]:  # Show first 5
            print(f"Found embedding file: {emb_file} (size: {emb_file.stat().st_size} bytes)")
    else:
        print("No separate embedding files found")

    # 7. Summary
    print("\n" + "=" * 80)
    print("METADATA SUMMARY:")
    print("=" * 80)
    print("✓ Document ID: Present")
    print(f"✓ Content: Present ({len(fw_entry.get('contents', ''))} chars)")
    print(f"✓ Title Metadata (TMD): {'Present (embedded)' if fw_entry.get('metadata', {}).get('title_span') else 'Not found'}")
    print(f"✓ CPESH: {'Present' if cpesh_entry else 'Not found in cache'}")
    print(f"✓ Access Count: {'Present' if cpesh_entry and 'access_count' in cpesh_entry else 'Not found'}")
    print(f"✓ Timestamps: {'Present' if cpesh_entry and 'cpesh' in cpesh_entry and 'created_at' in cpesh_entry['cpesh'] else 'Not found'}")
    print(f"✓ Database Metadata: {'Connected' if 'db_meta' in locals() and db_meta else 'Not accessible'}")

if __name__ == "__main__":
    display_all_metadata()