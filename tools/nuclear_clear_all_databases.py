#!/usr/bin/env python3
"""
Nuclear Clear: Remove ALL data from ALL databases
==================================================

Clears:
- PostgreSQL: All tables (cpe_entry, cpe_vectors, etc.)
- Neo4j: All nodes and relationships
- FAISS: All index files
- NPZ: All vector files
"""

import psycopg2
import subprocess
from pathlib import Path
import sys

print("\n" + "="*80)
print("🔴 NUCLEAR DATABASE CLEAR")
print("="*80 + "\n")

# 1. Clear PostgreSQL
print("1️⃣  Clearing PostgreSQL...")
try:
    conn = psycopg2.connect("dbname=lnsp")
    cur = conn.cursor()

    # Get all tables
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
    """)
    tables = [row[0] for row in cur.fetchall()]

    print(f"   Found {len(tables)} tables: {', '.join(tables)}")

    # Truncate all tables (CASCADE to handle foreign keys)
    for table in tables:
        print(f"   Truncating {table}...")
        cur.execute(f"TRUNCATE TABLE {table} CASCADE")

    conn.commit()
    cur.close()
    conn.close()

    print("   ✅ PostgreSQL cleared\n")
except Exception as e:
    print(f"   ⚠️  PostgreSQL error: {e}\n")

# 2. Clear Neo4j
print("2️⃣  Clearing Neo4j...")
try:
    # Delete all nodes and relationships
    result = subprocess.run([
        'cypher-shell', '-u', 'neo4j', '-p', 'password',
        'MATCH (n) DETACH DELETE n'
    ], capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print("   ✅ Neo4j cleared\n")
    else:
        print(f"   ⚠️  Neo4j error: {result.stderr}\n")
except Exception as e:
    print(f"   ⚠️  Neo4j error: {e}\n")

# 3. Clear FAISS indices
print("3️⃣  Clearing FAISS indices...")
artifacts_dir = Path("artifacts")
if artifacts_dir.exists():
    index_files = list(artifacts_dir.glob("**/*.index"))
    print(f"   Found {len(index_files)} FAISS index files")

    for index_file in index_files:
        print(f"   Deleting {index_file}")
        index_file.unlink()

    print("   ✅ FAISS indices cleared\n")
else:
    print("   ⚠️  No artifacts directory found\n")

# 4. Clear NPZ vector files
print("4️⃣  Clearing NPZ vector files...")
if artifacts_dir.exists():
    npz_files = list(artifacts_dir.glob("**/*vectors*.npz"))
    print(f"   Found {len(npz_files)} NPZ files")

    for npz_file in npz_files:
        print(f"   Deleting {npz_file}")
        npz_file.unlink()

    print("   ✅ NPZ files cleared\n")
else:
    print("   ⚠️  No artifacts directory found\n")

# 5. Verification
print("="*80)
print("🔍 VERIFICATION")
print("="*80 + "\n")

# Check PostgreSQL
try:
    conn = psycopg2.connect("dbname=lnsp")
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cpe_entry")
    count = cur.fetchone()[0]
    print(f"PostgreSQL cpe_entry count: {count} (should be 0)")

    cur.execute("SELECT COUNT(*) FROM cpe_vectors")
    count = cur.fetchone()[0]
    print(f"PostgreSQL cpe_vectors count: {count} (should be 0)")

    cur.close()
    conn.close()
except Exception as e:
    print(f"PostgreSQL verification error: {e}")

# Check Neo4j
try:
    result = subprocess.run([
        'cypher-shell', '-u', 'neo4j', '-p', 'password',
        'MATCH (n) RETURN count(n) as count'
    ], capture_output=True, text=True, timeout=10)

    if result.returncode == 0:
        print(f"Neo4j node count: {result.stdout.strip()}")
except Exception as e:
    print(f"Neo4j verification error: {e}")

print("\n" + "="*80)
print("✅ NUCLEAR CLEAR COMPLETE")
print("="*80 + "\n")
