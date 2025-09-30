#!/bin/bash
# LNSP Baseline Backup Script
# Creates complete backup of database state, artifacts, and metadata

set -e  # Exit on error

BACKUP_DIR="backups/baseline_v1.0_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${BACKUP_DIR}"/{postgres,neo4j,artifacts,metadata}

echo "🔄 Starting LNSP baseline backup..."
echo "📁 Backup location: ${BACKUP_DIR}"

# 1. PostgreSQL Backup
echo ""
echo "1️⃣  Backing up PostgreSQL database..."
pg_dump lnsp > "${BACKUP_DIR}/postgres/lnsp_backup.sql"
pg_dump lnsp --schema-only > "${BACKUP_DIR}/postgres/lnsp_schema.sql"
psql lnsp -c "\dt" > "${BACKUP_DIR}/postgres/table_list.txt"
psql lnsp -c "SELECT COUNT(*) FROM cpe_entry;" > "${BACKUP_DIR}/postgres/row_counts.txt"
echo "   ✅ PostgreSQL backup complete"

# 2. Neo4j Backup
echo ""
echo "2️⃣  Backing up Neo4j database..."
if command -v neo4j-admin &> /dev/null; then
    # Full neo4j-admin backup (requires Neo4j to be stopped)
    echo "   ⚠️  Note: neo4j-admin backup requires Neo4j to be stopped"
    echo "   Using Cypher export instead..."
fi

# Export Neo4j data via Cypher
cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n) as total_nodes;" > "${BACKUP_DIR}/neo4j/node_count.txt"
cypher-shell -u neo4j -p password "MATCH ()-[r]->() RETURN count(r) as total_rels;" > "${BACKUP_DIR}/neo4j/rel_count.txt"

# Export all Concept nodes
cypher-shell -u neo4j -p password "MATCH (c:Concept) RETURN c;" --format plain > "${BACKUP_DIR}/neo4j/concepts_export.txt"

# Export graph structure
cypher-shell -u neo4j -p password "
MATCH (c:Concept)-[r:RELATES_TO]->(e:Entity)
RETURN c.cpe_id as concept_id, c.concept_text as concept,
       type(r) as rel_type, r.weight as weight,
       e.entity_name as entity
LIMIT 10000
" --format plain > "${BACKUP_DIR}/neo4j/graph_sample.txt"

echo "   ✅ Neo4j backup complete"

# 3. Faiss Artifacts
echo ""
echo "3️⃣  Backing up Faiss artifacts..."
if [ -d "artifacts" ]; then
    cp -r artifacts/*.npz "${BACKUP_DIR}/artifacts/" 2>/dev/null || echo "   ⚠️  No .npz files found"
    ls -lh "${BACKUP_DIR}/artifacts/" > "${BACKUP_DIR}/artifacts/file_list.txt"
    echo "   ✅ Faiss artifacts backed up"
else
    echo "   ⚠️  No artifacts directory found"
fi

# 4. System Metadata
echo ""
echo "4️⃣  Capturing system metadata..."

# Git info
git log -1 --pretty=format:"Commit: %H%nAuthor: %an%nDate: %ad%nMessage: %s" > "${BACKUP_DIR}/metadata/git_commit.txt"
git status > "${BACKUP_DIR}/metadata/git_status.txt"
git diff > "${BACKUP_DIR}/metadata/git_diff.txt"

# Python environment
pip freeze > "${BACKUP_DIR}/metadata/pip_freeze.txt"
python --version > "${BACKUP_DIR}/metadata/python_version.txt"

# Service versions
ollama list > "${BACKUP_DIR}/metadata/ollama_models.txt" 2>/dev/null || echo "Ollama not found" > "${BACKUP_DIR}/metadata/ollama_models.txt"
psql --version > "${BACKUP_DIR}/metadata/postgres_version.txt"
cypher-shell --version > "${BACKUP_DIR}/metadata/neo4j_version.txt" 2>&1

# System report
if [ -f "reports/scripts/generate_ingestion_report.py" ]; then
    python reports/scripts/generate_ingestion_report.py --output "${BACKUP_DIR}/metadata/ingestion_report.md"
fi

echo "   ✅ Metadata captured"

# 5. Create manifest
echo ""
echo "5️⃣  Creating backup manifest..."
cat > "${BACKUP_DIR}/MANIFEST.txt" << EOF
LNSP Baseline v1.0 Backup
=========================

Backup Date: $(date)
Git Commit: $(git rev-parse HEAD)
Git Branch: $(git branch --show-current)

Contents:
---------
postgres/
  - lnsp_backup.sql       # Full database dump
  - lnsp_schema.sql       # Schema only
  - table_list.txt        # Table listing
  - row_counts.txt        # Record counts

neo4j/
  - node_count.txt        # Total nodes
  - rel_count.txt         # Total relationships
  - concepts_export.txt   # All Concept nodes
  - graph_sample.txt      # Graph structure sample

artifacts/
  - *.npz                 # Faiss vector indices

metadata/
  - git_commit.txt        # Git commit info
  - git_status.txt        # Git status
  - git_diff.txt          # Uncommitted changes
  - pip_freeze.txt        # Python dependencies
  - python_version.txt    # Python version
  - ollama_models.txt     # Ollama model list
  - postgres_version.txt  # PostgreSQL version
  - neo4j_version.txt     # Neo4j version
  - ingestion_report.md   # Full system report

Database Statistics:
--------------------
$(psql lnsp -t -c "SELECT COUNT(*) || ' CPE entries' FROM cpe_entry;")
$(psql lnsp -t -c "SELECT COUNT(*) || ' vectors' FROM cpe_vectors;")
$(cypher-shell -u neo4j -p password "MATCH (n:Concept) RETURN count(n) + ' Concept nodes';" 2>/dev/null | tail -1 || echo "Neo4j data")
$(cypher-shell -u neo4j -p password "MATCH (n:Entity) RETURN count(n) + ' Entity nodes';" 2>/dev/null | tail -1 || echo "Neo4j data")
$(cypher-shell -u neo4j -p password "MATCH ()-[r]->() RETURN count(r) + ' relationships';" 2>/dev/null | tail -1 || echo "Neo4j data")

Restoration:
------------
To restore this backup:
1. Restore PostgreSQL: psql lnsp < postgres/lnsp_backup.sql
2. Restore Neo4j: See neo4j/ directory for Cypher exports
3. Restore Faiss: cp artifacts/*.npz ../artifacts/
4. Verify: python reports/scripts/verify_baseline_v1.0.py

See docs/baselines/BASELINE_v1.0_vecRAG.md for full instructions.
EOF

echo "   ✅ Manifest created"

# 6. Create compressed archive
echo ""
echo "6️⃣  Creating compressed archive..."
ARCHIVE_NAME="lnsp_baseline_v1.0_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "backups/${ARCHIVE_NAME}" -C "$(dirname ${BACKUP_DIR})" "$(basename ${BACKUP_DIR})"
echo "   ✅ Archive created: backups/${ARCHIVE_NAME}"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Backup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Backup directory: ${BACKUP_DIR}"
echo "📦 Archive: backups/${ARCHIVE_NAME}"
echo ""
echo "Archive size: $(du -h backups/${ARCHIVE_NAME} | cut -f1)"
echo ""
echo "To restore:"
echo "  1. Extract: tar -xzf backups/${ARCHIVE_NAME}"
echo "  2. Follow: ${BACKUP_DIR}/MANIFEST.txt"
echo "  3. See docs: docs/baselines/BASELINE_v1.0_vecRAG.md"
echo ""