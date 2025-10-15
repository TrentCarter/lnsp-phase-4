#!/bin/bash
# scripts/nuclear_clear.sh
# Complete database clear with automatic backup
set -e

echo "🚨 WARNING: This will DELETE ALL DATA from PostgreSQL, FAISS, and Neo4j"
echo "Backup will be created automatically"
echo ""
echo "Current data:"
echo "  PostgreSQL: $(psql -h localhost -U lnsp -d lnsp -tAc 'SELECT COUNT(*) FROM cpe_entry;') concepts"
echo "  Neo4j: $(cypher-shell -u neo4j -p password --format plain 'MATCH (n) RETURN count(n);' 2>/dev/null | tail -1) nodes"
echo "  FAISS: $(ls -1 artifacts/*.index 2>/dev/null | wc -l | xargs) index files"
echo ""
read -p "Type 'DELETE ALL' to confirm: " CONFIRM

if [ "$CONFIRM" != "DELETE ALL" ]; then
    echo "❌ Aborted"
    exit 1
fi

# 1. Backup
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/pre_clear_$DATE"
mkdir -p "$BACKUP_DIR"

echo ""
echo "📦 Creating backup at $BACKUP_DIR..."
pg_dump -h localhost -U lnsp -d lnsp -F c -f "$BACKUP_DIR/lnsp.dump"
cp artifacts/*.{index,json,npz} "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup complete ($(du -sh $BACKUP_DIR | cut -f1))"

# 2. Clear PostgreSQL
echo ""
echo "🗑️  Clearing PostgreSQL..."
psql -h localhost -U lnsp -d lnsp -c "TRUNCATE TABLE cpe_entry CASCADE;" >/dev/null
echo "✅ PostgreSQL cleared"

# 3. Clear FAISS
echo ""
echo "🗑️  Clearing FAISS..."
rm -f artifacts/*.index artifacts/faiss_meta.json artifacts/index_meta.json
echo "✅ FAISS indexes cleared"

# 4. Clear Neo4j
echo ""
echo "🗑️  Clearing Neo4j..."
cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n;" 2>/dev/null
echo "✅ Neo4j graph cleared"

# 5. Verify
echo ""
echo "📊 Verification:"
PG_COUNT=$(psql -h localhost -U lnsp -d lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")
NEO4J_COUNT=$(cypher-shell -u neo4j -p password --format plain "MATCH (n) RETURN count(n);" 2>/dev/null | tail -1)
FAISS_COUNT=$(ls -1 artifacts/*.index 2>/dev/null | wc -l | xargs)

echo "  PostgreSQL: $PG_COUNT concepts (should be 0)"
echo "  Neo4j: $NEO4J_COUNT nodes (should be 0)"
echo "  FAISS: $FAISS_COUNT index files (should be 0)"
echo "  Backup: $BACKUP_DIR"

if [ "$PG_COUNT" = "0" ] && [ "$NEO4J_COUNT" = "0" ] && [ "$FAISS_COUNT" = "0" ]; then
    echo ""
    echo "✅ All databases cleared successfully"
    echo ""
    echo "Next steps:"
    echo "  1. Start ingestion API: ./.venv/bin/uvicorn app.api.ingest_chunks:app --host 127.0.0.1 --port 8004"
    echo "  2. Ingest ontology data via FastAPI pipeline"
    echo "  3. Verify 3-way sync: ./scripts/verify_data_sync.sh"
    exit 0
else
    echo ""
    echo "❌ ERROR: Clear incomplete"
    exit 1
fi
