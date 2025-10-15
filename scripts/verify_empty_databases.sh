#!/bin/bash
# scripts/verify_empty_databases.sh
# Verify all databases are empty after clear

echo "=== Database Clear Verification ==="
echo ""

# PostgreSQL
PG_COUNT=$(psql -h localhost -U lnsp -d lnsp -tAc "SELECT COUNT(*) FROM cpe_entry;")
echo "PostgreSQL: $PG_COUNT concepts (should be 0)"

# FAISS
FAISS_COUNT=$(ls -1 artifacts/*.index 2>/dev/null | wc -l | xargs)
echo "FAISS: $FAISS_COUNT index files (should be 0)"

# Neo4j
NEO4J_COUNT=$(cypher-shell -u neo4j -p password --format plain \
  "MATCH (n) RETURN count(n);" 2>/dev/null | tail -1)
echo "Neo4j: $NEO4J_COUNT nodes (should be 0)"

echo ""

# Check all are zero
if [ "$PG_COUNT" = "0" ] && [ "$FAISS_COUNT" = "0" ] && [ "$NEO4J_COUNT" = "0" ]; then
    echo "✅ All databases are empty"
    exit 0
else
    echo "❌ ERROR: Some databases still have data"
    exit 1
fi
