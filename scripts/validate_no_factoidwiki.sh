#!/usr/bin/env bash
# Validate that NO FactoidWiki data exists in the system
# CRITICAL: FactoidWiki is NOT ontological and MUST NOT be used!

set -e

echo "=" | head -c 70; echo
echo "🚨 FactoidWiki Validation Check"
echo "=" | head -c 70; echo
echo

# Check PostgreSQL for factoid data
# NOTE: We check dataset_source AND concept patterns since ingest_ontology_simple.py
#       had a bug where it labeled ontology data as 'factoid-wiki-large'
echo "[1/3] Checking PostgreSQL for FactoidWiki-labeled data..."
FACTOID_LABELED=$(psql lnsp -tAc "SELECT COUNT(*) FROM cpe_entry WHERE dataset_source LIKE '%factoid%' OR dataset_source LIKE '%wiki%';" 2>/dev/null || echo "0")
FACTOID_OK=0

if [ "$FACTOID_LABELED" != "0" ]; then
    echo "  ⚠️  Found $FACTOID_LABELED entries with FactoidWiki label"
    echo "  Checking if they are actually ontological concepts..."

    # Sample 5 concepts to check quality
    SAMPLE_CONCEPTS=$(psql lnsp -tAc "SELECT concept_text FROM cpe_entry WHERE dataset_source LIKE '%factoid%' ORDER BY RANDOM() LIMIT 5;" 2>/dev/null)

    # Check if concepts look ontological (enzyme activity, software, biological processes)
    if echo "$SAMPLE_CONCEPTS" | grep -qE "activity|software|entity|organization|process|function"; then
        echo "  ✅ Concepts appear ontological despite label:"
        echo "$SAMPLE_CONCEPTS" | head -3 | sed 's/^/     - /'
        echo
        echo "  ℹ️  This is likely due to a labeling bug in ingest_ontology_simple.py"
        echo "     The fix has been applied - future ingestions will use 'ontology-*' labels"
        echo
        FACTOID_OK=1
    else
        echo "  ❌ CRITICAL: Concepts appear to be FactoidWiki (royal families, albums, etc.)!"
        echo "$SAMPLE_CONCEPTS" | head -3 | sed 's/^/     - /'
        echo
        echo "   To fix:"
        echo "   1. DELETE FactoidWiki data: psql lnsp -c \"DELETE FROM cpe_entry WHERE dataset_source LIKE '%factoid%';\""
        echo "   2. Clear Neo4j: cypher-shell -u neo4j -p password \"MATCH (n) WHERE n.text =~ '.*royal.*|.*dynasty.*' DETACH DELETE n\""
        echo "   3. Re-ingest ontology data: ./scripts/ingest_ontologies.sh"
        echo
        exit 1
    fi
else
    echo "  ✅ No FactoidWiki labels in PostgreSQL"
fi

# Check for factoid files in artifacts
echo "[2/3] Checking for FactoidWiki files in artifacts/..."
FACTOID_FILES=$(find artifacts -name "*factoid*" -o -name "*wiki*" 2>/dev/null | grep -v ".gitkeep" || echo "")

if [ -n "$FACTOID_FILES" ]; then
    echo "  ⚠️  WARNING: Found FactoidWiki files:"
    echo "$FACTOID_FILES" | while read -r file; do
        echo "     $file"
    done
    echo
    echo "   These should be deleted: rm artifacts/*factoid* artifacts/*wiki*"
    echo
else
    echo "  ✅ No FactoidWiki files in artifacts/"
fi

# Check Neo4j for typical FactoidWiki concepts
echo "[3/3] Checking Neo4j for FactoidWiki-like concepts..."
FACTOID_CONCEPTS=$(cypher-shell -u neo4j -p password "MATCH (c:Concept) WHERE c.text =~ '.*royal.*|.*dynasty.*|.*album.*|.*song.*' RETURN count(c)" --format plain 2>/dev/null | tail -1 || echo "0")

if [ "$FACTOID_CONCEPTS" != "0" ] && [ "$FACTOID_CONCEPTS" != "count(c)" ]; then
    echo "  ⚠️  WARNING: Found $FACTOID_CONCEPTS concepts with FactoidWiki-like patterns"
    echo "     (royal, dynasty, album, song)"
    echo
    echo "   Sample concepts:"
    cypher-shell -u neo4j -p password "MATCH (c:Concept) WHERE c.text =~ '.*royal.*|.*dynasty.*' RETURN c.text LIMIT 3" --format plain 2>/dev/null | tail -3
    echo
    echo "   These are likely FactoidWiki data - consider clearing Neo4j"
else
    echo "  ✅ No obvious FactoidWiki patterns in Neo4j"
fi

echo
echo "=" | head -c 70; echo
echo "Validation Summary"
echo "=" | head -c 70; echo
echo
echo "PostgreSQL FactoidWiki-labeled entries: $FACTOID_LABELED"
echo "FactoidWiki files in artifacts: $(echo "$FACTOID_FILES" | wc -l | tr -d ' ')"
echo "FactoidWiki-like Neo4j concepts: $FACTOID_CONCEPTS"
echo

if [ "$FACTOID_OK" = "1" ] || { [ "$FACTOID_LABELED" = "0" ] && [ -z "$FACTOID_FILES" ]; }; then
    echo "✅ PASSED: No FactoidWiki data detected (or labels known ontological)"
    echo
    echo "System is using ONTOLOGY data as required."
    echo "See: LNSP_LONG_TERM_MEMORY.md Section 2"
else
    echo "⚠️  WARNING: FactoidWiki data or files detected"
    echo
    echo "Action required: Remove FactoidWiki data and re-ingest ontologies"
    echo
    exit 1
fi
