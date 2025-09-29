import pytest
from src.db_neo4j import Neo4jDB, get_driver

# Mark all tests in this file as slow, as they require a database connection
pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def neo4j_db():
    """Provides a Neo4jDB instance and cleans up the test data afterwards."""
    pytest.importorskip("neo4j")
    db = Neo4jDB(enabled=True)
    if not db.driver:
        pytest.skip("Neo4j not available")

    # Setup: Create some test nodes
    with db.driver.session() as session:
        session.run("""
        MERGE (c1:Concept {cpe_id: 'test:cpe:001'}) SET c1.concept_text = 'Concept 1'
        MERGE (c2:Concept {cpe_id: 'test:cpe:002'}) SET c2.concept_text = 'Concept 2'
        """)

    yield db

    # Teardown: Clean up test data
    with db.driver.session() as session:
        session.run("MATCH (c:Concept) WHERE c.cpe_id STARTS WITH 'test:' DETACH DELETE c")
    db.close()

def test_insert_relation_idempotency(neo4j_db):
    """Test that inserting the same relation twice doesn't create duplicates."""
    src_id, dst_id, rel_type = "test:cpe:001", "test:cpe:002", "IS_A"

    # Insert twice
    assert neo4j_db.insert_relation_triple(src_id, dst_id, rel_type) is True
    assert neo4j_db.insert_relation_triple(src_id, dst_id, rel_type) is True

    # Verify only one relationship exists
    with neo4j_db.driver.session() as session:
        result = session.run("""
        MATCH (s:Concept {cpe_id: $src_id})-[r]->(d:Concept {cpe_id: $dst_id})
        WHERE r.type = $rel_type
        RETURN count(r) as count
        """, src_id=src_id, dst_id=dst_id, rel_type=rel_type)
        assert result.single()["count"] == 1

def test_duplicate_concept_fails_gracefully(neo4j_db):
    """Test that creating a concept with a duplicate cpe_id fails gracefully."""
    # This test relies on a unique constraint on cpe_id. 
    # First, ensure the constraint exists.
    with neo4j_db.driver.session() as session:
        session.run("CREATE CONSTRAINT concept_cpe_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.cpe_id IS UNIQUE")

    # The fixture already created 'test:cpe:001'. Attempting to insert it again 
    # via a method that doesn't use MERGE should fail.
    # We'll simulate a direct create call that violates the constraint.
    with neo4j_db.driver.session() as session:
        try:
            # This direct CREATE should violate the constraint
            session.run("CREATE (:Concept {cpe_id: 'test:cpe:001'})")
            # If it doesn't, the test setup is wrong.
            pytest.fail("Duplicate concept creation did not raise an error, constraint may be missing.")
        except Exception as e:
            # We expect a ConstraintError
            assert "already exists" in str(e).lower()

    # The `insert_concept` method uses MERGE, so it should succeed.
    record = {'cpe_id': 'test:cpe:001', 'concept_text': 'Updated Concept 1'}
    assert neo4j_db.insert_concept(record) is True
