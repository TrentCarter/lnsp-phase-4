#!/usr/bin/env python3
"""
Test PostgreSQL database initialization for LNSP pipeline.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import psycopg2
    from psycopg2 import sql
    print("✓ psycopg2 available")
except ImportError:
    print("✗ psycopg2 not available - install with: pip install psycopg2-binary")
    sys.exit(1)

def test_pg_connection():
    """Test connection to PostgreSQL database."""
    # Use environment variables or defaults
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    dbname = os.getenv("PG_DBNAME", "lnsp")
    user = os.getenv("PG_USER", "lnsp")
    password = os.getenv("PG_PASSWORD", "lnsp")

    try:
        print(f"Connecting to PostgreSQL: {user}@{host}:{port}/{dbname}")
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        print("✓ Connection successful")

        # Test basic operations
        cur = conn.cursor()

        # Check extensions
        cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'vector');")
        extensions = [row[0] for row in cur.fetchall()]
        print(f"✓ Extensions found: {extensions}")

        if 'uuid-ossp' not in extensions:
            print("⚠️ uuid-ossp extension not found - run: CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
        if 'vector' not in extensions:
            print("⚠️ vector extension not found - run: CREATE EXTENSION IF NOT EXISTS vector;")

        # Check tables
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'cpe%';")
        tables = [row[0] for row in cur.fetchall()]
        print(f"✓ CPE tables found: {tables}")

        # Check enums
        cur.execute("SELECT typname FROM pg_type WHERE typname IN ('content_type', 'validation_status');")
        enums = [row[0] for row in cur.fetchall()]
        print(f"✓ Enums found: {enums}")

        cur.close()
        conn.close()
        print("✓ Database check complete")

    except psycopg2.OperationalError as e:
        print(f"✗ Connection failed: {e}")
        print("\nTo start PostgreSQL:")
        print("1. Install PostgreSQL locally, or")
        print("2. Use Docker: docker run -d --name postgres-lnsp -e POSTGRES_DB=lnsp -e POSTGRES_USER=lnsp -e POSTGRES_PASSWORD=lnsp -p 5432:5432 postgres:15")
        print("3. Run init script: PGPASSWORD=lnsp psql -h localhost -U lnsp -d lnsp -f scripts/init_postgres.sql")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    return True

def test_schema_creation():
    """Test schema creation by reading and validating the SQL file."""
    sql_file = Path(__file__).parent / "scripts" / "init_postgres.sql"

    if not sql_file.exists():
        print(f"✗ SQL file not found: {sql_file}")
        return False

    print(f"✓ Found SQL file: {sql_file}")

    # Read and validate SQL content
    with open(sql_file, 'r') as f:
        sql_content = f.read()

    # Check for required components
    required_elements = [
        "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"",
        "CREATE EXTENSION IF NOT EXISTS vector",
        "CREATE TYPE content_type AS ENUM",
        "CREATE TYPE validation_status AS ENUM",
        "CREATE TABLE IF NOT EXISTS cpe_entry",
        "CREATE TABLE IF NOT EXISTS cpe_vectors",
        "CREATE INDEX IF NOT EXISTS cpe_lane_idx"
    ]

    for element in required_elements:
        if element in sql_content:
            print(f"✓ Found: {element}")
        else:
            print(f"✗ Missing: {element}")
            return False

    print("✓ SQL schema validation passed")
    return True

if __name__ == "__main__":
    print("=== LNSP PostgreSQL Database Test ===\n")

    # Test schema file
    schema_ok = test_schema_creation()
    print()

    # Test connection
    conn_ok = test_pg_connection()
    print()

    if schema_ok and conn_ok:
        print("🎉 All database tests passed!")
    elif schema_ok:
        print("⚠️ Schema is valid but database connection failed")
        print("Run: docker compose up -d postgres")
    else:
        print("❌ Database setup issues detected")
