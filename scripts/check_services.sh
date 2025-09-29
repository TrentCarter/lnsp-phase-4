#!/usr/bin/env bash
# ===================================================================
# LNSP Phase-4: Service Check and Startup Helper
# ===================================================================
# Checks and optionally starts required services for ingestion:
#   - PostgreSQL (required)
#   - Neo4j (optional for graph features)
#
# Usage: source ./scripts/check_services.sh
#
# Environment variables:
#   NO_DOCKER     - Skip docker checks (default: 0)
#   SKIP_NEO4J    - Skip Neo4j startup prompts (default: 0)
#   AUTO_START    - Auto-start services without prompting (default: 0)
# ===================================================================

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check and start PostgreSQL
check_postgresql() {
    local PG_OK=true

    if command -v pg_isready >/dev/null 2>&1; then
        if pg_isready -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ PostgreSQL is running${NC}"
        else
            echo -e "${RED}❌ PostgreSQL is not running${NC}"
            PG_OK=false

            # Try to start PostgreSQL if brew services available
            if command -v brew >/dev/null 2>&1; then
                if [ "${AUTO_START:-0}" = "1" ]; then
                    START_PG="y"
                else
                    echo -e "${BLUE}   Would you like to start PostgreSQL? [Y/n]: ${NC}"
                    read -n 1 -r START_PG
                    echo
                    START_PG=${START_PG:-y}
                fi

                if [[ $START_PG =~ ^[Yy]$ ]]; then
                    echo -e "${YELLOW}   Attempting to start PostgreSQL...${NC}"
                    brew services start postgresql@14 2>/dev/null || brew services start postgresql 2>/dev/null
                    sleep 3
                    if pg_isready -h "${PGHOST:-localhost}" -p "${PGPORT:-5432}" >/dev/null 2>&1; then
                        echo -e "${GREEN}   ✓ PostgreSQL started successfully${NC}"
                        PG_OK=true
                    else
                        echo -e "${RED}   Failed to start PostgreSQL${NC}"
                        echo -e "${YELLOW}   Try: brew services restart postgresql@14${NC}"
                    fi
                fi
            else
                echo -e "${YELLOW}   Please start PostgreSQL manually:${NC}"
                echo -e "${BLUE}     brew services start postgresql@14${NC}"
            fi
        fi
    else
        echo -e "${YELLOW}⚠️  pg_isready not found - PostgreSQL status unknown${NC}"
        echo -e "${YELLOW}   Proceeding, but database operations may fail${NC}"
        PG_OK=true  # Don't block if we can't check
    fi

    if [ "$PG_OK" = false ]; then
        echo -e "${RED}══════════════════════════════════════════════${NC}"
        echo -e "${RED}❌ PostgreSQL is required but not available${NC}"
        echo -e "${RED}══════════════════════════════════════════════${NC}"
        return 1
    fi
    return 0
}

# Function to check and optionally start Neo4j
check_neo4j() {
    local NEO4J_RUNNING=false
    local SKIP_NEO4J=${SKIP_NEO4J:-0}

    if nc -z localhost 7687 2>/dev/null; then
        echo -e "${GREEN}✓ Neo4j is running (port 7687 open)${NC}"
        NEO4J_RUNNING=true
    else
        echo -e "${YELLOW}⚠️  Neo4j is not running (port 7687 closed)${NC}"
        echo -e "${YELLOW}   Neo4j is optional - graph features will be disabled${NC}"

        if [ "$SKIP_NEO4J" != "1" ]; then
            # Check if Neo4j is installed via brew
            if command -v brew >/dev/null 2>&1 && brew list neo4j >/dev/null 2>&1; then
                if [ "${AUTO_START:-0}" = "1" ]; then
                    START_NEO4J="n"  # Default to no for auto-start
                else
                    echo -e "${BLUE}   Would you like to start Neo4j? [y/N]: ${NC}"
                    read -n 1 -r START_NEO4J
                    echo
                fi

                if [[ $START_NEO4J =~ ^[Yy]$ ]]; then
                    echo -e "${YELLOW}   Starting Neo4j via brew services...${NC}"
                    brew services start neo4j
                    echo -e "${YELLOW}   Waiting for Neo4j to start (this may take 10-20 seconds)...${NC}"

                    # Wait up to 30 seconds for Neo4j to start
                    for i in {1..30}; do
                        if nc -z localhost 7687 2>/dev/null; then
                            echo -e "${GREEN}   ✓ Neo4j started successfully${NC}"
                            NEO4J_RUNNING=true
                            break
                        fi
                        sleep 1
                        if [ $((i % 5)) -eq 0 ]; then
                            echo -e "${YELLOW}   Still waiting... ($i seconds)${NC}"
                        fi
                    done

                    if [ "$NEO4J_RUNNING" = false ]; then
                        echo -e "${YELLOW}   Neo4j is taking longer to start. Proceeding without it.${NC}"
                    fi
                else
                    echo -e "${BLUE}   Proceeding without Neo4j (graph features disabled)${NC}"
                fi
            else
                # Check if Neo4j can be installed
                if command -v brew >/dev/null 2>&1; then
                    echo -e "${YELLOW}   Neo4j not installed. To enable graph features:${NC}"
                    echo -e "${BLUE}     brew install neo4j${NC}"
                    echo -e "${BLUE}     brew services start neo4j${NC}"
                fi
                echo -e "${BLUE}   Proceeding without Neo4j (graph features disabled)${NC}"
            fi
        else
            echo -e "${BLUE}   SKIP_NEO4J=1 - proceeding without Neo4j${NC}"
        fi
    fi

    # Neo4j is optional, always return success
    return 0
}

# Main service check function
check_all_services() {
    echo -e "${BLUE}══════════════════════════════════════════════${NC}"
    echo -e "${BLUE}Checking Required Services${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════${NC}"
    echo ""

    if [ "${NO_DOCKER:-0}" = "1" ]; then
        echo -e "${YELLOW}📝 NO_DOCKER mode - ensure services are running manually${NC}"
        echo ""
    fi

    # Check PostgreSQL (required)
    if ! check_postgresql; then
        return 1
    fi

    echo ""

    # Check Neo4j (optional)
    check_neo4j

    echo ""
    echo -e "${GREEN}✓ Service checks complete${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════${NC}"
    echo ""

    return 0
}

# Export functions if sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    export -f check_postgresql
    export -f check_neo4j
    export -f check_all_services
else
    # If run directly, execute checks
    check_all_services
fi