#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Real-time Updates End-to-End Test${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="$PROJECT_ROOT/artifacts/registry/registry.db"

# Check if database exists
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}✗ Registry database not found at: $DB_PATH${NC}"
    exit 1
fi

# Clean up old test data
echo -e "${YELLOW}➜ Cleaning up old test data...${NC}"
sqlite3 "$DB_PATH" "DELETE FROM action_logs WHERE task_id LIKE 'test-realtime-%'"
echo -e "${GREEN}✓ Cleaned up old test data${NC}"
echo ""

# Check if HMI server is running
echo -e "${YELLOW}➜ Checking if HMI server is running...${NC}"
if curl -s http://localhost:6101/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ HMI server is running on port 6101${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  IMPORTANT: If you ran this test before, restart the HMI server!${NC}"
    echo -e "${YELLOW}   The server initializes its SSE tracker on startup.${NC}"
    echo ""
    echo -e "${YELLOW}To restart:${NC}"
    echo -e "  1. Stop the current HMI server (Ctrl+C)"
    echo -e "  2. Start it again: ./.venv/bin/python services/webui/hmi_app.py"
    echo -e "  3. Re-run this test script"
    echo ""
    read -p "Press ENTER if you just restarted the server, or Ctrl+C to restart now..."
else
    echo -e "${RED}✗ HMI server is NOT running!${NC}"
    echo ""
    echo -e "${YELLOW}Please start the HMI server first:${NC}"
    echo -e "  cd $PROJECT_ROOT"
    echo -e "  ./.venv/bin/python services/webui/hmi_app.py"
    echo ""
    exit 1
fi
echo ""

# Instructions for the user
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  INSTRUCTIONS${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "1. ${YELLOW}Open your browser${NC} and navigate to:"
echo -e "   ${GREEN}http://localhost:6101/tree?task_id=test-realtime-003${NC}"
echo ""
echo -e "2. ${YELLOW}Open the browser console${NC} (F12 or Cmd+Opt+I)"
echo -e "   You should see:"
echo -e "   - SSE connection messages"
echo -e "   - Real-time update events"
echo ""
echo -e "3. ${YELLOW}Also open the Sequencer${NC} in another tab:"
echo -e "   ${GREEN}http://localhost:6101/sequencer?task_id=test-realtime-003${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press ENTER when you're ready to start the simulation...${NC}"
read -r

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Simulation${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Watch the browser for real-time updates!${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the simulation${NC}"
echo ""

# Run the Python test script
cd "$PROJECT_ROOT"
exec ./.venv/bin/python3 << 'PYTHON_SCRIPT'
import sqlite3
import time
import os
import json
from datetime import datetime

DB_PATH = "artifacts/registry/registry.db"
TASK_ID = "test-realtime-003"

def insert_action_log(action_type, action_name, from_agent, to_agent, tier_from, tier_to, status='running'):
    """Insert a new action_log entry"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO action_logs
        (task_id, action_type, action_name, from_agent, to_agent, tier_from, tier_to, status, timestamp, action_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        TASK_ID,
        action_type,
        action_name,
        from_agent,
        to_agent,
        tier_from,
        tier_to,
        status,
        datetime.now().isoformat(),
        json.dumps({'test': True, 'step': action_name})
    ))

    conn.commit()
    log_id = cursor.lastrowid
    conn.close()

    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"  [{timestamp}] ✓ {action_name}")
    print(f"            ({from_agent} → {to_agent}, log_id={log_id})")
    return log_id

# Delegation flow scenario
scenarios = [
    # Step 1: User -> VP
    {
        'action_type': 'delegate',
        'action_name': 'User requests new feature',
        'from_agent': 'user',
        'to_agent': 'vp_001',
        'tier_from': 0,
        'tier_to': 1,
        'status': 'running'
    },
    # Step 2: VP -> Director
    {
        'action_type': 'delegate',
        'action_name': 'VP delegates to Code Director',
        'from_agent': 'vp_001',
        'to_agent': 'director_code',
        'tier_from': 1,
        'tier_to': 2,
        'status': 'running'
    },
    # Step 3: Director -> Manager
    {
        'action_type': 'delegate',
        'action_name': 'Director assigns to Backend Manager',
        'from_agent': 'director_code',
        'to_agent': 'manager_backend',
        'tier_from': 2,
        'tier_to': 3,
        'status': 'running'
    },
    # Step 4: Manager -> Programmer
    {
        'action_type': 'delegate',
        'action_name': 'Manager assigns task to Programmer',
        'from_agent': 'manager_backend',
        'to_agent': 'programmer_001',
        'tier_from': 3,
        'tier_to': 4,
        'status': 'running'
    },
    # Step 5: Programmer works on feature
    {
        'action_type': 'execute',
        'action_name': 'Programmer implements feature',
        'from_agent': 'programmer_001',
        'to_agent': None,
        'tier_from': 4,
        'tier_to': 4,
        'status': 'running'
    },
    # Step 6: Programmer completes -> Manager
    {
        'action_type': 'report',
        'action_name': 'Programmer completes implementation',
        'from_agent': 'programmer_001',
        'to_agent': 'manager_backend',
        'tier_from': 4,
        'tier_to': 3,
        'status': 'completed'
    },
    # Step 7: Manager reviews -> Director
    {
        'action_type': 'report',
        'action_name': 'Manager reviews and approves',
        'from_agent': 'manager_backend',
        'to_agent': 'director_code',
        'tier_from': 3,
        'tier_to': 2,
        'status': 'completed'
    },
    # Step 8: Director validates -> VP
    {
        'action_type': 'report',
        'action_name': 'Director validates quality',
        'from_agent': 'director_code',
        'to_agent': 'vp_001',
        'tier_from': 2,
        'tier_to': 1,
        'status': 'completed'
    },
    # Step 9: VP reports to user
    {
        'action_type': 'report',
        'action_name': 'VP reports completion to user',
        'from_agent': 'vp_001',
        'to_agent': 'user',
        'tier_from': 1,
        'tier_to': 0,
        'status': 'completed'
    },
]

print("Inserting action logs (3 second intervals)...\n")

for i, scenario in enumerate(scenarios, 1):
    print(f"Step {i}/{ len(scenarios)}:")
    insert_action_log(**scenario)
    print()

    # Wait 3 seconds between steps (except after last one)
    if i < len(scenarios):
        for remaining in range(3, 0, -1):
            print(f"  Next update in {remaining}s...", end='\r', flush=True)
            time.sleep(1)
        print(" " * 40, end='\r')  # Clear the countdown line

print("\n" + "="*64)
print("✓ Simulation complete!")
print("="*64)
print("\nExpected results:")
print("  Tree View:")
print("    • 9 nodes (user, vp_001, director_code, manager_backend, programmer_001)")
print("    • Green pulsing glow on new nodes/edges")
print("    • Hierarchical tree layout (auto-expanding)")
print()
print("  Sequencer:")
print("    • 9 tasks visible on timeline")
print("    • Timeline starts at 0:00 (left edge = task start)")
print("    • Time labels: 0:00, 5:00, 10:00, etc. (minutes:seconds)")
print("    • Tasks have text labels (action names)")
print("    • Color-coded by status (blue/yellow/green/gray)")
print("    • Auto-plays at 1x speed (for active tasks)")
print()
print("If you didn't see updates:")
print("  1. Restart HMI server (it initializes SSE tracker on startup)")
print("  2. Check browser console for SSE connection errors")
print("  3. Verify Network tab shows /api/stream/tree/... connection")
print()
PYTHON_SCRIPT
