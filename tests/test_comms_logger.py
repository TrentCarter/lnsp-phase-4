#!/usr/bin/env python3
"""
Test script for CommsLogger

Tests the flat log communication logger.
"""
import sys
import pathlib

# Add services/common to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "services"))

from common.comms_logger import get_logger, MessageType


def test_basic_logging():
    """Test basic logging functionality"""
    logger = get_logger()

    # Test 1: Log a command
    logger.log_cmd(
        from_agent="Gateway",
        to_agent="PAS Root",
        message="Submit Prime Directive: Add docstrings",
        run_id="test-run-001",
        metadata={"files": ["app.py", "utils.py"]}
    )
    print("✓ Logged CMD")

    # Test 2: Log a status update
    logger.log_status(
        from_agent="PAS Root",
        to_agent="Gateway",
        message="Started execution",
        run_id="test-run-001",
        status="running",
        progress=0.1
    )
    print("✓ Logged STATUS")

    # Test 3: Log a heartbeat
    logger.log_heartbeat(
        from_agent="Aider-LCO",
        to_agent="PAS Root",
        message="Processing file 3 of 5",
        llm_model="ollama/qwen2.5-coder:7b-instruct",
        run_id="test-run-001",
        status="running",
        progress=0.6,
        metadata={"files_done": 3, "files_total": 5}
    )
    print("✓ Logged HEARTBEAT")

    # Test 4: Log a response
    logger.log_response(
        from_agent="Aider-LCO",
        to_agent="PAS Root",
        message="Execution completed successfully",
        llm_model="ollama/qwen2.5-coder:7b-instruct",
        run_id="test-run-001",
        status="completed",
        metadata={"duration_s": 42.5, "rc": 0}
    )
    print("✓ Logged RESPONSE")

    # Test 5: Log with LLM metadata
    logger.log(
        from_agent="Dir Code",
        to_agent="Mgr Backend",
        msg_type=MessageType.CMD,
        message="Implement API endpoint /users",
        llm_model="anthropic/claude-3-7-sonnet",
        run_id="test-run-002",
        metadata={"lane": "backend", "complexity": "medium"}
    )
    print("✓ Logged with LLM metadata")

    print("\n✅ All tests passed!")
    print(f"\nCheck logs at:")
    print(f"  Global log: artifacts/logs/pas_comms_<date>.txt")
    print(f"  Per-run logs: artifacts/runs/test-run-001/comms.txt")
    print(f"\nView logs with:")
    print(f"  ./tools/parse_comms_log.py")
    print(f"  ./tools/parse_comms_log.py --run-id test-run-001")
    print(f"  ./tools/parse_comms_log.py --type HEARTBEAT")
    print(f"  ./tools/parse_comms_log.py --llm qwen")


if __name__ == "__main__":
    test_basic_logging()
