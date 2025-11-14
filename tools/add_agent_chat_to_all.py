#!/usr/bin/env python3
"""
Automatically Add Agent Chat Mixin to All Agents

This script adds the Agent Chat Mixin to all agents that don't have it yet.

It performs the following modifications to each app.py:
1. Adds import for agent_chat_mixin
2. Adds import for agent_chat (if not present)
3. Initializes agent_chat client (if not present)
4. Adds handle_incoming_message function
5. Adds startup event handler with message poller
6. Adds agent_chat_routes via add_agent_chat_routes()

Usage:
    python tools/add_agent_chat_to_all.py --dry-run  # Preview changes
    python tools/add_agent_chat_to_all.py            # Apply changes
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Agent configurations (agent_id, parent_agent, port, file_path)
AGENTS_TO_UPDATE = [
    # Directors
    ("Dir-Models", "Architect", 6112, "services/pas/director_models/app.py"),

    # Managers
    ("Mgr-Models-01", "Dir-Models", 6144, "services/pas/manager_models_01/app.py"),
    ("Mgr-Data-01", "Dir-Data", 6145, "services/pas/manager_data_01/app.py"),
    ("Mgr-DevSecOps-01", "Dir-DevSecOps", 6146, "services/pas/manager_devsecops_01/app.py"),
    ("Mgr-Docs-01", "Dir-Docs", 6147, "services/pas/manager_docs_01/app.py"),

    # Programmers (single app.py serves all 10 instances via PROGRAMMER_ID env var)
    ("Prog-ALL", "Manager", 6151, "services/tools/aider_rpc/app.py"),
]


def check_if_has_agent_chat(content: str) -> bool:
    """Check if file already has Agent Chat"""
    return "agent_chat_mixin" in content or "add_agent_chat_routes" in content


def check_if_has_agent_chat_client(content: str) -> bool:
    """Check if file already initializes agent_chat client"""
    return "agent_chat = get_agent_chat_client()" in content


def add_imports(content: str) -> str:
    """Add necessary imports"""
    # Find the imports section (after the docstring)
    lines = content.split("\n")

    # Find where to insert imports (after existing agent_chat import or after other imports)
    insert_idx = None
    for i, line in enumerate(lines):
        if "from services.common.agent_chat import" in line:
            # Insert after this line
            insert_idx = i + 1
            break
        elif line.startswith("from services.common") and "import" in line:
            # Remember this as potential insertion point
            insert_idx = i + 1

    if insert_idx is None:
        # Fallback: insert after app = FastAPI line
        for i, line in enumerate(lines):
            if "app = FastAPI" in line:
                insert_idx = i
                break

    if insert_idx is None:
        return content  # Can't find insertion point

    # Check if agent_chat import exists
    has_agent_chat_import = any("from services.common.agent_chat import" in line for line in lines)

    # Build import lines
    import_lines = []

    if not has_agent_chat_import:
        import_lines.append("from services.common.agent_chat import get_agent_chat_client, AgentChatMessage")

    import_lines.extend([
        "from services.common.agent_chat_mixin import (",
        "    add_agent_chat_routes,",
        "    start_message_poller,",
        "    send_message_to_parent,",
        "    send_message_to_child",
        ")"
    ])

    # Insert import lines
    lines = lines[:insert_idx] + import_lines + lines[insert_idx:]

    return "\n".join(lines)


def add_agent_chat_client_init(content: str, agent_id: str) -> str:
    """Add agent_chat client initialization if not present"""
    if check_if_has_agent_chat_client(content):
        return content  # Already has it

    # Find where to insert (after other service initializations)
    lines = content.split("\n")
    insert_idx = None

    for i, line in enumerate(lines):
        if "logger = get_logger()" in line or "heartbeat_monitor = get_monitor()" in line:
            insert_idx = i + 1

    if insert_idx is None:
        return content

    # Add initialization
    lines.insert(insert_idx, "agent_chat = get_agent_chat_client()")

    return "\n".join(lines)


def add_message_handler(content: str, agent_id: str, parent_agent: str) -> str:
    """Add handle_incoming_message function"""
    # Check if already exists
    if "async def handle_incoming_message" in content:
        return content

    # For Programmers, use get_agent_id() function for dynamic ID
    agent_id_var = "get_agent_id()" if agent_id == "Prog-ALL" else f'"{agent_id}"'

    handler_code = f'''

# === Agent Chat Message Handler ===

async def handle_incoming_message(message: AgentChatMessage):
    """
    Handle incoming messages from parent ({parent_agent}) or children.

    Called automatically by the message poller when new messages arrive.
    """
    print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Received {{{{message.message_type}}}} from {{{{message.from_agent}}}}")

    if message.message_type == "delegation":
        # Handle delegation from parent
        print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Delegation: {{{{message.content}}}}")
        # TODO: Process delegation

    elif message.message_type == "question":
        # Handle question from child
        print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Question: {{{{message.content}}}}")
        # TODO: Answer question

    elif message.message_type == "status":
        # Handle status update
        print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Status: {{{{message.content}}}}")

    elif message.message_type == "completion":
        # Handle completion
        print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Completion: {{{{message.content}}}}")

    elif message.message_type == "error":
        # Handle error
        print(f"[{{{{get_agent_id() if 'get_agent_id' in dir() else '{agent_id}'}}}}] Error: {{{{message.content}}}}")
'''

    # Insert before the first @app.get or @app.post
    lines = content.split("\n")
    insert_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("@app.get") or line.strip().startswith("@app.post"):
            insert_idx = i
            break

    if insert_idx is None:
        # Fallback: insert at end
        return content + handler_code

    lines = lines[:insert_idx] + [handler_code] + lines[insert_idx:]
    return "\n".join(lines)


def add_startup_handler(content: str, agent_id: str) -> str:
    """Add startup event handler with message poller"""
    # Check if already exists
    if '@app.on_event("startup")' in content or "@app.on_event('startup')" in content:
        # Check if it has start_message_poller
        if "start_message_poller" in content:
            return content  # Already complete
        else:
            # Add start_message_poller to existing startup
            # This is complex, so skip for now
            return content

    # For Programmers, use get_agent_id() for dynamic ID
    agent_id_expr = "get_agent_id()" if agent_id == "Prog-ALL" else f'"{agent_id}"'

    startup_code = f'''

# === Startup - Initialize Agent Chat ===

@app.on_event("startup")
async def startup():
    """Initialize agent and start message poller"""
    agent_id_val = {agent_id_expr}
    print(f"[{{agent_id_val}}] Starting up...")

    # Start message poller
    await start_message_poller(
        agent_id=agent_id_val,
        agent_chat=agent_chat,
        poll_interval=2.0
    )

    print(f"[{{agent_id_val}}] Startup complete - Agent Chat enabled")
'''

    # Insert at end before the last line (usually if __name__ == "__main__")
    lines = content.split("\n")

    # Find insertion point (before if __name__ block)
    insert_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if 'if __name__ == "__main__"' in lines[i] or "if __name__ == '__main__'" in lines[i]:
            insert_idx = i
            break

    lines = lines[:insert_idx] + [startup_code] + lines[insert_idx:]
    return "\n".join(lines)


def add_agent_chat_routes_call(content: str, agent_id: str) -> str:
    """Add add_agent_chat_routes() call"""
    # Check if already exists
    if "add_agent_chat_routes(" in content:
        return content

    # For Programmers, use get_agent_id() for dynamic ID
    agent_id_expr = "get_agent_id()" if agent_id == "Prog-ALL" else f'"{agent_id}"'

    routes_code = f'''

# === Add Agent Chat Routes ===

_agent_id_for_chat = {agent_id_expr}
add_agent_chat_routes(
    app=app,
    agent_id=_agent_id_for_chat,
    agent_chat=agent_chat,
    on_message_received=handle_incoming_message
)

print(f"[{{_agent_id_for_chat}}] Agent Chat routes added")
'''

    # Insert after app = FastAPI(...) or near the end
    lines = content.split("\n")

    # Find insertion point (after startup handler or before if __name__)
    insert_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if 'if __name__ == "__main__"' in lines[i] or "if __name__ == '__main__'" in lines[i]:
            insert_idx = i
            break
        elif '@app.on_event("startup")' in lines[i] or "@app.on_event('startup')" in lines[i]:
            # Find end of startup function
            for j in range(i, len(lines)):
                if lines[j].strip() and not lines[j].startswith(" ") and j > i:
                    insert_idx = j
                    break

    lines = lines[:insert_idx] + [routes_code] + lines[insert_idx:]
    return "\n".join(lines)


def update_agent_file(agent_id: str, parent_agent: str, port: int, file_path: str, dry_run: bool = False) -> bool:
    """Update a single agent file"""
    path = Path(file_path)

    if not path.exists():
        print(f"⚠️  {agent_id}: File not found: {file_path}")
        return False

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Processing {agent_id} ({file_path})...")

    # Read file
    content = path.read_text()

    # Check if already has Agent Chat
    if check_if_has_agent_chat(content):
        print(f"  ✓ {agent_id} already has Agent Chat Mixin")
        return True

    # Apply transformations
    original_content = content
    content = add_imports(content)
    content = add_agent_chat_client_init(content, agent_id)
    content = add_message_handler(content, agent_id, parent_agent)
    content = add_startup_handler(content, agent_id)
    content = add_agent_chat_routes_call(content, agent_id)

    if content == original_content:
        print(f"  ⚠️  No changes made to {agent_id}")
        return False

    if dry_run:
        print(f"  ℹ️  Would update {agent_id}")
        return True

    # Write file
    path.write_text(content)
    print(f"  ✅ Updated {agent_id}")
    return True


def main():
    """Main entry point"""
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("Agent Chat Mixin - Automatic Integration")
    print("=" * 60)

    if dry_run:
        print("\n⚠️  DRY RUN MODE - No files will be modified\n")

    updated = 0
    failed = 0
    skipped = 0

    for agent_id, parent_agent, port, file_path in AGENTS_TO_UPDATE:
        result = update_agent_file(agent_id, parent_agent, port, file_path, dry_run)
        if result:
            updated += 1
        else:
            if Path(file_path).exists():
                skipped += 1
            else:
                failed += 1

    print("\n" + "=" * 60)
    print(f"Summary: {updated} updated, {skipped} skipped, {failed} failed")
    print("=" * 60)

    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
