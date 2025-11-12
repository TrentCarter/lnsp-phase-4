#!/usr/bin/env python3
"""
Generate Manager Service Templates

Creates Manager services for all lanes based on Manager-Code-01 template.

Usage:
    python tools/generate_manager_services.py
"""
import os
import shutil
from pathlib import Path

# Manager definitions: (lane, number, port)
MANAGERS = [
    ("code", 1, 6141),  # Already exists (template)
    ("code", 2, 6142),
    ("code", 3, 6143),
    ("models", 1, 6144),
    ("data", 1, 6145),
    ("devsecops", 1, 6146),
    ("docs", 1, 6147),
]

# Base paths
BASE_DIR = Path(__file__).parent.parent
TEMPLATE_DIR = BASE_DIR / "services" / "pas" / "manager_code_01"
SERVICES_DIR = BASE_DIR / "services" / "pas"
CONFIGS_DIR = BASE_DIR / "configs" / "pas"


def generate_manager(lane: str, num: int, port: int):
    """Generate Manager service from template"""
    # Service directory name
    service_name = f"manager_{lane}_{num:02d}"
    service_dir = SERVICES_DIR / service_name

    # Agent ID and display name
    agent_id = f"Mgr-{lane.title()}-{num:02d}"
    display_name = f"Manager-{lane.title()}-{num:02d}"

    # Parent director
    parent_director = f"Dir-{lane.title()}"

    print(f"Generating {display_name}...")

    # Skip if already exists (template)
    if service_dir.exists() and lane == "code" and num == 1:
        print(f"  ✓ Template already exists")
        return

    # Create service directory
    service_dir.mkdir(exist_ok=True)

    # Copy and modify app.py
    app_py = TEMPLATE_DIR / "app.py"
    dest_app_py = service_dir / "app.py"

    with open(app_py, "r") as f:
        content = f.read()

    # Replace placeholders
    content = content.replace("Manager-Code-01", display_name)
    content = content.replace("manager_code_01", service_name)
    content.replace(f"Port: 6141", f"Port: {port}")
    content = content.replace('SERVICE_NAME = "Manager-Code-01"', f'SERVICE_NAME = "{display_name}"')
    content = content.replace("SERVICE_PORT = 6141", f"SERVICE_PORT = {port}")
    content = content.replace('AGENT_ID = "Mgr-Code-01"', f'AGENT_ID = "{agent_id}"')
    content = content.replace('PARENT_AGENT = "Dir-Code"', f'PARENT_AGENT = "{parent_director}"')
    content = content.replace('LANE = "Code"', f'LANE = "{lane.title()}"')
    content = content.replace("MGR_CODE_01_PRIMARY_LLM", f"MGR_{lane.upper()}_{num:02d}_PRIMARY_LLM")
    content = content.replace("MGR_CODE_01_BACKUP_LLM", f"MGR_{lane.upper()}_{num:02d}_BACKUP_LLM")

    with open(dest_app_py, "w") as f:
        f.write(content)

    # Create __init__.py
    init_py = service_dir / "__init__.py"
    with open(init_py, "w") as f:
        f.write(f'"""{display_name} Service - {lane.title()} Lane Task Breakdown"""\n')
        f.write('__version__ = "1.0.0"\n')

    # Create config file
    config_file = CONFIGS_DIR / f"{service_name}.yaml"
    config_content = f"""# {display_name} Configuration

service:
  name: "{display_name}"
  port: {port}
  host: "127.0.0.1"
  tier: 4
  lane: "{lane.title()}"

agent_metadata:
  agent_id: "{agent_id}"
  role: "manager"
  parent: "{parent_director}"
  grandparent: "Architect"
  tier: "manager"
  lane: "{lane.title()}"

llm:
  primary:
    provider: "google"
    model: "gemini-2.5-flash"
    temperature: 0.3
    max_tokens: 4096
    api_key_env: "GEMINI_API_KEY"
  backup:
    provider: "anthropic"
    model: "claude-haiku-4"
    temperature: 0.3
    max_tokens: 4096
    api_key_env: "ANTHROPIC_API_KEY"

resources:
  max_concurrent_tasks: 5
  max_programmers: 5
  timeout_s: 600

decomposition:
  strategy: "surgical"  # Break into atomic, single-file tasks
  max_tasks_per_job: 10
  min_tasks_per_job: 1
  prefer_parallel: true
"""

    with open(config_file, "w") as f:
        f.write(config_content)

    print(f"  ✓ Created {service_dir}/")
    print(f"  ✓ Created {config_file}")


def main():
    """Generate all Manager services"""
    print("=" * 80)
    print("Generating Manager Services")
    print("=" * 80)

    for lane, num, port in MANAGERS:
        generate_manager(lane, num, port)

    print("\n" + "=" * 80)
    print("✓ Manager services generated successfully")
    print("=" * 80)
    print(f"\nGenerated {len(MANAGERS)} Manager services:")
    for lane, num, port in MANAGERS:
        print(f"  - Manager-{lane.title()}-{num:02d} (port {port})")


if __name__ == "__main__":
    main()
