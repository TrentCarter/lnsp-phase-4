#!/usr/bin/env python3
"""
Programmer Service Generator

Generates 10 Programmer FastAPI services from template:
- Programmer-001 to Programmer-010
- Ports 6151-6160
- All LLM-agnostic (runtime-selectable)
- Generic naming (not LLM-specific)

Based on Programmer-001 template.
"""
import sys
from pathlib import Path
import shutil

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Template paths
TEMPLATE_DIR = PROJECT_ROOT / "services" / "tools" / "programmer_001"
TEMPLATE_APP = TEMPLATE_DIR / "app.py"
TEMPLATE_INIT = TEMPLATE_DIR / "__init__.py"
TEMPLATE_CONFIG = PROJECT_ROOT / "configs" / "pas" / "programmer_001.yaml"

# Service configuration
PROGRAMMERS = [
    {"num": 1, "port": 6151, "agent_id": "Prog-001", "name": "Programmer-001", "parent": "Mgr-Code-01"},
    {"num": 2, "port": 6152, "agent_id": "Prog-002", "name": "Programmer-002", "parent": "Mgr-Code-01"},
    {"num": 3, "port": 6153, "agent_id": "Prog-003", "name": "Programmer-003", "parent": "Mgr-Code-01"},
    {"num": 4, "port": 6154, "agent_id": "Prog-004", "name": "Programmer-004", "parent": "Mgr-Code-02"},
    {"num": 5, "port": 6155, "agent_id": "Prog-005", "name": "Programmer-005", "parent": "Mgr-Code-02"},
    {"num": 6, "port": 6156, "agent_id": "Prog-006", "name": "Programmer-006", "parent": "Mgr-Code-03"},
    {"num": 7, "port": 6157, "agent_id": "Prog-007", "name": "Programmer-007", "parent": "Mgr-Code-03"},
    {"num": 8, "port": 6158, "agent_id": "Prog-008", "name": "Programmer-008", "parent": "Mgr-Models-01"},
    {"num": 9, "port": 6159, "agent_id": "Prog-009", "name": "Programmer-009", "parent": "Mgr-Data-01"},
    {"num": 10, "port": 6160, "agent_id": "Prog-010", "name": "Programmer-010", "parent": "Mgr-Docs-01"},
]


def generate_app_py(template_path: Path, output_path: Path, config: dict):
    """Generate app.py from template with replacements"""
    content = template_path.read_text()

    # Replace service-specific values
    replacements = {
        'SERVICE_NAME = "Programmer-001"': f'SERVICE_NAME = "{config["name"]}"',
        'SERVICE_PORT = 6151': f'SERVICE_PORT = {config["port"]}',
        'AGENT_ID = "Prog-001"': f'AGENT_ID = "{config["agent_id"]}"',
        'PARENT_AGENT = "Mgr-Code-01"': f'PARENT_AGENT = "{config["parent"]}"',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    output_path.write_text(content)
    print(f"  ✓ Generated {output_path.relative_to(PROJECT_ROOT)}")


def generate_init_py(template_path: Path, output_path: Path, config: dict):
    """Generate __init__.py from template"""
    content = f'"""{config["name"]} Service - LLM-Agnostic Code Execution Agent"""'
    output_path.write_text(content + "\n")
    print(f"  ✓ Generated {output_path.relative_to(PROJECT_ROOT)}")


def generate_config_yaml(template_path: Path, output_path: Path, config: dict):
    """Generate config YAML from template with replacements"""
    content = template_path.read_text()

    # Replace service-specific values
    replacements = {
        'name: "Programmer-001"': f'name: "{config["name"]}"',
        'port: 6151': f'port: {config["port"]}',
        'agent_id: "Prog-001"': f'agent_id: "{config["agent_id"]}"',
        'parent: "Mgr-Code-01"': f'parent: "{config["parent"]}"',
    }

    for old, new in replacements.items():
        content = content.replace(old, new)

    output_path.write_text(content)
    print(f"  ✓ Generated {output_path.relative_to(PROJECT_ROOT)}")


def generate_programmer_service(config: dict, skip_existing: bool = False):
    """Generate a complete Programmer service from template"""
    num = config["num"]
    name = config["name"]

    print(f"\n{name} (Port {config['port']}):")

    # Create service directory
    service_dir = PROJECT_ROOT / "services" / "tools" / f"programmer_{num:03d}"
    if service_dir.exists():
        if skip_existing:
            print(f"  ⊘ Skipping (already exists)")
            return
        else:
            print(f"  ⚠ Overwriting existing service")

    service_dir.mkdir(parents=True, exist_ok=True)

    # Generate files
    generate_app_py(TEMPLATE_APP, service_dir / "app.py", config)
    generate_init_py(TEMPLATE_INIT, service_dir / "__init__.py", config)

    # Generate config
    config_path = PROJECT_ROOT / "configs" / "pas" / f"programmer_{num:03d}.yaml"
    generate_config_yaml(TEMPLATE_CONFIG, config_path, config)


def main():
    """Generate all Programmer services"""
    print("=" * 60)
    print("Programmer Service Generator")
    print("=" * 60)

    # Check if template exists
    if not TEMPLATE_APP.exists():
        print(f"❌ Template not found: {TEMPLATE_APP}")
        sys.exit(1)

    print(f"\nGenerating {len(PROGRAMMERS)} Programmer services...")

    # Generate services (skip Programmer-001 since it's the template)
    for config in PROGRAMMERS:
        if config["num"] == 1:
            print(f"\n{config['name']} (Port {config['port']}):")
            print(f"  ⊘ Skipping (template)")
            continue

        generate_programmer_service(config, skip_existing=False)

    print("\n" + "=" * 60)
    print("✓ Generation complete!")
    print("=" * 60)

    # Print summary
    print("\nGenerated Programmers:")
    for config in PROGRAMMERS:
        print(f"  {config['name']:<20} Port {config['port']}  Parent: {config['parent']}")

    print("\nNext steps:")
    print("  1. Review generated services in services/tools/programmer_*/")
    print("  2. Run: bash scripts/start_all_programmers.sh")
    print("  3. Verify: curl http://localhost:6151/health (and 6152-6160)")


if __name__ == "__main__":
    main()
