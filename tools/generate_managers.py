#!/usr/bin/env python3
"""Generate Manager services from template"""

from pathlib import Path

# Read template
template_path = Path("services/pas/manager_code_01/app.py")
template = template_path.read_text()

# Manager configurations
managers = [
    {
        "name": "Manager-Models-01",
        "agent_id": "Mgr-Models-01",
        "port": 6144,
        "parent": "Dir-Models",
        "lane": "Models",
        "env_prefix": "MGR_MODELS_01",
        "dir": "manager_models_01"
    },
    {
        "name": "Manager-Data-01",
        "agent_id": "Mgr-Data-01",
        "port": 6145,
        "parent": "Dir-Data",
        "lane": "Data",
        "env_prefix": "MGR_DATA_01",
        "dir": "manager_data_01"
    },
    {
        "name": "Manager-DevSecOps-01",
        "agent_id": "Mgr-DevSecOps-01",
        "port": 6146,
        "parent": "Dir-DevSecOps",
        "lane": "DevSecOps",
        "env_prefix": "MGR_DEVSECOPS_01",
        "dir": "manager_devsecops_01"
    },
    {
        "name": "Manager-Docs-01",
        "agent_id": "Mgr-Docs-01",
        "port": 6147,
        "parent": "Dir-Docs",
        "lane": "Docs",
        "env_prefix": "MGR_DOCS_01",
        "dir": "manager_docs_01"
    }
]

for mgr in managers:
    print(f"Creating {mgr['name']}...")

    # Replace template placeholders
    content = template
    content = content.replace("Manager-Code-01", mgr["name"])
    content = content.replace("Mgr-Code-01", mgr["agent_id"])
    content = content.replace("Dir-Code", mgr["parent"])
    content = content.replace("6141", str(mgr["port"]))
    content = content.replace('"Code"', f'"{mgr["lane"]}"')
    content = content.replace("MGR_CODE_01", mgr["env_prefix"])

    # Write to new file
    output_dir = Path(f"services/pas/{mgr['dir']}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "app.py"
    output_file.write_text(content)
    output_file.chmod(0o755)

    print(f"  ✓ Created {output_file} (port {mgr['port']})")

print("\n✅ All Manager services created!")
print("\nCreated:")
for mgr in managers:
    print(f"  - {mgr['agent_id']:20} (port {mgr['port']})")
